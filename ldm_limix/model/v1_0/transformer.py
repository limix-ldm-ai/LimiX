import nvtx
import gc
import torch
import torch.nn as nn
from .layer import LayerStack
from typing import Any, Literal
from .encoders import get_x_encoder, get_cls_y_encoder, get_reg_y_encoder, preprocesss_4_x
from torch.amp import autocast
from typing import List
from .utils import SetRandomSeed, create_mlp_layer
from .autobatch import AutobatchConfig
from .operators.triton_rmsnorm import TritonRMSNorm, build_norm


def _contains_nan(t: torch.Tensor, chunk_elems: int = 1_048_576) -> bool:
    """Scan for NaNs without allocating a full-size boolean mask."""
    if t is None or t.numel() == 0:
        return False
    flat = t.reshape(-1)
    for i in range(0, flat.numel(), chunk_elems):
        if torch.isnan(flat[i:i + chunk_elems]).any():
            return True
    return False


def make_feature_padding_mask(real_feature_nums: List[int], padded_feature_nums: int, device: torch.device) -> torch.Tensor:
    """Build a padding mask for variable-length feature sequences.

    Args:
        real_feature_nums: Per-sample true feature counts, length batch_size.
        padded_feature_nums: Padded feature count per sample.
        device: Compute device, e.g. 'cpu' or 'cuda'.

    Returns:
        mask: Feature padding mask of shape (batch_size, padded_feature_nums).
    """
    real_feature_nums_tensor = torch.tensor(real_feature_nums)

    indices = torch.arange(padded_feature_nums).unsqueeze(0)  # shape: (1, max_len)

    thresholds = real_feature_nums_tensor.unsqueeze(1)  # shape: (len, 1)

    mask = indices < thresholds  # shape: (len, max_len)
    mask = mask.to(device)

    return mask


class FeaturesTransformer(nn.Module):
    def __init__(
                self,
                *,
                preprocess_config_x:dict[str, Any],
                encoder_config_x:dict[str, Any],
                encoder_config_y:dict[str, Any],
                decoder_config:dict[str, Any],
                feature_positional_embedding_type:Literal['subspace','subortho','none'],
                nlayers:int,
                nhead: int, 
                embed_dim: int, 
                hid_dim:int,
                mask_feature_embedding_type:Literal['mask_embedding']='mask_embedding',
                enable_mask_feature_pred: bool = False,
                enable_mask_indicator_pred: bool = False,
                features_per_group:int = 2,
                pre_norm: bool=False,
                device: torch.device|None=None,
                dtype: torch.dtype|None=None,
                init_seed_dict: dict = {},
                **kwargs:Any,
                ):
        super().__init__()

        print(f"use limix v1.0")

        self.preprocess_config_x = preprocess_config_x
        self.encoder_config_x = encoder_config_x
        self.encoder_config_y = encoder_config_y
        self.decoder_config = decoder_config
        self.feature_positional_embedding_type = feature_positional_embedding_type
        self.nlayers = nlayers
        self.nhead = nhead
        self.embed_dim = embed_dim
        self.hid_dim = hid_dim
        self.mask_feature_embedding_type = mask_feature_embedding_type
        self.enable_mask_feature_pred = enable_mask_feature_pred
        self.enable_mask_indicator_pred = enable_mask_indicator_pred
        self.features_per_group = features_per_group
        self.device = device
        self.dtype = dtype
        self.pre_norm = pre_norm
        self.config_dict = kwargs
        self.init_seed_dict = init_seed_dict

        self.enable_classification_y_pred = self.config_dict.get('enable_classification_y_pred', True)
        self.enable_regression_y_pred = self.config_dict.get('enable_regression_y_pred', False)
        self.num_buckets = self.config_dict.get('num_buckets', 0)
        self.profile_stage = self.config_dict.get('profile_stage', False)
        self.decoder_dropout = self.config_dict.get('decoder_dropout', 0.0)
        self.layer_recompute = self.config_dict.get('layer_recompute', False)
        self.use_rmsnorm = self.config_dict.get('use_rmsnorm', False)

        self.model_structure_config = self.config_dict.get('model_structure_config', None)
        if self.model_structure_config is None:
            raise ValueError("model_structure_config未在kwargs中找到")
        self.nlayers = self.model_structure_config['nlayers']

        assert self.enable_classification_y_pred or self.enable_regression_y_pred, "You must enable at least one: enable_classification_y_pred or enable_regression_y_pred"
        self.get_encoders(encoder_config_x, encoder_config_y)
        
        self.transformer_encoder = LayerStack(
            device=self.device,
            dtype=self.dtype,
            init_seed_dict=init_seed_dict,
            residual_scale=1.0,
            **kwargs,
        )
        
        with SetRandomSeed(self.init_seed_dict.get("transformer_encoder_output_layernor_seed", None)):
            encoder_out_norm_use_affine = self.config_dict.get('encoder_out_norm_use_affine', False)
            if self.config_dict.get('use_rmsnorm', False):
                self.encoder_out_norm = TritonRMSNorm(hs=self.embed_dim, eps=1e-5, elementwise_affine=encoder_out_norm_use_affine, recompute=self.layer_recompute) if pre_norm else nn.Identity()
            else:
                self.encoder_out_norm = nn.LayerNorm(self.embed_dim, eps=1e-5, elementwise_affine=encoder_out_norm_use_affine) if pre_norm else nn.Identity()
            
            self.feature_encoder_out_norm = self.encoder_out_norm
            self.reg_y_encoder_out_norm = self.encoder_out_norm
            self.cls_y_encoder_out_norm = self.encoder_out_norm
            self.last_layer_idx = self.nlayers-1

        with SetRandomSeed(self.init_seed_dict.get("decoder_cls_y_seed", None)):
            self.make_classification_y_decoder()
        with SetRandomSeed(self.init_seed_dict.get("decoder_reg_y_seed", None)):
            self.make_regression_y_decoder()
        with SetRandomSeed(self.init_seed_dict.get("decoder_feature_seed", None)):
            self.make_feature_decoder()
            
        with SetRandomSeed(self.init_seed_dict.get("feature_positional_embedding_seed", None)):
            if feature_positional_embedding_type in ("subspace", "subortho"):
                self.feature_positional_embedding = nn.Linear(self.embed_dim // 4, self.embed_dim)

        self.x_preprocess = preprocesss_4_x(**preprocess_config_x)
        self.share_model_module()

    def share_model_module(self):
        if self.config_dict.get('use_shared_encoder', False):
            shared_none_embedding = self.reg_y_encoder[0].none_embedding
            shared_numeric_encoder = self.reg_y_encoder[0].numeric_encoder
            del self.encoder_x[0].mask_embedding
            del self.encoder_x[0].numeric_mlp
            self.encoder_x[0].add_module("mask_embedding", shared_none_embedding)
            self.encoder_x[0].add_module("numeric_mlp", shared_numeric_encoder)


    def get_encoders(self, encoder_config_x, encoder_config_y):
        extra = {
            'use_rmsnorm': self.use_rmsnorm,
            'layer_recompute': self.layer_recompute,
        }
        self.encoder_x = get_x_encoder(**{**encoder_config_x, **extra})
        if self.enable_classification_y_pred:
            self.cls_y_encoder = get_cls_y_encoder(**{**encoder_config_y, **extra})
        if self.enable_regression_y_pred:
            self.reg_y_encoder = get_reg_y_encoder(**{**encoder_config_y, **extra})

    def make_classification_y_decoder(self):
        if not self.enable_classification_y_pred:
            return
        
        if 'original' == self.config_dict.get('cls_y_decoder_type', 'original'):
            self.cls_y_decoder = nn.Sequential(
                nn.Linear(self.embed_dim, self.hid_dim),
                nn.GELU(),
                nn.Linear(self.hid_dim, self.decoder_config['num_classes']),
            )
        elif 'Free_MLP' == self.config_dict.get('cls_y_decoder_type', 'original'):
            mlp_pattern = self.config_dict.get('cls_y_decoder_mlp_pattern', 'linear-768_GELU_linear-10')
            self.cls_y_decoder = create_mlp_layer(
                mlp_type=self.config_dict.get('cls_y_decoder_type', 'original'),
                input_dim_size=self.embed_dim,
                hidden_size=self.config_dict.get('decoder_hidden_dim', 192 * 4),
                output_dim_size=self.decoder_config['num_classes'],
                dropout=self.decoder_dropout,
                bias=True,
                mlp_pattern=mlp_pattern,
                use_rmsnorm=self.use_rmsnorm,
                layer_recompute=self.layer_recompute,
            )
        elif 'MLP_' in self.config_dict.get('cls_y_decoder_type', 'original'):
            self.cls_y_decoder = create_mlp_layer(
                mlp_type=self.config_dict.get('cls_y_decoder_type', 'original'),
                input_dim_size=self.embed_dim,
                hidden_size=self.config_dict.get('decoder_hidden_dim', 192 * 4),
                output_dim_size=self.decoder_config['num_classes'],
                dropout=self.decoder_dropout,
                bias=True,
                use_rmsnorm=self.use_rmsnorm,
                layer_recompute=self.layer_recompute,
            )
        else:
            raise ValueError(f"Unknown cls_y_decoder_type: {self.config_dict.get('cls_y_decoder_type', 'original')}")


    def make_regression_y_decoder(self):
        if not self.enable_regression_y_pred:
            return

        input_dim_size = self.embed_dim
        if self.num_buckets > 1:
            self.reg_y_decoder = nn.Sequential(
                nn.Linear(input_dim_size, self.hid_dim),
                nn.GELU(),
                nn.Linear(self.hid_dim, self.num_buckets),
            )
            # Placeholder slots; checkpoint state_dict overwrites the values.
            self.register_buffer("_reg_borders", torch.zeros(self.num_buckets + 1))
            self.register_buffer("reg_log_widths", torch.zeros(self.num_buckets))
        elif 'original' == self.config_dict.get('reg_y_mse_decoder_type', 'original'):
            self.reg_y_decoder = nn.Sequential(
                nn.Linear(input_dim_size, self.hid_dim),
                build_norm(self.hid_dim, use_rmsnorm=self.use_rmsnorm, recompute=self.layer_recompute),
                nn.GELU(),
                nn.Linear(self.hid_dim, 1),
            )
        elif 'MLP_' in self.config_dict.get('reg_y_mse_decoder_type', 'MLP_RELU'):
            self.reg_y_decoder = create_mlp_layer(
                mlp_type=self.config_dict.get('reg_y_mse_decoder_type', 'MLP_RELU'),
                input_dim_size=input_dim_size,
                hidden_size=self.config_dict.get('decoder_hidden_dim', 192 * 4),
                output_dim_size=1,
                dropout=self.decoder_dropout,
                bias=True,
                use_rmsnorm=self.use_rmsnorm,
                layer_recompute=self.layer_recompute,
            )
        else:
            raise ValueError(f"unkown reg_y_mse_decoder_type: {self.config_dict.get('reg_y_mse_decoder_type')}")


    def make_feature_decoder(self):
        if not self.enable_mask_feature_pred:
            return
        if 'original' == self.config_dict.get('feature_mse_decoder_type', 'original'):
            self.feature_decoder = nn.Sequential(
                nn.Linear(self.embed_dim, self.hid_dim),
                build_norm(self.hid_dim, use_rmsnorm=self.use_rmsnorm, recompute=self.layer_recompute),
                nn.GELU(),
                nn.Linear(self.hid_dim, self.features_per_group),
            )
        elif 'MLP_' in self.config_dict.get('feature_mse_decoder_type', 'MLP_RELU'):
            self.feature_decoder = create_mlp_layer(
                mlp_type=self.config_dict.get('feature_mse_decoder_type', 'MLP_RELU'),
                input_dim_size=self.embed_dim,
                hidden_size=self.config_dict.get('decoder_hidden_dim', 192 * 4),
                output_dim_size=self.features_per_group,
                dropout=self.decoder_dropout,
                bias=True,
                use_rmsnorm=self.use_rmsnorm,
                layer_recompute=self.layer_recompute,
            )
        else:
            raise ValueError(f"Unknown feature_decoder_type: {self.config_dict.get('feature_mse_decoder_type')}")

    def _apply_out_norm_seq(self, norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Apply LN along seq and write back, avoiding a second full-size encoder_out tensor."""
        if isinstance(norm, nn.Identity) or x.numel() == 0:
            return x
        if not AutobatchConfig.ENABLE_AUTOBATCH:
            y = norm(x)
            return y.to(dtype=x.dtype) if y.dtype != x.dtype else y
        from .autobatch import _cuda_free_bytes, _is_retryable_cuda_error
        seq = int(x.shape[1])
        per = max((x.numel() // max(seq, 1)) * x.element_size() * 8, 1)
        free = _cuda_free_bytes(x.device)
        chunk = seq if free is None else max(1, min(seq, int(0.25 * free / per)))
        if chunk >= seq:
            y = norm(x)
            return y.to(dtype=x.dtype) if y.dtype != x.dtype else y
        start = 0
        while start < seq:
            end = min(start + chunk, seq)
            try:
                sl = x[:, start:end]
                h = sl if sl.is_contiguous() else sl.contiguous()
                y = norm(h)
                sl.copy_(y.to(dtype=x.dtype))
                del y, h
                start = end
            except Exception as e:
                if not _is_retryable_cuda_error(e) or chunk == 1:
                    raise
                print(f"encoder-out LN chunk_size={chunk} OOM, retry with half")
                chunk = max(1, chunk // 2)
                gc.collect()
                if x.device.type == 'cuda':
                    torch.cuda.empty_cache()
        return x


    def forward(self, x: torch.Tensor, 
                y: torch.Tensor, 
                eval_pos: int, 
                x_mask: torch.Tensor = None,
                y_type: torch.Tensor = None,
                x_categorical_mask: torch.Tensor = None,
                real_feature_nums: List = None,
                task_type: Literal["Classification", "Feature_imputation", "Regression"] = "Classification",
                calculate_sample_attention: bool = False,
                calculate_feature_attention: bool = False,
                **layer_kwargs
                ) -> torch.Tensor | dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        '''
            x: The input x, which includes both train x and test x, Shape: [batch, sequence, feature]
            y: The input y, which includes both train y and test y, Shape: [batch, label]
            eval_pos: Train x and train y split point
            x_mask: Mask the specified features in x, Used during training
            task_type: Type of task, options: cls(classification), reg(regression)
        '''
        assert x is not None and y is not None, "x and y must not be none"
        assert eval_pos > 0, "eval_pos must be a positive number"
        assert len(x.shape)==3, "x must be [Batch, seq, Feature]"
        assert len(y.shape)==2, "y must be [Batch, label]"
        assert eval_pos < x.shape[1] and eval_pos <= y.shape[1], "The split point between train x and test x must be less than the feature dimension of x, and less than or equal to the label dimension of y"
        
        if task_type == "Regression":
            assert self.enable_regression_y_pred, "Regression mode requires enable_regression_y_pred=True at model initialization"
        if x_mask is None:
            x_mask = torch.isnan(x).to(torch.int32).to(x.device)

        batch_size, seq_len, num_feature = x.shape
        x = {'data':x, 'mask':x_mask}
        y = {'data':y}
        with nvtx.annotate('encoder'):
            with nvtx.annotate('x-encoder-all'):
                x, y, feature_to_add = self.padding_xy(x, y)
                if real_feature_nums is None:
                    real_feature_nums = [num_feature] * batch_size
                
                feature_padding_mask = make_feature_padding_mask(real_feature_nums, padded_feature_nums=x['data'].shape[2], device=x['data'].device)

                for k in x:
                    x[k] = x[k].reshape(batch_size, seq_len, x[k].shape[2]//self.features_per_group, self.features_per_group)
                x['eval_pos'] = eval_pos

                extra_encoders_args = {}
                extra_encoders_args['feature_padding_mask'] = feature_padding_mask
                x_emb_result, real_x = self.x_encoder(x, **extra_encoders_args)
            
            with nvtx.annotate('y-encoder'):
                y["data"][:, eval_pos:] = torch.nan

                _enable_cls_y_pred = False
                _enable_reg_y_pred = False
                _enable_mask_feature_pred = False
                if task_type == "Classification":
                    y_type = torch.zeros_like(y['data'], device=y['data'].device)
                    _enable_cls_y_pred = True
                elif task_type == "Regression":
                    y_type = torch.ones_like(y['data'], device=y['data'].device)
                    _enable_reg_y_pred = True
                elif task_type == "Feature_imputation":
                    y_type = torch.ones_like(y['data'], device=y['data'].device)
                    _enable_mask_feature_pred = True
                    _enable_reg_y_pred = True
                else:
                    raise ValueError(f"Unsupported task_type: {task_type}")
                y_type = y_type.squeeze(-1)

                embedded_y = self.y_encode(y, seq_len, batch_size, eval_pos, y_type=y_type,
                    _enable_cls_y_pred=_enable_cls_y_pred, _enable_reg_y_pred=_enable_reg_y_pred)

            with nvtx.annotate('add-embeddings'):
                n_groups = x['mask'].shape[2]
                y_tok = embedded_y.unsqueeze(2).to(dtype=x_emb_result.dtype)
                if AutobatchConfig.ENABLE_AUTOBATCH and x_emb_result.shape[2] == n_groups + 1:
                    self.add_embeddings(x_emb_result[:, :, :n_groups])
                    x_emb_result[:, :, n_groups:n_groups + 1] = y_tok
                    embedded_all = x_emb_result
                    embedded_x = x_emb_result[:, :, :n_groups]
                else:
                    embedded_x = self.add_embeddings(x_emb_result)
                    embedded_all = torch.cat((embedded_x, y_tok), dim=2)
                if not AutobatchConfig.ENABLE_AUTOBATCH:
                    if _contains_nan(embedded_x) or _contains_nan(embedded_y):
                        raise ValueError("embedded_all contains NaN values; please add a NanEncoder in the encoder")
                
                feature_mask = None
                del embedded_x, embedded_y, y_tok
                if embedded_all is not x_emb_result:
                    del x_emb_result
                del x['data']

        with nvtx.annotate('transformer-encoder'):
            if calculate_sample_attention or calculate_feature_attention:
                return self.transformer_encoder(embedded_all, feature_mask=feature_mask, eval_pos=eval_pos,
                                                calculate_sample_attention=calculate_sample_attention,
                                                calculate_feature_attention=calculate_feature_attention, **layer_kwargs)
            encoder_out_total = self.transformer_encoder(embedded_all, 
                                                         feature_attention_mask=None, 
                                                         eval_pos=eval_pos, 
                                                         y_type=y_type, 
                                                         **layer_kwargs)
            encoder_out = encoder_out_total[0]
            feature_emb = encoder_out
            reg_y_emb = encoder_out
            cls_y_emb = encoder_out
            
            del encoder_out_total
        with nvtx.annotate('encoder-norm'):
            same_src = feature_emb is encoder_out and reg_y_emb is encoder_out and cls_y_emb is encoder_out
            same_norm = (
                self.feature_encoder_out_norm is self.reg_y_encoder_out_norm
                and self.reg_y_encoder_out_norm is self.cls_y_encoder_out_norm
            )
            if AutobatchConfig.ENABLE_AUTOBATCH and same_src and same_norm:
                encoder_out = self._apply_out_norm_seq(self.encoder_out_norm, encoder_out)
                feature_emb = reg_y_emb = cls_y_emb = encoder_out
            else:
                if AutobatchConfig.ENABLE_AUTOBATCH:
                    if _enable_mask_feature_pred:
                        feature_emb = self._apply_out_norm_seq(self.feature_encoder_out_norm, feature_emb)
                    if _enable_reg_y_pred:
                        reg_y_emb = self._apply_out_norm_seq(self.reg_y_encoder_out_norm, reg_y_emb)
                    if _enable_cls_y_pred:
                        cls_y_emb = self._apply_out_norm_seq(self.cls_y_encoder_out_norm, cls_y_emb)
                else:
                    feature_emb = self.feature_encoder_out_norm(feature_emb)
                    reg_y_emb = self.reg_y_encoder_out_norm(reg_y_emb)
                    cls_y_emb = self.cls_y_encoder_out_norm(cls_y_emb)
        
        with nvtx.annotate('decoder'):
            if _enable_cls_y_pred :
                with nvtx.annotate('cls_y-decoder'):
                    cls_output = self.decoder_for_clasification_task(cls_y_emb, eval_pos, y_type, y_true=y["data"])
            if _enable_reg_y_pred:
                with nvtx.annotate('reg_y-decoder'):
                    reg_output = self.decoder_for_regression_task(reg_y_emb, eval_pos, y_type, y_true=y["data"])
            if _enable_mask_feature_pred:
                with nvtx.annotate('feature_reconstruct-decoder'):
                    feature_pred = self.decoder_for_feature_reconstruct(feature_emb, real_x, x['mask'])
                    mask_feature_pred = {}
                    if self.training:
                        mask_feature_pred["real_x"] = real_x
                    if feature_pred is not None:
                        mask_feature_pred['feature_pred'] = feature_pred
            
            output_decoded = {}
            if _enable_mask_feature_pred:
                output_decoded.update(mask_feature_pred)
                output_decoded["feature_process_config"] = {
                    "n_x_padding": feature_to_add,
                    "features_per_group": self.x_preprocess.modules_dict['valid_feature_encoder'].num_features,
                    "num_used_features": self.x_preprocess.modules_dict['valid_feature_encoder'].valid_feature_num,
                    "mean_for_normalization": self.x_preprocess.modules_dict['normalization_encoder'].mean,
                    "std_for_normalization": self.x_preprocess.modules_dict['normalization_encoder'].std,
                }
            if _enable_cls_y_pred:
                output_decoded.update(cls_output)    
            if _enable_reg_y_pred:
                output_decoded.update(reg_output)
        
        return output_decoded


    def padding_xy(self, x:dict, y:dict):
        '''Pad features to a multiple of features_per_group. Training data is already padded; this is needed for inference.'''
        batch_size, seq_len, num_feature = x['data'].shape
        feature_to_add = (self.features_per_group - num_feature % self.features_per_group) % self.features_per_group
        if feature_to_add > 0:
            # Extend the feature dimension of x when it is insufficient
            for k in x:
                x[k] = torch.cat(
                    (
                        x[k],
                        torch.zeros(
                            batch_size,
                            seq_len,
                            feature_to_add,
                            device=x[k].device,
                            dtype=x[k].dtype
                        )
                    ),
                    dim=-1
                )
        for k in y:
            # Extend the label dimension of y when it is insufficient
            y[k] = y[k].unsqueeze(-1)
            if y[k].shape[1] < x['data'].shape[1]:
                y[k] = torch.cat(
                    (
                        y[k],
                        torch.nan
                        * torch.zeros(
                            y[k].shape[0],
                            x["data"].shape[1] - y[k].shape[1],
                            y[k].shape[2],
                            device=y[k].device,
                            dtype=y[k].dtype,
                        ),
                    ),
                    dim=1
                )
        return x, y, feature_to_add
    
    def x_encoder(self, x:dict, **kwargs):
        feature_padding_mask = kwargs.get('feature_padding_mask', None)

        with nvtx.annotate('x-preprocess'):
            preprocessed_x = self.x_preprocess(x)
            real_x = preprocessed_x['data'].clone().detach()
            if self.enable_mask_feature_pred:
                preprocessed_x = self.mask_process_4_x(preprocessed_x)

        with nvtx.annotate('x-encoder'):
            preprocessed_x['feature_padding_mask'] = feature_padding_mask
            x_encoder_result = self.encoder_x(preprocessed_x)
        x_emb_result = x_encoder_result['data']

        return x_emb_result, real_x

    def y_encode(self, y, seq_len, batch_size, eval_pos, y_type:torch.Tensor, _enable_cls_y_pred:bool, _enable_reg_y_pred:bool):
        if _enable_cls_y_pred:
            with nvtx.annotate('cls_y-encoder'):
                cls_y_mask = (y_type == 0)
                cls_y_mask = cls_y_mask.unsqueeze(2)
                cls_y = {k: v.clone().detach() for k, v in y.items()}
                cls_y['eval_pos'] = eval_pos
                cls_y['data'] = torch.where(cls_y_mask, y['data'], 0)
                cls_embedded_y = self.cls_y_encoder(cls_y)
                cls_embedded_y = cls_embedded_y['data']
        
        if _enable_reg_y_pred:
            with nvtx.annotate('reg_y-encoder'):
                reg_y_mask = (y_type == 1)
                reg_y_mask = reg_y_mask.unsqueeze(2)
                reg_y = {k: v.clone().detach() for k, v in y.items()}
                reg_y['data'] = torch.where(reg_y_mask, y['data'], 0)
                reg_y['eval_pos'] = eval_pos
                reg_embedded_y = self.reg_y_encoder(reg_y)
                reg_embedded_y: torch.Tensor = reg_embedded_y['data']

        if _enable_cls_y_pred and _enable_reg_y_pred:
            flat_y_type = y_type.reshape(-1)
            flat_cls_embedded_y = cls_embedded_y.reshape(-1, self.embed_dim)
            flat_reg_embedded_y = reg_embedded_y.reshape(-1, self.embed_dim)
            flat_embedded_y = torch.empty(seq_len * batch_size, self.embed_dim)
            flat_y_type = flat_y_type.to(torch.bool)
            flat_y_type = flat_y_type.unsqueeze(dim=-1)
            flat_embedded_y = torch.where(flat_y_type, flat_reg_embedded_y, flat_cls_embedded_y)
            embedded_y = flat_embedded_y.reshape(batch_size, seq_len, -1)
        elif _enable_cls_y_pred:
            embedded_y = cls_embedded_y.reshape(batch_size, seq_len, -1)
        elif _enable_reg_y_pred:
            embedded_y = reg_embedded_y.reshape(batch_size, seq_len, -1)

        if torch.isnan(embedded_y).any():
            raise ValueError(
                f"{torch.isnan(embedded_y).any()=}, make sure to add nan handlers"
                " to the ys that are not fully provided (test set missing)",
            )

        return embedded_y


    def mask_process_4_x(self, data:dict):
        x_input = data['data']
        mask = data['mask']
        x_feature_mean = torch.nanmean(x_input, dim=(1), keepdim=True)
        x_feature_mean = torch.where(torch.isnan(x_feature_mean), 0, x_feature_mean)
        x_input = torch.where(mask==1, float('nan'), x_input)
        x_input = torch.where(mask==2, x_feature_mean, x_input)
        x_input = torch.where(mask==3, x_input, x_input)
        x_input = torch.where(mask==4, x_input + torch.randn_like(x_input)*0.01, x_input)
        data['data'] = x_input
        data['mask'] = mask.to(torch.bool)
        return data
    
    def add_embeddings(self, x:torch.Tensor):
        if self.feature_positional_embedding_type == "subspace":
            embs = torch.randn(
                (x.shape[2], x.shape[3] // 4),
                device=x.device,
                dtype=x.dtype,
            )
            embs = self.feature_positional_embedding(embs)
            x += embs[None, None]
        elif self.feature_positional_embedding_type == "subortho":
            with autocast(device_type=x.device.type, enabled=False):
                embs = torch.randn(
                    (x.shape[2], x.shape[3] // 4),
                    device=x.device,
                    dtype=torch.float32,
                )
                torch.nn.init.orthogonal_(embs)
            embs =self.feature_positional_embedding(embs.to(x.dtype))
            x += embs[None, None]
        elif self.feature_positional_embedding_type is None or self.feature_positional_embedding_type == "none":
            pass
        else:
            raise ValueError(f"Unknown feature_positional_embedding_type={self.feature_positional_embedding_type}")
        return x

    def decoder_for_clasification_task(self, encoder_out, eval_pos, y_type, y_true):
        num_target_tokens = self.config_dict.get('num_target_tokens', 0)
        test_encoder_out = encoder_out[:, eval_pos:, -(num_target_tokens+1), :]
        test_y_type = y_type[:, eval_pos:]
        cls_mask = (test_y_type == 0)
        cls_mask = cls_mask.unsqueeze(2)
        zeros = torch.zeros_like(test_encoder_out)
        encoder_out_for_cls = torch.where(cls_mask, test_encoder_out, zeros)
        output = self.cls_y_decoder(encoder_out_for_cls)
        return {'cls_output': output}


    def decoder_for_regression_task(self, reg_y_emb, eval_pos, y_type, y_true):
        num_target_tokens = self.config_dict.get('num_target_tokens', 0)
        test_encoder_out = reg_y_emb[:, eval_pos:, -(num_target_tokens+1), :]
        test_y_type = y_type[:, eval_pos:]
        reg_mask = (test_y_type == 1)
        reg_mask = reg_mask.unsqueeze(2)
        zeros = torch.zeros_like(test_encoder_out)
        encoder_out_for_reg = torch.where(reg_mask, test_encoder_out, zeros)
        output_ = self.reg_y_decoder(encoder_out_for_reg)
        return {'reg_output': [output_]}

    
    def decoder_for_feature_reconstruct(self, encoder_out:torch.Tensor, real_x: torch.Tensor, x_mask: torch.Tensor):
        num_target_tokens = self.config_dict.get('num_target_tokens', 0)
        encoder_out = encoder_out[:, :, :-(num_target_tokens + 1), :]
        group_mask = x_mask.any(dim=-1)
        feature_pred = real_x.clone()
        masked_pred = self.feature_decoder(encoder_out[group_mask]).to(real_x.dtype)
        filled = torch.where(x_mask[group_mask], masked_pred, real_x[group_mask])
        feature_pred[group_mask] = filled
        return feature_pred
