import nvtx
import torch
import torch.nn as nn
from .layer import LayerStack
from typing import Any, Literal
from .encoders import get_x_encoder, get_cls_y_encoder, get_reg_y_encoder, preprocesss_4_x
from typing import List
from .utils import SetRandomSeed
from .utils import create_mlp_layer, AdapterWithResidual, create_mlp_4_free_type
from .operators.rmsnorm import build_rmsnorm
from .add_task_info import AddTaskInfo


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
                feature_positional_embedding_type:Literal['subspace'],
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
                **kwargs:Any,               # TODO: use a dataclass instead of unbounded kwargs
                ):
        super().__init__()
        print(f"use limix v2.0")

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
        self.pre_norm = bool(pre_norm)
        self.all_norm = bool(kwargs.get('all_norm', False))
        # Both pre-norm and all-norm require one final normalization after the
        # transformer layer stack. Post-norm does not.
        self.use_encoder_out_norm = self.pre_norm or self.all_norm
        self.config_dict = kwargs
        self.init_seed_dict = init_seed_dict
        self.y_token_k = int(self.config_dict.get('num_cls_tokens', 1))
        if self.y_token_k <= 0:
            raise ValueError("y_token_k must be greater than 0")
        self.y_encoder_embedding_size = self.embed_dim * self.y_token_k
        self.y_decoder_input_dim = self.y_encoder_embedding_size
        self.enable_classification_y_pred = self.config_dict.get('enable_classification_y_pred', True)
        self.enable_regression_y_pred = self.config_dict.get('enable_regression_y_pred', False)
        self.num_buckets = self.config_dict.get('num_buckets', 0)
        self.decoder_dropout = self.config_dict.get('decoder_dropout', 0.0)
        self.enable_add_task_info = self.config_dict.get('add_task_type', False)
        self.layer_recompute = self.config_dict.get('layer_recompute', False)
        self.rmsnorm_impl = self.config_dict.get('rmsnorm_impl', 'triton')

        self.model_structure_config = self.config_dict.get('model_structure_config', None)
        if self.model_structure_config is None:
            raise ValueError("model_structure_config未在kwargs中找到，请确保在st_train.py中已正确解析YAML配置")

        self.nlayers = self.model_structure_config['nlayers']
        self.uses_decoupled_feature_attention = (
            self.config_dict.get('feature_attention_pipeline', 'legacy') == 'decoupled'
            or any(
                layer_config.get('feature_attention_pipeline') == 'decoupled'
                for layer_config in self.model_structure_config['layers']
            )
        )
        self.last_layer_idx = self.nlayers - 1
        cls_only_start_layer = self.config_dict.get('cls_only_start_layer', -1)
        cls_only_start_layer = -1 if cls_only_start_layer is None else int(cls_only_start_layer)
        if cls_only_start_layer < -1 or cls_only_start_layer > self.last_layer_idx:
            raise ValueError(
                "cls_only_start_layer must be -1 (disabled) or a valid zero-based "
                f"layer index in [0, {self.last_layer_idx}], got {cls_only_start_layer}"
            )
        self.cls_only_start_layer = cls_only_start_layer
        self.config_dict['cls_only_start_layer'] = cls_only_start_layer
        for output_layer_key in ('feature_emb_layer', 'reg_y_emb_layer', 'cls_y_emb_layer'):
            output_layer_idx = self.config_dict.get(output_layer_key, self.last_layer_idx)
            if output_layer_idx is None or output_layer_idx == -1:
                output_layer_idx = self.last_layer_idx
            if not 0 <= output_layer_idx <= self.last_layer_idx:
                raise ValueError(
                    f"{output_layer_key}={output_layer_idx} is outside the valid range "
                    f"[0, {self.last_layer_idx}] for nlayers={self.nlayers}"
                )
            self.config_dict[output_layer_key] = output_layer_idx

        feature_output_enabled = self.enable_mask_feature_pred
        if (
            feature_output_enabled
            and self.cls_only_start_layer >= 0
            and self.config_dict['feature_emb_layer'] >= self.cls_only_start_layer
        ):
            raise ValueError(
                "Feature reconstruction needs an X-bearing feature_emb_layer before "
                "cls_only_start_layer; set feature_emb_layer to at most "
                f"{self.cls_only_start_layer - 1}"
            )

        assert self.enable_classification_y_pred or self.enable_regression_y_pred, "You must enable at least one: enable_classification_y_pred or enable_regression_y_pred"
        self.get_encoders(encoder_config_x, encoder_config_y)

        self.transformer_encoder = LayerStack(
            device=self.device,
            dtype=self.dtype,
            init_seed_dict=init_seed_dict,
            **kwargs,
        )

        with SetRandomSeed(self.init_seed_dict.get("transformer_encoder_output_layernor_seed", None)):
            encoder_out_norm_use_affine = self.config_dict.get('encoder_out_norm_use_affine', False)
            self.encoder_out_norm = build_rmsnorm(self.embed_dim, eps=1e-5, elementwise_affine=encoder_out_norm_use_affine, norm_impl=self.rmsnorm_impl, recompute=self.layer_recompute) if self.use_encoder_out_norm else nn.Identity()

            self.feature_encoder_out_norm = self.encoder_out_norm
            self.reg_y_encoder_out_norm = self.encoder_out_norm
            self.cls_y_encoder_out_norm = self.encoder_out_norm
            feature_output_enabled = self.enable_mask_feature_pred
            active_output_layers = []
            if feature_output_enabled:
                active_output_layers.append(self.config_dict.get('feature_emb_layer', self.last_layer_idx))
            if self.enable_regression_y_pred:
                active_output_layers.append(self.config_dict.get('reg_y_emb_layer', self.last_layer_idx))
            if self.enable_classification_y_pred:
                active_output_layers.append(self.config_dict.get('cls_y_emb_layer', self.last_layer_idx))
            if self.last_layer_idx not in active_output_layers:
                # Keep checkpoint keys compatible, but do not expose a norm that no loss can reach to DDP.
                self.encoder_out_norm.requires_grad_(False)

            if self.config_dict.get('feature_emb_layer', self.last_layer_idx) < self.last_layer_idx:
                self.feature_encoder_out_norm = build_rmsnorm(self.embed_dim, eps=1e-5, elementwise_affine=encoder_out_norm_use_affine, norm_impl=self.rmsnorm_impl, recompute=self.layer_recompute) if self.use_encoder_out_norm else nn.Identity()
                if not feature_output_enabled:
                    self.feature_encoder_out_norm.requires_grad_(False)
            if self.config_dict.get('reg_y_emb_layer', self.last_layer_idx) < self.last_layer_idx:
                self.reg_y_encoder_out_norm = build_rmsnorm(self.embed_dim, eps=1e-5, elementwise_affine=encoder_out_norm_use_affine, norm_impl=self.rmsnorm_impl, recompute=self.layer_recompute) if self.use_encoder_out_norm else nn.Identity()
                if not self.enable_regression_y_pred:
                    self.reg_y_encoder_out_norm.requires_grad_(False)
            if self.config_dict.get('cls_y_emb_layer', self.last_layer_idx) < self.last_layer_idx:
                self.cls_y_encoder_out_norm = build_rmsnorm(self.embed_dim, eps=1e-5, elementwise_affine=encoder_out_norm_use_affine, norm_impl=self.rmsnorm_impl, recompute=self.layer_recompute) if self.use_encoder_out_norm else nn.Identity()
                if not self.enable_classification_y_pred:
                    self.cls_y_encoder_out_norm.requires_grad_(False)

        with SetRandomSeed(self.init_seed_dict.get("decoder_cls_y_seed", None)):
            self.make_classification_y_decoder()
        with SetRandomSeed(self.init_seed_dict.get("decoder_reg_y_seed", None)):
            self.make_regression_y_decoder()
        with SetRandomSeed(self.init_seed_dict.get("decoder_feature_seed", None)):
            self.make_feature_decoder()

        with SetRandomSeed(self.init_seed_dict.get("feature_positional_embedding_seed", None)):
            if feature_positional_embedding_type != "subspace":
                raise ValueError(
                    f"Unknown feature_positional_embedding_type={feature_positional_embedding_type}"
                )
            self.feature_positional_embedding = nn.Linear(self.embed_dim // 4, self.embed_dim)

        self.x_preprocess = preprocesss_4_x(**preprocess_config_x)

        with SetRandomSeed(self.init_seed_dict.get("reg_cls_embedding_projection_seed", None)):
            self.make_feature_post_adapter()
            self.make_cls_post_adapter()
            self.make_reg_post_adapter()

        with SetRandomSeed(self.init_seed_dict.get("add_task_info_seed", None)):
            if self.enable_add_task_info:
                self.add_task_info = AddTaskInfo(
                    embedding_size=self.embed_dim,
                    num_y_tokens=self.y_token_k,
                )

    def make_feature_post_adapter(self):
        """Build the feature post-processing adapter."""
        if not self.config_dict.get('use_feature_post_adapter', False):
            return

        self.feature_post_adapter = create_mlp_4_free_type(
            mlp_pattern = self.config_dict.get('feature_post_adapter_pattern', 'linear-192'),
            input_dim_size = self.embed_dim,
            output_dim_size = self.embed_dim,
            bias=True,
            norm_impl=self.rmsnorm_impl,
            recompute=self.layer_recompute,
        )
        self.feature_post_adapter = AdapterWithResidual(
            self.feature_post_adapter, self.embed_dim,
            use_residual=self.config_dict.get('use_feature_post_adapter_residual', False),
            norm_impl=self.rmsnorm_impl,
            recompute=self.layer_recompute,
        )
        if not self.enable_mask_feature_pred:
            self.feature_post_adapter.requires_grad_(False)


    def make_cls_post_adapter(self):
        """Build the classification post-processing adapter."""
        if not self.config_dict.get('use_cls_post_adapter', False):
            return

        self.cls_post_adapter = create_mlp_4_free_type(
            mlp_pattern = self.config_dict.get('cls_post_adapter_pattern', 'linear-192'),
            input_dim_size = self.y_decoder_input_dim,
            output_dim_size = self.y_decoder_input_dim,
            bias=True,
            y_token=self.y_token_k,
            norm_impl=self.rmsnorm_impl,
            recompute=self.layer_recompute,
        )
        self.cls_post_adapter = AdapterWithResidual(
            self.cls_post_adapter, self.y_decoder_input_dim,
            use_residual=self.config_dict.get('use_cls_post_adapter_residual', False),
            norm_impl=self.rmsnorm_impl,
            recompute=self.layer_recompute,
        )
        if not self.enable_classification_y_pred:
            self.cls_post_adapter.requires_grad_(False)


    def make_reg_post_adapter(self):
        if not self.config_dict.get('use_reg_post_adapter', False):
            return

        self.reg_post_adapter = create_mlp_4_free_type(
            mlp_pattern = self.config_dict.get('reg_post_adapter_pattern', 'linear-192'),
            input_dim_size = self.y_decoder_input_dim,
            output_dim_size = self.y_decoder_input_dim,
            bias=True,
            y_token=self.y_token_k,
            norm_impl=self.rmsnorm_impl,
            recompute=self.layer_recompute,
        )
        self.reg_post_adapter = AdapterWithResidual(
            self.reg_post_adapter, self.y_decoder_input_dim,
            use_residual=self.config_dict.get('use_reg_post_adapter_residual', False),
            norm_impl=self.rmsnorm_impl,
            recompute=self.layer_recompute,
        )
        if not self.enable_regression_y_pred:
            self.reg_post_adapter.requires_grad_(False)


    def get_encoders(self, encoder_config_x, encoder_config_y):
        encoder_config_x = dict(encoder_config_x)
        encoder_config_y = dict(encoder_config_y)
        encoder_config_x['rmsnorm_impl'] = self.rmsnorm_impl
        encoder_config_x['layer_recompute'] = self.layer_recompute
        encoder_config_y['rmsnorm_impl'] = self.rmsnorm_impl
        encoder_config_y['layer_recompute'] = self.layer_recompute
        encoder_config_y['embedding_size'] = self.y_encoder_embedding_size
        self.encoder_x = get_x_encoder(**encoder_config_x)
        if self.enable_classification_y_pred:
            self.cls_y_encoder = get_cls_y_encoder(**encoder_config_y)
        if self.enable_regression_y_pred:
            self.reg_y_encoder = get_reg_y_encoder(**encoder_config_y)

    def make_classification_y_decoder(self):
        if not self.enable_classification_y_pred:
            return
        if self.config_dict.get('cls_y_decoder_type', 'original') != 'Free_MLP':
            raise ValueError(
                f"Unknown cls_y_decoder_type: {self.config_dict.get('cls_y_decoder_type', 'original')}"
            )
        self.cls_y_decoder = create_mlp_layer(
            mlp_type='Free_MLP',
            input_dim_size=self.y_decoder_input_dim,
            hidden_size=self.config_dict.get('decoder_hidden_dim', self.y_decoder_input_dim * 4),
            output_dim_size=self.decoder_config['num_classes'],
            dropout=self.decoder_dropout,
            bias=True,
            mlp_pattern=self.config_dict.get(
                'cls_y_decoder_mlp_pattern', 'linear-768_GELU_linear-10'
            ),
            y_token=self.y_token_k,
            norm_impl=self.rmsnorm_impl,
            recompute=self.layer_recompute,
        )

    def make_regression_y_decoder(self):
        if not self.enable_regression_y_pred:
            return
        if self.num_buckets <= 1:
            raise ValueError("regression decoder requires num_buckets > 1")
        if self.config_dict.get('reg_y_decoder_type', 'original') != 'Free_MLP':
            raise ValueError(
                f"Unknown reg_y_decoder_type: {self.config_dict.get('reg_y_decoder_type', 'original')}"
            )
        self.reg_y_decoder = create_mlp_layer(
            mlp_type='Free_MLP',
            input_dim_size=self.y_decoder_input_dim,
            hidden_size=self.config_dict.get('decoder_hidden_dim', self.y_decoder_input_dim * 4),
            output_dim_size=self.num_buckets,
            dropout=self.decoder_dropout,
            bias=True,
            mlp_pattern=self.config_dict.get(
                'reg_y_decoder_mlp_pattern', 'linear-768_GELU_linear-10'
            ),
            y_token=self.y_token_k,
            norm_impl=self.rmsnorm_impl,
            recompute=self.layer_recompute,
        )
        # Placeholder slots; checkpoint state_dict overwrites the values.
        self.register_buffer("_reg_borders", torch.zeros(self.num_buckets + 1))
        self.register_buffer("reg_log_widths", torch.zeros(self.num_buckets))


    def make_feature_decoder(self):
        if not self.enable_mask_feature_pred:
            return
        feature_mse_decoder_type = self.config_dict.get(
            'feature_mse_decoder_type', 'original'
        )
        if 'MLP_' not in feature_mse_decoder_type:
            raise ValueError(f"Unknown feature_decoder_type: {feature_mse_decoder_type}")
        self.feature_decoder = create_mlp_layer(
            mlp_type=feature_mse_decoder_type,
            input_dim_size=self.embed_dim,
            hidden_size=self.config_dict.get('decoder_hidden_dim', 192 * 4),
            output_dim_size=self.features_per_group,
            dropout=self.decoder_dropout,
            bias=True,
            norm_impl=self.rmsnorm_impl,
            recompute=self.layer_recompute,
        )

    def _encoder(
        self,
        x: dict,
        y: dict,
        eval_pos: int,
        y_type: torch.Tensor = None,
        x_categorical_mask: torch.Tensor = None,
        real_feature_nums: List = None,
        task_type: Literal["Classification", "Feature_imputation", "Classification-Feature_imputation", "Regression", "Regression-Feature_imputation"] = "Classification",
        feature_positional_embedding_generator: torch.Generator | None = None,
    ):
        batch_size, seq_len, num_feature = x['data'].shape

        with nvtx.annotate('x-encoder-all'):
            x, y, feature_to_add = self.padding_xy(x, y)
            if real_feature_nums is None:
                real_feature_nums = [num_feature] * batch_size

            feature_padding_mask = make_feature_padding_mask(real_feature_nums, padded_feature_nums=x['data'].shape[2], device=x['data'].device)

            x_categorical_mask = None

            for k in x:
                x[k] = x[k].reshape(batch_size, seq_len, x[k].shape[2]//self.features_per_group, self.features_per_group)
            x['eval_pos'] = eval_pos

            extra_encoders_args = {}
            extra_encoders_args['x_categorical_mask'] = x_categorical_mask
            extra_encoders_args['feature_padding_mask'] = feature_padding_mask
            x_emb_result, real_x = self.x_encoder(x, **extra_encoders_args)

        # Mask the test y
        with nvtx.annotate('y-encoder'):
            y["data"][:, eval_pos:] = torch.nan

            if y_type is None:
                if task_type in ["Classification", "Classification-Feature_imputation"]:
                    y_type =  torch.zeros_like(y['data'], device=y['data'].device)
                elif task_type in ["Regression", "Regression-Feature_imputation"]:
                    y_type =  torch.ones_like(y['data'], device=y['data'].device)
                elif task_type in ["Feature_imputation"]:
                    y_type =  torch.ones_like(y['data'], device=y['data'].device)
                else:
                    raise ValueError(f"Unsupported task_type: {task_type}")
                y_type = y_type.squeeze(-1)

                enable_classification_y_pred_temp = False
                enable_regression_y_pred_temp = False
                enable_mask_feature_pred_temp = False
                if task_type in ["Classification", "Classification-Feature_imputation"]:
                    enable_classification_y_pred_temp = True
                if task_type in ["Regression", "Regression-Feature_imputation"]:
                    enable_regression_y_pred_temp = True
                if task_type in ["Feature_imputation", "Classification-Feature_imputation", "Regression-Feature_imputation"]:
                    enable_mask_feature_pred_temp = True
                    enable_regression_y_pred_temp = True
            else:
                enable_classification_y_pred_temp = self.enable_classification_y_pred
                enable_regression_y_pred_temp = self.enable_regression_y_pred
                enable_mask_feature_pred_temp = self.enable_mask_feature_pred

            embedded_y = self.y_encode(
                y,
                seq_len,
                batch_size,
                eval_pos,
                y_type=y_type,
                enable_classification_y_pred_temp=enable_classification_y_pred_temp,
                enable_regression_y_pred_temp=enable_regression_y_pred_temp,
            )

        with nvtx.annotate('add-embeddings'):
            embedded_x = self.add_embeddings(
                x_emb_result,
                generator=feature_positional_embedding_generator,
            )
            embedded_all = torch.cat((embedded_x, embedded_y), dim=2)
            if torch.isnan(embedded_all).any():
                raise ValueError("embedded_all contains NaN values; please add a NanEncoder in the encoder")

            x_feature_mask = None
            if self.config_dict.get('enable_feature_attention_mask', False):
                x_feature_mask = x['mask'].to(torch.bool)
                if x_feature_mask.dim() == 4:
                    x_feature_mask = x_feature_mask.any(dim=-1)

            if self.uses_decoupled_feature_attention:
                # DStI summaries must never aggregate padded feature
                # groups, even when value-level missing masks are disabled.
                grouped_feature_is_valid = feature_padding_mask.reshape(
                    batch_size, -1, self.features_per_group
                ).any(dim=-1)
                padded_group_mask = (~grouped_feature_is_valid).unsqueeze(1)
                padded_group_mask = padded_group_mask.expand(-1, seq_len, -1)
                x_feature_mask = (
                    padded_group_mask
                    if x_feature_mask is None
                    else x_feature_mask | padded_group_mask
                )

            if x_feature_mask is not None:
                # Task tokens are always valid queries; DStI slices the X
                # part before applying this cell/feature mask.
                y_feature_mask = torch.zeros(
                    *x_feature_mask.shape[:-1],
                    self.y_token_k,
                    device=x_feature_mask.device,
                    dtype=torch.bool,
                )
                feature_mask = torch.cat([x_feature_mask, y_feature_mask], dim=-1)
            else:
                feature_mask = None
            del embedded_x, embedded_y, x_emb_result
            del x['data']

        with nvtx.annotate('add-task-info'):
            if self.enable_add_task_info:
                embedded_all = self.add_task_info(embedded_all, y_type=y_type)

        # NOTE: after encoder, the transformer layer will convert output as dtype for autocast,
        #       so here pre-convert it to reduce memory footprint
        device_type = embedded_all.device.type
        dtype = torch.get_autocast_dtype(device_type) if torch.is_autocast_enabled() else torch.float32
        embedded_all = embedded_all.to(dtype)

        return embedded_all, feature_mask, y_type, real_x, feature_to_add, \
            enable_classification_y_pred_temp, enable_regression_y_pred_temp, \
            enable_mask_feature_pred_temp

    def forward(self, x: torch.Tensor,
                y: torch.Tensor,
                eval_pos: int,
                x_mask: torch.Tensor = None,
                y_type: torch.Tensor = None,
                x_categorical_mask: torch.Tensor = None,
                real_feature_nums: List = None,
                task_type: Literal["Classification", "Feature_imputation", "Classification-Feature_imputation", "Regression", "Regression-Feature_imputation"] = "Classification",
                calculate_sample_attention: bool = False,
                calculate_feature_attention: bool = False,
                x_encoder_adapter_input: Any = None,
                feature_positional_embedding_generator: torch.Generator | None = None,
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
        if task_type=="cls":
            task_type="Classification"
        elif task_type=="reg":
            task_type=="Regression"
        if task_type in ["Regression", "Regression-Feature_imputation"]:
            assert self.enable_regression_y_pred, "Regression mode requires enable_regression_y_pred=True at model initialization"
        if x_mask is None:
            x_mask = torch.isnan(x).to(torch.int32).to(x.device)
            if 'mean' == self.config_dict.get('mask_feature_filling_type', 'mask_token'):
                x_mask = x_mask * 2

        x = {'data':x, 'mask':x_mask}
        y = {'data':y}
        with nvtx.annotate('encoder'):
            embedded_all, feature_mask, y_type, real_x, feature_to_add, \
            enable_classification_y_pred_temp, enable_regression_y_pred_temp, \
            enable_mask_feature_pred_temp = self._encoder(
                x, y, eval_pos, y_type, x_categorical_mask, real_feature_nums, task_type,
                feature_positional_embedding_generator,
            )

        with nvtx.annotate('transformer-encoder'):
            if calculate_sample_attention or calculate_feature_attention:
                return self.transformer_encoder(embedded_all, feature_mask=feature_mask, eval_pos=eval_pos,
                                                calculate_sample_attention=calculate_sample_attention,
                                                calculate_feature_attention=calculate_feature_attention, **layer_kwargs)
            encoder_out_total = self.transformer_encoder(embedded_all,
                                                         feature_mask=feature_mask,
                                                         eval_pos=eval_pos,
                                                         y_type=y_type,
                                                         **layer_kwargs)
            encoder_out = encoder_out_total[0]
            feature_output_enabled = enable_mask_feature_pred_temp
            feature_emb = None
            if feature_output_enabled:
                feature_emb = encoder_out_total[3] if self.config_dict.get('feature_emb_layer', self.last_layer_idx) < self.last_layer_idx else encoder_out
            reg_y_emb = None
            if enable_regression_y_pred_temp:
                reg_y_emb = encoder_out_total[4] if self.config_dict.get('reg_y_emb_layer', self.last_layer_idx) < self.last_layer_idx else encoder_out
                reg_y_emb = reg_y_emb[:, eval_pos:, -self.y_token_k:, :]
            cls_y_emb = None
            if enable_classification_y_pred_temp:
                cls_y_emb = encoder_out_total[5] if self.config_dict.get('cls_y_emb_layer', self.last_layer_idx) < self.last_layer_idx else encoder_out
                cls_y_emb = cls_y_emb[:, eval_pos:, -self.y_token_k:, :]

            del encoder_out_total
        with nvtx.annotate('encoder-norm'):
            if feature_output_enabled:
                feature_emb = self.feature_encoder_out_norm(feature_emb)
            if enable_regression_y_pred_temp:
                reg_y_emb = self.reg_y_encoder_out_norm(reg_y_emb)
            if enable_classification_y_pred_temp:
                cls_y_emb = self.cls_y_encoder_out_norm(cls_y_emb)

        with nvtx.annotate('decoder'):
            if enable_classification_y_pred_temp :
                with nvtx.annotate('cls_y-decoder'):
                    cls_y_emb = self.flatten_y_tokens(cls_y_emb)
                    cls_output = self.decoder_for_clasification_task(cls_y_emb, eval_pos, y_type, y_true=y["data"])
            if enable_regression_y_pred_temp:
                with nvtx.annotate('reg_y-decoder'):
                    reg_y_emb = self.flatten_y_tokens(reg_y_emb)
                    reg_output = self.decoder_for_regression_task(reg_y_emb, eval_pos, y_type, y_true=y["data"])
            if enable_mask_feature_pred_temp:
                with nvtx.annotate('feature_reconstruct-decoder'):
                    feature_pred, categorical_feature_pred, numerical_feature_pred = self.decoder_for_feature_reconstruct(feature_emb, real_x, x['mask'])
                    mask_feature_pred = {}
                    if feature_pred is not None:
                        mask_feature_pred['feature_pred'] = feature_pred
                    if categorical_feature_pred is not None:
                        mask_feature_pred['categorical_feature_pred'] = categorical_feature_pred
                    if numerical_feature_pred is not None:
                        mask_feature_pred['numerical_feature_pred'] = numerical_feature_pred
            output_decoded = {}
            if enable_mask_feature_pred_temp:
                output_decoded.update(mask_feature_pred)
                output_decoded["feature_process_config"] = {
                    "n_x_padding": feature_to_add,
                    "features_per_group": self.x_preprocess.modules_dict['valid_feature_encoder'].num_features,
                    "num_used_features": self.x_preprocess.modules_dict['valid_feature_encoder'].valid_feature_num,
                    "mean_for_normalization": self.x_preprocess.modules_dict['normalization_encoder'].mean,
                    "std_for_normalization": self.x_preprocess.modules_dict['normalization_encoder'].std,
                }
                if x_categorical_mask is not None:
                    output_decoded["feature_process_config"]["x_categorical_mask"] = x_categorical_mask

            if enable_classification_y_pred_temp:
                output_decoded.update(cls_output)
            if enable_regression_y_pred_temp:
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
            preprocessed_x = self.mask_process_4_x(preprocessed_x)

        with nvtx.annotate('x-encoder'):
            preprocessed_x['x_categorical_mask'] = None
            preprocessed_x['feature_padding_mask'] = feature_padding_mask
            x_encoder_result = self.encoder_x(preprocessed_x)
            x_emb_result = x_encoder_result['data']

        return x_emb_result, real_x

    def y_encode(
        self,
        y,
        seq_len,
        batch_size,
        eval_pos,
        y_type:torch.Tensor,
        enable_classification_y_pred_temp: bool | None = None,
        enable_regression_y_pred_temp: bool | None = None,
    ):
        if enable_classification_y_pred_temp is None:
            enable_classification_y_pred_temp = self.enable_classification_y_pred
        if enable_regression_y_pred_temp is None:
            enable_regression_y_pred_temp = self.enable_regression_y_pred
        if enable_classification_y_pred_temp:
            with nvtx.annotate('cls_y-encoder'):
                cls_y_mask = (y_type == 0)
                cls_y_mask = cls_y_mask.unsqueeze(2)
                cls_y = {k: v.clone().detach() for k, v in y.items()}
                cls_y['eval_pos'] = eval_pos
                cls_y['data'] = torch.where(cls_y_mask, y['data'], 0)
                cls_embedded_y = self.cls_y_encoder(cls_y)
                cls_embedded_y = cls_embedded_y['data']

        if enable_regression_y_pred_temp:
            with nvtx.annotate('reg_y-encoder'):
                reg_y_mask = (y_type == 1)
                reg_y_mask = reg_y_mask.unsqueeze(2)
                reg_y = {k: v.clone().detach() for k, v in y.items()}
                reg_y['data'] = torch.where(reg_y_mask, y['data'], 0)
                reg_y['eval_pos'] = eval_pos
                reg_embedded_y = self.reg_y_encoder(reg_y)
                reg_embedded_y: torch.Tensor = reg_embedded_y['data']

        if enable_classification_y_pred_temp and enable_regression_y_pred_temp:
            flat_y_type = y_type.reshape(-1)
            flat_cls_embedded_y = cls_embedded_y.reshape(-1, self.y_encoder_embedding_size)
            flat_reg_embedded_y = reg_embedded_y.reshape(-1, self.y_encoder_embedding_size)
            flat_embedded_y = torch.empty(
                seq_len * batch_size,
                self.y_encoder_embedding_size,
                device=flat_reg_embedded_y.device,
                dtype=flat_reg_embedded_y.dtype,
            )
            flat_y_type = flat_y_type.to(torch.bool)
            flat_y_type = flat_y_type.unsqueeze(dim=-1)
            flat_embedded_y = torch.where(flat_y_type, flat_reg_embedded_y, flat_cls_embedded_y)
            embedded_y = flat_embedded_y.reshape(batch_size, seq_len, self.y_token_k, self.embed_dim)
        elif enable_classification_y_pred_temp:
            embedded_y = cls_embedded_y.reshape(batch_size, seq_len, self.y_token_k, self.embed_dim)
        elif enable_regression_y_pred_temp:
            embedded_y = reg_embedded_y.reshape(batch_size, seq_len, self.y_token_k, self.embed_dim)

        if torch.isnan(embedded_y).any():
            raise ValueError(
                f"{torch.isnan(embedded_y).any()=}, make sure to add nan handlers"
                " to the ys that are not fully provided (test set missing)",
            )

        return embedded_y


    def mask_process_4_x(self, data:dict, x_categorical_mask=None):
        x_input = data['data']
        mask = data['mask']

        x_feature_mean = torch.nanmean(x_input, dim=(1), keepdim=True)
        x_feature_mean = torch.where(torch.isnan(x_feature_mean), 0, x_feature_mean)
        if x_categorical_mask is not None:
            ori_x_input = x_input.clone().detach()

        x_input = torch.where(mask==1, float('nan'), x_input)
        x_input = torch.where(mask==2, x_feature_mean, x_input)
        x_input = torch.where(mask==3, x_input, x_input)
        # One randn over the whole sequence consumes a global-RNG offset proportional
        # to train+test rows, so later global draws change when only the test set
        # changes. Draw two fixed seeds, then sample each side from its own generator.
        # Skip the draw entirely when mask 4 is absent; the seed count must not
        # depend on how many cells are masked.
        if (mask == 4).any():
            eval_pos = int(data['eval_pos'])
            seeds = torch.randint(
                0,
                2**31 - 1,
                (2,),
                dtype=torch.int64,
                device=x_input.device,
            )
            train_seed, test_seed = (int(seed) for seed in seeds.tolist())
            generator = torch.Generator(device=x_input.device)
            generator.manual_seed(train_seed)
            train_noise = torch.randn(
                x_input[:, :eval_pos].shape,
                device=x_input.device,
                dtype=x_input.dtype,
                generator=generator,
            )
            generator.manual_seed(test_seed)
            test_noise = torch.randn(
                x_input[:, eval_pos:].shape,
                device=x_input.device,
                dtype=x_input.dtype,
                generator=generator,
            )
            noise = torch.cat((train_noise, test_noise), dim=1)
            x_input = torch.where(mask == 4, x_input + noise * 0.01, x_input)

        if x_categorical_mask is not None:
            x_input = torch.where(x_categorical_mask.unsqueeze(1) & (mask==2), float('nan'), x_input)
            randperm = torch.randperm(ori_x_input.shape[0])
            ori_x_input = ori_x_input[randperm]
            x_input = torch.where(x_categorical_mask.unsqueeze(1) & (mask==4), ori_x_input, x_input)

        data['data'] = x_input
        data['mask'] = mask.to(torch.bool)

        return data

    def add_embeddings(
        self,
        x: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ):
        if generator is not None and torch.device(generator.device) != x.device:
            raise ValueError(
                "feature positional embedding generator must use the input device: "
                f"generator={generator.device}, input={x.device}"
            )
        embs = torch.randn(
            (x.shape[2], x.shape[3] // 4),
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )
        embs = self.feature_positional_embedding(embs)
        x += embs[None, None]
        return x
    def flatten_y_tokens(self, y_tokens: torch.Tensor) -> torch.Tensor:
        return y_tokens.reshape(*y_tokens.shape[:2], self.y_decoder_input_dim)

    def decoder_for_clasification_task(self, test_encoder_out, eval_pos, y_type, y_true):
        # num_target_tokens = self.config_dict.get('num_target_tokens', 0)
        # test_encoder_out = encoder_out[:, eval_pos:, -num_target_tokens:]
        # if 'true_token' == self.config_dict.get('cls_target_token_type', 'true_token'):
        #     test_encoder_out = encoder_out[:, eval_pos:, -(num_target_tokens+1):, :]
        # elif 'concat' == self.config_dict.get('cls_target_token_type', 'true_token'):
        #     test_encoder_out = test_encoder_out.flatten(start_dim=2)
        # elif 'mean' == self.config_dict.get('cls_target_token_type', 'true_token'):
        #     test_encoder_out = test_encoder_out.mean(dim=2)
        # else:
        #     raise ValueError(f"Unknown cls_target_token_type={self.config_dict.get('cls_target_token_type', 'true_token')}")

        test_y_type = y_type[:, eval_pos:]

        if self.config_dict.get('use_cls_post_adapter', False):
            test_encoder_out = self.cls_post_adapter(test_encoder_out, task_type='cls')

        cls_mask = (test_y_type == 0)
        cls_mask = cls_mask.unsqueeze(2)
        zeros = torch.zeros_like(test_encoder_out)
        encoder_out_for_cls = torch.where(cls_mask, test_encoder_out, zeros)
        output = self.cls_y_decoder(encoder_out_for_cls)
        return {'cls_output': output}


    def decoder_for_regression_task(self, test_encoder_out, eval_pos, y_type, y_true):
        # num_target_tokens = self.config_dict.get('num_target_tokens', 0)
        # test_encoder_out = reg_y_emb[:, eval_pos:, -num_target_tokens:]
        # if 'true_token' == self.config_dict.get('reg_target_token_type', 'true_token'):
        #     test_encoder_out = reg_y_emb[:, eval_pos:, -(num_target_tokens+1):, :]
        # elif 'concat' == self.config_dict.get('reg_target_token_type', 'true_token'):
        #     test_encoder_out = test_encoder_out.flatten(start_dim=2)
        # elif 'mean' == self.config_dict.get('reg_target_token_type', 'true_token'):
        #     test_encoder_out = test_encoder_out.mean(dim=2)
        # else:
        #     raise ValueError(f"Unknown reg_target_token_type={self.config_dict.get('reg_target_token_type', 'true_token')}")

        test_y_type = y_type[:, eval_pos:]

        if self.config_dict.get('use_reg_post_adapter', False):
            test_encoder_out = self.reg_post_adapter(test_encoder_out, task_type='reg')

        reg_mask = (test_y_type == 1)
        reg_mask = reg_mask.unsqueeze(2)
        zeros = torch.zeros_like(test_encoder_out)
        encoder_out_for_reg = torch.where(reg_mask, test_encoder_out, zeros)
        return {'reg_output': [self.reg_y_decoder(encoder_out_for_reg)]}


    def decoder_for_feature_reconstruct(self, encoder_out:torch.Tensor, real_x: torch.Tensor, x_mask: torch.Tensor):
        """Reconstruct features from encoder outputs."""
        encoder_out = encoder_out[:, :, :-self.y_token_k, :]

        categorical_feature_pred = None
        numerical_feature_pred = None
        feature_pred = None

        if self.config_dict.get('use_feature_post_adapter', False):
            encoder_out = self.feature_post_adapter(encoder_out)

        group_mask = x_mask.any(dim=-1)
        feature_pred = real_x.clone()
        masked_pred = self.feature_decoder(encoder_out[group_mask]).to(real_x.dtype)
        filled = torch.where(x_mask[group_mask], masked_pred, real_x[group_mask])
        feature_pred[group_mask] = filled
        return feature_pred, None, None
