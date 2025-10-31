import torch
import torch.nn as nn
from model.layer import EncoderBaseLayer, MLP, LayerStack
from typing import Any, Literal
from model.encoders import get_x_encoder, get_cls_y_encoder, get_reg_y_encoder, preprocesss_4_x




class FeaturesTransformer(nn.Module):
    def __init__(
                self,
                *,
                preprocess_config_x:dict[str, Any],
                encoder_config_x:dict[str, Any],
                encoder_config_y:dict[str, Any],
                decoder_config:dict[str, Any],
                nlayers:int,
                nhead: int, 
                embed_dim: int, 
                hid_dim:int,
                mask_prediction: bool = False,
                features_per_group:int = 2,
                dropout: float=0,
                activation: str='gelu',
                layer_norm_eps: float=1e-5,
                device: torch.device|None=None,
                dtype: torch.dtype|None=None,
                recompute_attn: bool=False,
                calculate_sample_attention: bool = False,
                calculate_feature_attention: bool = False
                ):
        super().__init__()
        
        self.preprocess_config_x = preprocess_config_x
        self.encoder_config_x = encoder_config_x
        self.encoder_config_y = encoder_config_y
        self.decoder_config = decoder_config
        self.nlayers = nlayers
        self.nhead = nhead
        self.embed_dim = embed_dim
        self.hid_dim = hid_dim
        self.mask_prediction = mask_prediction
        self.features_per_group = features_per_group
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.device = device
        self.dtype = dtype
        self.recompute_attn = recompute_attn

        layer_creator = lambda: EncoderBaseLayer(
            embed_dim=self.embed_dim,
            hid_dim=self.hid_dim,
            nhead=self.nhead,
            dropout=self.dropout,
            activation=self.activation, # type: ignore
            layer_norm_eps=self.layer_norm_eps,
            device=self.device,
            dtype=self.dtype,
            recompute_attn=self.recompute_attn,
            calculate_sample_attention=calculate_sample_attention,
            calculate_feature_attention=calculate_feature_attention
        )

        self.encoder_x = get_x_encoder( **encoder_config_x)
        self.cls_y_encoder = get_cls_y_encoder(**encoder_config_y)
        self.reg_y_encoder = get_reg_y_encoder(**encoder_config_y)

        self.transformer_encoder = LayerStack([layer_creator() for _ in range(self.nlayers)])
        self.encoder_out_norm = nn.LayerNorm(self.embed_dim, eps=1e-5, elementwise_affine=False)

        self.cls_y_decoder = nn.Sequential(
                                            nn.Linear(self.embed_dim, self.hid_dim),
                                            nn.GELU(),
                                            nn.Linear(self.hid_dim, decoder_config['num_classes']),
                                            )
        
        self.reg_y_decoder = nn.Sequential(
                                        nn.Linear(self.embed_dim, self.hid_dim),
                                        nn.LayerNorm(self.hid_dim),
                                        nn.GELU(),
                                        nn.Linear(self.hid_dim, 1),
                                        )
        self.feature_decoder = nn.Sequential(
                                        nn.Linear(self.embed_dim, self.hid_dim),
                                        nn.LayerNorm(self.hid_dim),
                                        nn.GELU(),
                                        nn.Linear(self.hid_dim, self.features_per_group),
                                        )
        
        self.feature_positional_embedding = nn.Linear(self.embed_dim // 4, self.embed_dim)
        
        self.x_preprocess = preprocesss_4_x(**preprocess_config_x)
        self.calculate_sample_attention = calculate_sample_attention
        self.calculate_feature_attention = calculate_feature_attention

    def forward(self, x: torch.Tensor, 
                y: torch.Tensor, 
                eval_pos: int, 
                task_type: Literal['reg', 'cls'] = 'cls') -> torch.Tensor | dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        '''
            x: The input x, which includes both train x and test x, Shape: [batch, sequence, feature]
            y: The input y, which includes both train y and test y, Shape: [batch, label]
            eval_pos: Train x and train y split point
            task_type: Type of task, options: cls(classification), reg(regression)
        '''
        assert x is not None and y is not None, "x and y must not be none"
        assert eval_pos > 0, "eval_pos must be a positive number"
        assert len(x.shape)==3, "x must be [Batch, seq, Feature]"
        assert len(y.shape)==2, "y must be [Batch, label]"
        assert eval_pos < x.shape[1] and eval_pos <= y.shape[1], "The split point between train x and test x must be less than the feature dimension of x, and less than or equal to the label dimension of y"
        
        batch_size, seq_len, num_feature = x.shape
        x = {'data':x, 'mask':torch.isnan(x).to(torch.int32).to(x.device)}
        y = {'data':y}
        
        feature_to_add = num_feature%self.features_per_group
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
        for k in x:
            x[k] = x[k].reshape(batch_size, seq_len, x[k].shape[2]//self.features_per_group, self.features_per_group)
        x['eval_pos'] = eval_pos
        preprocessed_x = self.x_preprocess(x)
        preprocessed_x = self.process_4_x(preprocessed_x)
        x_encoder_result = self.encoder_x(preprocessed_x)
        x_emb_result = x_encoder_result['data']
        
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
        # Mask the test y
        y["data"][eval_pos:] = torch.nan
        
        if task_type == 'cls':
            y_type =  torch.zeros_like(y['data'], device=y['data'].device)
        else:
            y_type =  torch.ones_like(y['data'], device=y['data'].device)
            
        embedded_y = self.mixed_y_embedding(y, y_type=y_type, eval_pos=eval_pos)

        if torch.isnan(embedded_y).any():
            raise ValueError("embedded_y contains NaN values; please add a NanEncoder in the encoder")
        
        embedded_x = self.add_embeddings(x_emb_result)
        embedded_all = torch.cat((embedded_x, embedded_y.unsqueeze(2)), dim=2)
        if torch.isnan(embedded_all).any():
            raise ValueError("embedded_all contains NaN values; please add a NanEncoder in the encoder")
        if self.calculate_sample_attention or self.calculate_feature_attention:
            return self.transformer_encoder(embedded_all, feature_atten_mask=None, eval_pos=eval_pos)
        else:
            pass
        encoder_out = self.transformer_encoder(embedded_all, feature_atten_mask=None, eval_pos=eval_pos)[0]
        encoder_out = self.encoder_out_norm(encoder_out)
        
        test_encoder_out = encoder_out[:, eval_pos:, -1]
        test_y_type = y_type[:,eval_pos:]
        encoder_out_4_feature = encoder_out[:, :, :-1, :]
        if self.mask_prediction:
            cls_output, reg_output = self.y_decoder(test_encoder_out, test_y_type)
            feature_pred = self.feature_decoder(encoder_out_4_feature)
            output_decoded = {
                "cls_output": cls_output,
                "reg_output": reg_output,
                "feature_pred": feature_pred,
                "process_config": {
                    "n_x_padding": feature_to_add,
                    "features_per_group": self.x_preprocess[3].num_features,
                    "num_used_features": self.x_preprocess[3].valid_feature_num,
                    "mean_for_normalization": self.x_preprocess[2].mean,
                    "std_for_normalization": self.x_preprocess[2].std
                }
            }
        else:
            cls_output, reg_output = self.y_decoder(test_encoder_out, test_y_type)
            if task_type=="cls":
                output_decoded = cls_output
            else:
                output_decoded = reg_output
            
        return output_decoded

    
    def mixed_y_embedding(self, y:dict, y_type:torch.Tensor, eval_pos:int):
        y = y['data']
        seq_len, batch_size, y_num = y.shape
        y_flat = y.reshape(-1)
        y_type_flat = y_type.reshape(-1)
        
        idx = torch.arange(len(y_flat), device=y.device)
        idx_cls = idx[y_type_flat == 0]
        idx_reg = idx[y_type_flat == 1]
        y_cls = y_flat[idx_cls]
        y_reg = y_flat[idx_reg]

        y_cls = y_cls.reshape(seq_len, -1, y_num)
        y_reg = y_reg.reshape(seq_len, -1, y_num)
        y_cls = {'data': y_cls, 'eval_pos':eval_pos}
        y_reg = {'data': y_reg, 'eval_pos':eval_pos}

        cls_y_emb = self.cls_y_encoder(y_cls) if len(idx_cls) > 0 else None
        reg_y_emb = self.reg_y_encoder(y_reg) if len(idx_reg) > 0 else None
        cls_y_emb = cls_y_emb['data'] if cls_y_emb is not None else None
        reg_y_emb = reg_y_emb['data'] if reg_y_emb is not None else None
        
        emb_size = self.embed_dim
        out = torch.empty(len(y_flat), emb_size, dtype=torch.float16, device=y_flat.device)
        if cls_y_emb is not None:            
            cls_y_emb_flat = cls_y_emb.reshape(-1, emb_size)
            out.index_put_((idx_cls,), cls_y_emb_flat)

        if reg_y_emb is not None:
            reg_y_emb_flat = reg_y_emb.reshape(-1, emb_size).to(torch.float16)
            out.index_put_((idx_reg,), reg_y_emb_flat)

        output = out.reshape(seq_len, batch_size, emb_size)
        return output
    
    def process_4_x(self, data:dict):
        x_input = data['data']
        mask = data['mask'].to(torch.bool)
        x_input = torch.where(mask, float('nan'), x_input)
        data['data'] = x_input
        return data
    
    def add_embeddings(self, x:torch.Tensor):
        with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu',
                                enabled=False):
            embs = torch.randn(
                (x.shape[2], x.shape[3] // 4),
                device=x.device,
                dtype=torch.float32,
            )
            torch.nn.init.orthogonal_(embs)
        embs =self.feature_positional_embedding(embs.to(x.dtype))
        x += embs[None, None]
        return x
    
    def y_decoder(self, test_encoder_out, test_y_type):
        seq_len, _, emb_size = test_encoder_out.shape
        flat_test_encoder_out = test_encoder_out.reshape(-1, emb_size)
        flat_test_y_type = test_y_type.reshape(-1)
        
        idx = torch.arange(len(flat_test_encoder_out), device=flat_test_encoder_out.device)
        idx_cls = idx[flat_test_y_type == 0]
        idx_reg = idx[flat_test_y_type == 1]

        cls_y_encoder_out = flat_test_encoder_out[idx_cls]
        reg_y_encoder_out = flat_test_encoder_out[idx_reg]
        cls_y_encoder_out = cls_y_encoder_out.reshape(seq_len, -1, emb_size)
        reg_y_encoder_out = reg_y_encoder_out.reshape(seq_len, -1, emb_size)

        cls_y = self.cls_y_decoder(cls_y_encoder_out)
        reg_y = self.reg_y_decoder(reg_y_encoder_out)

        return cls_y, reg_y
