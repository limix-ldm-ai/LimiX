import gc
import torch
import torch.nn as nn
from typing import Any
import einops
from .utils import SetRandomSeed, create_mlp_layer
from .operators.triton_rmsnorm import build_norm
from .autobatch import (
    AutobatchConfig,
    autobatch,
    _cuda_free_bytes,
    _is_retryable_cuda_error,
    _try_empty_tensor,
)


def calc_mean(x:torch.Tensor, dim:int):
    num = torch.sum(~torch.isnan(x), dim=dim).clip(min=1.0)
    return torch.nansum(x, dim=dim) / num, num

def calc_std(x:torch.Tensor, dim:int, mean_v:torch.Tensor|None = None, value_num:torch.Tensor|None=None ):
    if mean_v is None or value_num is None:
        mean_v, value_num = calc_mean(x, dim)
    mean_broadcast = torch.repeat_interleave(mean_v.unsqueeze(dim), x.shape[dim], dim=dim,)
    return torch.sqrt(torch.nansum(torch.square(mean_broadcast - x), dim=dim) / (value_num - 1))

def normalize_mean0_std1(
                        x:torch.Tensor, 
                        eval_pos:int=-1,
                        clip:bool=True,
                        dim:int=1,
                        mean: torch.Tensor | None = None,
                        std: torch.Tensor | None = None
                        ):
    '''Normalize along dim=1 (sample axis). For x of shape (1, 1391, 2, 2), mean/std have shape (2, 2).'''
    if mean is None:
        mean, value_num = calc_mean(x[:,:eval_pos], dim=dim)
        std = calc_std(x[:,:eval_pos], dim=dim, mean_v=mean, value_num=value_num) + 1e-20
        
        if x.shape[1] == 1 or eval_pos == 1:
            std[:] = 1.0
    x = (x - mean.unsqueeze(1).expand_as(x)) / std.unsqueeze(1).expand_as(x)
    if clip:
        x = torch.clip(x, min=-100, max=100)
    return x, mean, std
    

class LinearEncoder(nn.Module):
    """linear input encoder"""
    def __init__(
                self,
                num_features: int,
                emsize: int,
                bias: bool = True,
                in_keys:list[str]=['data'],
                out_key:str='data',
                **kwargs
    ):
        """Initialize the LinearEncoder.

        Args:
            num_features: The number of input features.
            emsize: The embedding size, i.e. the number of output features.
            bias: Whether to use a bias term in the linear layer. Defaults to True.
        """
        super().__init__()

        self.layer = nn.Linear(num_features, emsize, bias=bias)
        self.in_keys = in_keys
        self.out_key = out_key
        
    def forward(self, input:dict[str, torch.Tensor|int])->dict[str, torch.Tensor]:
        assert 'data' in input and 'nan_encoding' in input
        x = [input[key] for key in self.in_keys] 
        x = torch.cat(x, dim=-1) # type: ignore
        input[self.out_key] = self.layer(x)
        return input


class YNoneEmbeddingEncoder(nn.Module):
    """Embed valid and invalid Y values separately. Invalid values are mainly None and +/- Inf."""
    def __init__(
                self,
                num_features: int,
                emsize: int,
                numeric_embed_type: str = 'linear',
                bias: bool = True,
                in_keys: list[str] = ['data'],
                out_key: str = 'data',
                **kwargs,
    ):
        """Initialize the YNoneEmbeddingDecoder.

        Args:
            num_features: The number of input features.
            emsize: The embedding size, i.e. the number of output features.
            nan_to_zero: Whether to replace NaN values in the input by zero. Defaults to False.
            bias: Whether to use a bias term in the linear layer. Defaults to True.
        """
        super().__init__()
        self.in_keys = in_keys
        self.out_key = out_key
        self.emb_mlp_hidden_dim_ratio = kwargs.get("emb_mlp_hidden_dim_ratio", 0.5)
        self.embedding_dim = emsize
        self.mask_token_emb_type = kwargs.get("mask_token_emb_type", "mask_emb")
        self.kwargs = kwargs

        if self.mask_token_emb_type in ("mask_emb", "uniform_mask_emb_type"):
            self.none_embedding = nn.Embedding(1, self.embedding_dim)
        else:
            raise ValueError(f"Unknown mask_token_emb_type: {self.mask_token_emb_type}")

        if 'linear' == numeric_embed_type:
            self.numeric_encoder = nn.Linear(1, emsize, bias=bias)
        elif 'MLP_' in numeric_embed_type:
            self.numeric_encoder = create_mlp_layer(
                mlp_type=numeric_embed_type,
                input_dim_size=1,
                hidden_size=int(emsize * self.emb_mlp_hidden_dim_ratio),
                output_dim_size=emsize,
                dropout=0.0,
                bias=bias,
                use_rmsnorm=kwargs.get('use_rmsnorm', False),
                layer_recompute=kwargs.get('layer_recompute', False),
            )
        else:
            raise ValueError(f"Invalid reg_y_numeric_embed_type: {numeric_embed_type}")


    def forward(self, input:dict[str, torch.Tensor|int])->dict[str, torch.Tensor]:
        assert 'data' in input

        none_mask = torch.isnan(input['data']) | torch.isinf(input['data'])
        no_none_data = torch.where(none_mask, torch.zeros_like(input['data']), input['data'])
        
        non_none_data_emb = self.numeric_encoder(no_none_data.unsqueeze(-1))
        none_emb = self.none_embedding(torch.zeros_like(none_mask, dtype=torch.int64))
        combined_emb = torch.where(none_mask.unsqueeze(-1), none_emb, non_none_data_emb)
        # combined_emb = self._add_mask_indicator(combined_emb, none_mask.unsqueeze(-1))

        input[self.out_key] = combined_emb
        
        return input


class RBFembedding(nn.Module):
    """
        RBF embedding layer use rbf to rescale x and use kernels to represent x
        input = mantissa * (log_base ** exp_i) 
        x = 312 = 3.12 * (10**2) = 1.2187 * (2**8)
        we then use mantissa and exp_i to represent x.

    Args:
        nn (_type_): _description_
    """
    def __init__(
        self, 
        embedding_size: int = 96,
        exponent_digits: int = 1,
        log_base: int = 10,
        token_embed_dim: int = 32,
        n_kernels: int = 32,
        sigma: float = 1.05,
        center_range: tuple = (0.0, 10.0),
        use_learn_embeddings: bool = False,
        as_tokenizer: bool = False,
        dtype: torch.dtype = torch.float32,
        use_rmsnorm: bool = False,
        layer_recompute: bool = False,
    ):
        super().__init__()
        self.dtype = dtype
        self.log_base = log_base
        self.n_kernels = n_kernels
        self.exponent_digits = exponent_digits
        self.as_tokenizer = as_tokenizer

        min_val, max_val = center_range
        centers = torch.linspace(min_val, max_val, steps=n_kernels, dtype=torch.float64)
        self.register_buffer("centers", centers, persistent=False)
        self.register_buffer("sigma", torch.tensor(sigma, dtype=dtype))

        self.sign_embedding = nn.Embedding(2, token_embed_dim, dtype=dtype)       # 0:+, 1:-
        self.exp_sign_embedding = nn.Embedding(2, token_embed_dim, dtype=dtype)   # 0:exp+, 1:exp-
        self.exp_digit_embedding = nn.Embedding(10, token_embed_dim, dtype=dtype)
        if not use_learn_embeddings:
            self.sign_embedding.weight.requires_grad = False
            self.exp_sign_embedding.weight.requires_grad = False
            self.exp_digit_embedding.weight.requires_grad = False

        ctrl_in_dim = (exponent_digits + 2) * token_embed_dim
        self.gate_mlp = nn.Sequential(
            nn.Linear(ctrl_in_dim, 4 * token_embed_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(4 * token_embed_dim, 2 * n_kernels, dtype=dtype)
        )

        self.norm = build_norm(n_kernels, use_rmsnorm=use_rmsnorm, recompute=layer_recompute, dtype=None if use_rmsnorm else dtype)
        self.out_layer = nn.Linear(n_kernels, embedding_size, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: shape (batch_size, n_features)
        x = x.squeeze(-1)
        S, F = x.shape[0], (x.shape[1] if x.ndim > 1 else 1)
        x64 = x.to(torch.float64)
        abs_x = torch.abs(x64)
        is_zero = (abs_x == 0)
        safe = torch.where(is_zero, torch.ones_like(abs_x), abs_x)
        exp_f = torch.floor(torch.log(safe)/torch.log(torch.tensor(self.log_base, dtype=safe.dtype)))
        max_exp = 10**self.exponent_digits - 1
        exp_i = torch.clamp(exp_f, -max_exp, max_exp).to(torch.int64)
        x_scaled = abs_x / (self.log_base ** exp_i.double())
        x_scaled = torch.where(is_zero, torch.zeros_like(x_scaled), x_scaled)

        diff = x_scaled.unsqueeze(-1) - self.centers.view((1,)*x_scaled.dim() + (self.n_kernels,))
        rbf = torch.exp(-(diff ** 2) / (2 * (self.sigma ** 2))).to(self.dtype)

        sign_idx = (x64 < 0).to(torch.long)
        exp_sign_idx = (exp_i < 0).to(torch.long)
        abs_exp = exp_i.abs()
        sign_emb = self.sign_embedding(sign_idx)
        exp_sign_emb = self.exp_sign_embedding(exp_sign_idx)
        exp_digit_emb_list = []
        for power in range(self.exponent_digits):
            digit = (abs_exp // (10 ** power)) % 10
            exp_digit_emb_list.append(self.exp_digit_embedding(digit))
        exp_digits_emb = torch.stack(exp_digit_emb_list[::-1], dim=-2)
        exp_digits_emb_flat = einops.rearrange(exp_digits_emb, "... e D -> ... (e D)")
        ctrl = torch.cat([sign_emb, exp_sign_emb, exp_digits_emb_flat], dim=-1).to(self.dtype)
        gamma_beta = self.gate_mlp(ctrl)
        gamma, beta = torch.split(gamma_beta, self.n_kernels, dim=-1)
        rbf = rbf * torch.sigmoid(gamma) + torch.tanh(beta)
        rbf = self.norm(rbf)
        out = self.out_layer(rbf.to(self.dtype))
        return out.reshape(S, -1) if not self.as_tokenizer else out



class MaskEmbEncoder(nn.Module):
    """
    For masked features, use the mask vector to obtain their representations; 
    for numerical features, use a nonlinear network to obtain their representations
    """
    def __init__(
                self,
                num_features: int,
                emsize: int,
                mask_embedding_size: int,
                numeric_embed_type: str = "linear",
                RBF_config: dict|None = None,
                bias: bool = True,
                in_keys: list[str] = ['data'],
                out_key: str = 'data',
                feature_fusion_network_type: str = "original",
                **kwargs
    ):
        """Initialize the MaskEmbEncoder.

        Args:
            num_features: The number of input features.
            emsize: The embedding size, i.e. the number of output features.
            nan_to_zero: Whether to replace NaN values in the input by zero. Defaults to False.
            bias: Whether to use a bias term in the linear layer. Defaults to True.
        """
        super().__init__()
        self.embedding_dim = emsize
        self.mask_embedding_size = mask_embedding_size
        self.in_keys = in_keys
        self.out_key = out_key
        self.use_feature_group_encoder = kwargs.get("use_feature_group_encoder", True)
        self.mask_token_emb_type = kwargs.get("mask_token_emb_type", "mask_emb")
        self.kwargs = kwargs
        use_rmsnorm = kwargs.get("use_rmsnorm", False)
        layer_recompute = kwargs.get("layer_recompute", False)

        if "mask_emb" == self.mask_token_emb_type:
            self.mask_embedding = nn.Parameter(torch.randn(self.mask_embedding_size))
        elif "uniform_mask_emb_type" == self.mask_token_emb_type:
            self.mask_embedding = nn.Embedding(1, self.embedding_dim)
        else:
            raise ValueError(f"Unknown mask_token_emb_type: {self.mask_token_emb_type}")

        self.numeric_embed_type = numeric_embed_type
        if numeric_embed_type == "linear":
            self.numeric_mlp = nn.Sequential(
                nn.Linear(1, self.embedding_dim // 2),
                build_norm(self.embedding_dim // 2, use_rmsnorm=use_rmsnorm, recompute=layer_recompute),
                nn.ReLU(),
                nn.Linear(self.embedding_dim // 2, self.embedding_dim),
                build_norm(self.embedding_dim, use_rmsnorm=use_rmsnorm, recompute=layer_recompute),
                nn.ReLU()
            )
        elif numeric_embed_type == "RBF":
            self.numeric_mlp = RBFembedding(
                embedding_size=self.embedding_dim,
                exponent_digits=RBF_config['RBF_exponent_digits'],
                token_embed_dim=RBF_config['RBF_token_embed_dim'],
                n_kernels=RBF_config['RBF_n_kernels'],
                sigma=RBF_config['RBF_sigma'],
                use_learn_embeddings=RBF_config['RBF_use_learn_embeddings'],
                log_base=RBF_config['RBF_log_base'],
                center_range=(0.0, 10.0),
                as_tokenizer=True,
                use_rmsnorm=use_rmsnorm,
                layer_recompute=layer_recompute,
            )
        else:
            raise ValueError(f"Invalid numeric_embed_type: {numeric_embed_type}")

        if self.use_feature_group_encoder:
            if 'original' != feature_fusion_network_type:
                raise ValueError(f"Invalid feature_fusion_network_type: {feature_fusion_network_type}")
            self.fusion_network = nn.Sequential(
                nn.Linear(num_features * self.embedding_dim, self.embedding_dim, bias=bias),
                build_norm(self.embedding_dim, use_rmsnorm=use_rmsnorm, recompute=layer_recompute),
                nn.ReLU(),
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=bias),
                build_norm(self.embedding_dim, use_rmsnorm=use_rmsnorm, recompute=layer_recompute),
            )

    @autobatch(batch_dim=1)
    def _cal_numeric_mlp(self, x: torch.Tensor) -> torch.Tensor:
        return self.numeric_mlp(x)
    
    @autobatch(batch_dim=1)
    def _cal_fusion_network(self, x: torch.Tensor) -> torch.Tensor:
        return self.fusion_network(x)

    def _combine_mask_emb(self, x_emb: torch.Tensor, is_mask: torch.Tensor) -> torch.Tensor:
        """Merge mask embeddings; seq-sliced calls are numerically equivalent to full-tensor compute (LN is on the last dim)."""
        if 'mask_emb' == self.mask_token_emb_type:
            mask_emb = self.mask_embedding.expand_as(x_emb)
        else:
            mask_vec = self.mask_embedding.weight[0]
            mask_emb = mask_vec.view(*([1] * (x_emb.dim() - 1)), -1).expand_as(x_emb)
        return torch.where(is_mask, mask_emb, x_emb)

    @autobatch(batch_dim=1)
    def _apply_mask_emb(self, x_emb: torch.Tensor, is_mask: torch.Tensor) -> torch.Tensor:
        """Apply mask merge in seq_len batches, matching _cal_numeric_mlp."""
        return self._combine_mask_emb(x_emb, is_mask)

    def _make_feature_group_emb(self, x_emb: torch.Tensor) -> torch.Tensor:
        """Get the embedding of a feature group."""
        try:
            feature_group = self._cal_fusion_network(x_emb)
        except torch.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            feature_group = self._cal_fusion_network(x_emb)
        return feature_group

    def _encode_seq_slice(self, x_bgf: torch.Tensor) -> torch.Tensor:
        """x: [B, s, G, F] -> grouped [B, s, G, E] (or keep the feature dim when fusion is off)."""
        x = x_bgf.unsqueeze(-1)
        is_mask = torch.isnan(x)
        x = x.masked_fill(is_mask, 0.0)
        x_emb = self.numeric_mlp(x)
        combined_emb = self._combine_mask_emb(x_emb, is_mask)
        del x, is_mask, x_emb
        if self.use_feature_group_encoder:
            combined_emb = combined_emb.flatten(3)
            return self.fusion_network(combined_emb)
        return combined_emb

    def _seq_chunk_size(self, seq_len: int, group: int, feature_num: int, device, dtype) -> int:
        elem = torch.tensor([], dtype=dtype).element_size()
        per_seq = group * feature_num * self.embedding_dim * elem * 12
        free = _cuda_free_bytes(device)
        if free is None:
            return seq_len
        n = int(0.2 * free / max(per_seq, 1))
        return max(1, min(seq_len, n))

    def _forward_chunked(self, x: torch.Tensor, chunk_size: int | None = None, reserve_y_token: bool = False) -> torch.Tensor:
        """Run mlp+mask+fusion along seq, materializing only [B, S, G, E] (or the unfused equivalent).

        reserve_y_token: allocate one extra group slot for the later y token so the transformer does not cat a full x embedding.
        """
        batch_size, seq_len, group, feature_num = x.shape
        if chunk_size is None:
            chunk_size = self._seq_chunk_size(seq_len, group, feature_num, x.device, x.dtype)
        out_groups = group + 1 if (reserve_y_token and self.use_feature_group_encoder) else group

        output = None
        while chunk_size >= 1:
            try:
                probe = self._encode_seq_slice(x[:, :1])
                out_tail = probe.shape[2:]
                alloc_tail = (out_groups,) + tuple(out_tail[1:]) if self.use_feature_group_encoder else tuple(out_tail)
                output = _try_empty_tensor(
                    (batch_size, seq_len) + alloc_tail,
                    probe.dtype,
                    x.device,
                )
                if output is None and probe.dtype != torch.float16 and x.device.type == 'cuda':
                    print("MaskEmbEncoder grouped fp32 output too large, store fp16")
                    output = _try_empty_tensor(
                        (batch_size, seq_len) + alloc_tail,
                        torch.float16,
                        x.device,
                    )
                if output is None:
                    raise torch.cuda.OutOfMemoryError(
                        "MaskEmbEncoder grouped output does not fit in memory"
                    )
                output[:, :1, :group] = probe.to(dtype=output.dtype)
                del probe
                for start in range(1, seq_len, chunk_size):
                    end = min(start + chunk_size, seq_len)
                    chunk_out = self._encode_seq_slice(x[:, start:end])
                    output[:, start:end, :group] = chunk_out.to(dtype=output.dtype)
                    del chunk_out
                if out_groups == group + 1:
                    output[:, :, group:].zero_()
                return output.view(batch_size, seq_len, out_groups, -1)
            except Exception as e:
                if output is not None:
                    del output
                    output = None
                if not _is_retryable_cuda_error(e) or chunk_size == 1:
                    raise
                print(f"MaskEmbEncoder chunk_size={chunk_size} OOM, retry with half")
                chunk_size = max(1, chunk_size // 2)
                gc.collect()
                if x.device.type == 'cuda':
                    torch.cuda.empty_cache()
        raise RuntimeError("MaskEmbEncoder chunked encode failed")

    def _forward_full(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, group, feature_num = x.shape
        x = x.unsqueeze(-1)
        is_mask = torch.isnan(x)
        x = x.masked_fill(is_mask, 0.0)

        x_emb = self._cal_numeric_mlp(x)

        try:
            combined_emb = self._combine_mask_emb(x_emb, is_mask)
            del x, is_mask, x_emb
        except torch.OutOfMemoryError:
            print("_apply_mask_emb OOM")
            gc.collect()
            torch.cuda.empty_cache()
            if is_mask.any():
                x_emb[is_mask.squeeze(-1).nonzero(as_tuple=True)] = self.mask_embedding.to(x_emb.dtype)
            combined_emb = x_emb
            del x, is_mask, x_emb

        if self.use_feature_group_encoder:
            combined_emb = combined_emb.flatten(3)
            sample_representation = self._make_feature_group_emb(combined_emb)
        else:
            sample_representation = combined_emb

        return sample_representation.view(batch_size, seq_len, group, -1)

    def forward(self, input:dict[str, torch.Tensor|int])->dict[str, torch.Tensor]:
        assert 'data' in input and 'nan_encoding' in input
        x = [input[key] for key in self.in_keys]
        x:torch.Tensor = torch.cat(x, dim=-1) # type: ignore

        if AutobatchConfig.ENABLE_AUTOBATCH:
            _batch, seq_len, group, feature_num = x.shape
            chunk_size = self._seq_chunk_size(seq_len, group, feature_num, x.device, x.dtype)
            if chunk_size < seq_len:
                output = self._forward_chunked(x, chunk_size=chunk_size, reserve_y_token=True)
            else:
                output = self._forward_full(x)
        else:
            output = self._forward_full(x)

        input[self.out_key] = output
        return input

class NanEncoder(nn.Module):
    """Encoder stage that deals with NaN and infinite values in the input"""
    def __init__(
        self,
        nan_value: float = -2.0,
        inf_value: float = 2.0,
        neg_info_value: float = 4.0,
        in_keys:list[str]=['data'],
        out_key:str='nan_encoding'
    ):
        """Initialize the NanEncoder.

        Args:
            keep_nans: Flag to maintain NaN values as individual indicators. 
        """
        super().__init__()
        self.nan_value = nan_value
        self.inf_value = inf_value
        self.neg_info_value = neg_info_value
        self.in_keys = in_keys
        self.out_key = out_key

    def forward(self, input: dict[str, torch.Tensor | int]) -> dict[str, torch.Tensor]:
        x = input[self.in_keys[0]] # type: ignore
        eval_pos = input['eval_pos']
        # x = x.contiguous()

        mean_value, _ = calc_mean(x[:, :eval_pos], dim=1)
        mean_value = mean_value.unsqueeze(1)

        x_nan, x_inf = x.isnan(), x.isinf()
        nans_indicator = (
            x_nan * self.nan_value +
            torch.logical_and(x_inf, x > 0) * self.inf_value +
            torch.logical_and(x_inf, x < 0) * self.neg_info_value
        ).to(x.dtype)
        nan_mask = torch.logical_or(x_nan, x_inf)
        x_new = torch.where(nan_mask, mean_value, x)

        input[self.in_keys[0]] = x_new
        input[self.out_key] = nans_indicator
        return input

class ValidFeatureEncoder(nn.Module):
    """Valid feature encoder"""
    def __init__(
        self,
        num_features: int,
        nan_normalize: bool=True,
        sqrt_normalize: bool=True,
        in_keys:list[str]=['data'],
        out_key:str='data'
    ):
        """Initialize the ValidFeatureEncoder.

        Args:
            num_features: The target number of features to transform the input into.
            nan_normalize: Indicates whether to normalize based on the number of features actually used.
            sqrt_normalize: Legacy option to normalize using the square root rather than the count of used features.
        """
        super().__init__()
        self.num_features = num_features
        self.nan_normalize = nan_normalize
        self.sqrt_normalize = sqrt_normalize
        self.in_keys = in_keys
        self.out_key = out_key
        self.valid_feature_num = None
    
    def forward(self, input:dict[str, torch.Tensor|int])->dict[str, torch.Tensor]:
        x:torch.Tensor = input[self.in_keys[0]]  # type: ignore
        valid_feature = ~torch.all(x == x[:, 0:1, :], dim=1)
        self.valid_feature_num = torch.clip(valid_feature.sum(-1).unsqueeze(-1), min=1)
        
        # x.shape:     torch.Size([1, 1391, 2, 2])
        # valid_feature.shape:     torch.Size([1, 2, 2])
        # self.valid_feature_num.shape:     torch.Size([1, 2, 1])
        
        if self.nan_normalize:
            x = x * torch.sqrt(self.num_features / self.valid_feature_num).unsqueeze(1).expand_as(x)
        
        zeros = torch.zeros(
            *x.shape[:-1],
            self.num_features - x.shape[-1],
            device=x.device,
            dtype=x.dtype,
        )
        x = torch.cat([x, zeros], -1)
        
        input[self.out_key] = x
        return input

class EmbYEncoderStep(nn.Module):
    """
    Use for classification task
    A simple linear input encoder step.
    """

    def __init__(
        self,
        *,
        emsize: int,
        n_classes: int = 10,
        in_keys: list[str] = ['data'],
        out_key: str = 'data',
    ):
        """Initialize the EmbYEncoderStep.

        Args:
            emsize: The embedding size, i.e. the number of output features.
            n_classes: Number of classes
        """
        super().__init__()
        self.y_embedding = nn.Embedding(n_classes, emsize)
        self.y_mask = nn.Embedding(1, emsize)
        self.in_keys = in_keys
        self.out_key = out_key
        
    def forward(self, input:dict[str, torch.Tensor|int])->dict[str, torch.Tensor]:
        y = input[self.in_keys[0]]
        eval_pos = input['eval_pos']
        y = y.int() # type: ignore
        y_train = y[:,:eval_pos]
        y_test = torch.zeros_like(y[:, eval_pos:], dtype=torch.int)
        y_train_emb = self.y_embedding(y_train)
        y_test_emb = self.y_mask(y_test)
        y_emb = torch.cat([y_train_emb, y_test_emb], dim=1)
        input[self.out_key] = y_emb
        return input


class MulticlassTargetEncoder(nn.Module):
    """Use the target's index as the class value, with each class corresponding to an index"""
    def __init__(
        self,
        in_keys:list[str]=['data'],
        out_key:str='data',
        mode:str=None 
    ):
        super().__init__()
        self.mode = mode
        self.in_keys = in_keys
        self.out_key = out_key

    def forward(self, input: dict[str, torch.Tensor | int]) -> dict[str, torch.Tensor]:
        x:torch.Tensor = input[self.in_keys[0]]  # type: ignore
        eval_pos = input['eval_pos']

        unique_xs: list[torch.Tensor] = [torch.unique(x[b, :eval_pos]) for b in range(x.shape[0])]
        for b in range(x.shape[0]):
            x[b, :eval_pos] = (x[b, :eval_pos].unsqueeze(-1) > unique_xs[b]).sum(dim=-1)

        input[self.out_key] = x
        if x.shape[0] == 1:
            input['cls_head_mapping'] = torch.arange(10).unsqueeze(dim=0).to(x.device)
        else:
            input['cls_head_mapping'] = torch.concat([torch.arange(10).unsqueeze(dim=0) for _ in range(x.shape[0])], dim=0).to(x.device)
        input['cls_head_mask'] = torch.ones([x.shape[0], 10]).to(x.device)
        return input


class NormalizationEncoder(nn.Module):
    """normalize encoder"""
    def __init__(
                self, 
                train_only:bool,
                normalize_x:bool,
                remove_outliers:bool,
                std_sigma:float=4.0,
                in_keys:list[str]=['data'],
                out_key:str='data'
                
    ):
        super().__init__()
        self.train_only = train_only
        self.normalize_x = normalize_x
        self.remove_outliers = remove_outliers
        self.std_sigma = std_sigma
        self.in_keys = in_keys
        self.out_key = out_key
        self.mean = None
        self.std = None

    def forward(self, input:dict[str, torch.Tensor|int])->dict[str, torch.Tensor]:
        x = input[self.in_keys[0]]
        eval_pos = input['eval_pos']
        pos = eval_pos if self.train_only else -1
        if self.normalize_x:
            x, self.mean, self.std = normalize_mean0_std1(x, eval_pos=pos)
        
        input[self.out_key] = x
        return input



def get_x_encoder(
    *,
    num_features: int,
    embedding_size: int,
    mask_embedding_size: int,
    categorical_features_class_num:int,
    encoder_use_bias: bool,
    feature_embedding_type: str = 'mask_embedding',
    numeric_embed_type: str = "scino",
    RBF_config: dict|None = None,
    init_seed_dict:dict = {},
    in_keys: list = ['data'],
    **kwargs
):
    assert isinstance(in_keys, list), "The type of in_keys must be a list!"
    inputs_to_merge = {}
    for in_key in in_keys:
        inputs_to_merge[in_key] = {'dim': num_features}

    encoder_steps = []
    with SetRandomSeed(init_seed_dict.get('encoder_x_seed', None)):
        if 'mask_embedding' != feature_embedding_type:
            raise ValueError(f'Unknown feature embedding type: {feature_embedding_type}')
        encoder_steps += [
            MaskEmbEncoder(
                num_features=sum([i["dim"] for i in inputs_to_merge.values()]),
                emsize=embedding_size,
                mask_embedding_size=mask_embedding_size,
                bias=encoder_use_bias,
                RBF_config=RBF_config,
                numeric_embed_type=numeric_embed_type,
                **kwargs
            ),
        ]

    return nn.Sequential(*encoder_steps,)

def get_cls_y_encoder(
    *,
    num_inputs: int,
    embedding_size: int,
    nan_handling_y_encoder: bool,
    max_num_classes: int,
    yemb_freeze_type: str|None = None,
    RBF_config:dict|None = None,
    init_seed_dict:dict = {},
    **kwargs
) -> nn.Module:
    steps = []
    inputs_to_merge = [{"name": "data", "dim": num_inputs}]
    if nan_handling_y_encoder:
        steps += [NanEncoder(in_keys=['data'], out_key='nan_encoding')]
        inputs_to_merge += [{"name": "nan_indicators", "dim": num_inputs}]

    if max_num_classes >= 2:
        steps += [MulticlassTargetEncoder()]

    y_encoder_type = kwargs.get('cls_y_encoder_type', 'emby_embedding')
    with SetRandomSeed(init_seed_dict.get('encoder_cls_y_seed', None)):
        if 'emby_embedding' != y_encoder_type:
            raise ValueError(f'Unknown cls y_encoder_type: {y_encoder_type}')
        steps += [
            EmbYEncoderStep(
                emsize=embedding_size,
                n_classes=max_num_classes,
            )
        ]
    return nn.Sequential(*steps)

def get_reg_y_encoder(
    *,
    num_inputs: int,
    num_features: int,
    embedding_size: int,
    nan_handling_y_encoder: bool,
    max_num_classes: int,
    yemb_freeze_type: str = 'yemb_open',
    RBF_config:dict|None = None,
    init_seed_dict:dict = {},
    **kwargs
) -> nn.Module:
    steps = []
    inputs_to_merge = [{"name": "data", "dim": num_inputs}]

    y_encoder_type = kwargs.get('y_encoder_type', 'linear')
    if 'none_embedding' != y_encoder_type:
        if nan_handling_y_encoder:
            steps += [NanEncoder(in_keys=['data'], out_key='nan_encoding')]
            inputs_to_merge += [{"name": "nan_indicators", "dim": num_inputs}]

    with SetRandomSeed(init_seed_dict.get('encoder_reg_y_seed', None)):
        if 'linear' == y_encoder_type:
            steps += [
                LinearEncoder(
                    num_features=sum([i["dim"] for i in inputs_to_merge]),  # type: ignore
                    emsize=embedding_size,
                    in_keys=['data', 'nan_encoding'],
                    out_key='data',
                    **kwargs
                ),
            ]
        elif 'none_embedding' == y_encoder_type:
            steps += [
                YNoneEmbeddingEncoder(
                    num_features=sum([i["dim"] for i in inputs_to_merge]),  # type: ignore
                    emsize=embedding_size,
                    numeric_embed_type=kwargs.get('reg_y_numeric_embed_type', 'linear'),
                    in_keys=['data', 'nan_encoding'],
                    out_key='data',
                    RBF_config=RBF_config,
                    **kwargs,
                ),
            ]
        else:
            raise ValueError(f'Unknown reg y_encoder_type: {y_encoder_type}')
    return nn.Sequential(*steps)


class PreprocessPipeline(nn.Module):
    def __init__(
        self,
        *,
        num_features: int,
        nan_handling_enabled: bool,
        normalize_on_train_only: bool,
        normalize_x: bool,
        remove_outliers: bool,
        normalize_by_used_features: bool,
    ):
        super().__init__()

        modules = nn.ModuleDict()
        step_order = []

        # Step 1: NanEncoder
        modules['nan_encoder'] = NanEncoder(in_keys=['data'], out_key='nan_encoding')
        step_order.append('nan_encoder')

        # Step 2: Optional ValidFeatureEncoder for NaN indicators
        if nan_handling_enabled:
            modules['nan_valid_encoder'] = ValidFeatureEncoder(
                num_features=num_features,
                nan_normalize=False,
                in_keys=["nan_encoding"],
                out_key="nan_encoding"
            )
            step_order.append('nan_valid_encoder')

        # Step 3: Normalization
        modules['normalization_encoder'] = NormalizationEncoder(
            train_only=normalize_on_train_only,
            normalize_x=normalize_x,
            remove_outliers=remove_outliers,
        )
        step_order.append('normalization_encoder')

        # Step 4: Final ValidFeatureEncoder
        modules['valid_feature_encoder'] = ValidFeatureEncoder(
            num_features=num_features,
            nan_normalize=normalize_by_used_features,
        )
        step_order.append('valid_feature_encoder')

        self.modules_dict = modules
        self.step_order = step_order  # preserve execution order

    def forward(self, x):
        # x is expected to be a dict-like input containing at least 'data'
        out = x
        for name in self.step_order:
            out = self.modules_dict[name](out)
        return out

def preprocesss_4_x(
    *,
    num_features: int,
    nan_handling_enabled: bool,
    normalize_on_train_only: bool,
    normalize_x: bool,
    remove_outliers: bool,
    normalize_by_used_features: bool,
):
    """Feature preprocessing pipeline implemented with nn.ModuleDict."""
    return PreprocessPipeline(
        num_features=num_features,
        nan_handling_enabled=nan_handling_enabled,
        normalize_on_train_only=normalize_on_train_only,
        normalize_x=normalize_x,
        remove_outliers=remove_outliers,
        normalize_by_used_features=normalize_by_used_features,
    )