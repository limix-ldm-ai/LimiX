import gc
from typing import Callable, Literal, Optional, Any
import functools

import nvtx
import torch
import torch.nn as nn
from torch.cuda import OutOfMemoryError
from torch.utils.checkpoint import checkpoint
from functools import partial
import einops
from .utils import SetRandomSeed, sdpa_context
from .utils import get_logger
import re
from .autobatch import autobatch
import torch.nn.functional as F
from .softmax_scaling_mlp import SoftmaxScalingMLP
from .attention_precision import (
    cast_flash_attention_inputs,
    normalize_flash_attention_precision,
    resolve_flash_attention_dtype,
)
from .decoupled_structural_task_attention import (
    DecoupledStructuralTaskAttention,
)
from .operators.rmsnorm import build_rmsnorm, RMSNormMixedPrecision, TritonQKNorm

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_kvpacked_func

    HAVE_FLASH_ATTN = True
except (ModuleNotFoundError, ImportError):
    HAVE_FLASH_ATTN = False

from typing_extensions import override

Activation = Literal['gelu', 'relu', 'silu']

ACTIVATION_FN: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    'gelu': nn.GELU(),
    'relu': nn.ReLU(),
    'silu': nn.SiLU(),
}


class MLP(torch.nn.Module):
    """Multi-Layer Perceptron"""

    def __init__(self,
                 in_features: int,
                 hidden_size: int,
                 out_features: int,
                 has_bias: bool,
                 device: torch.device | None,
                 dtype: torch.dtype | None,
                 activation: Activation = 'gelu',
                 depth: int = 2,
                 **kwargs,
                 ):
        super().__init__()

        self.depth = depth
        self.hidden_size = hidden_size
        self.activation = activation
        self.tf_mlp_layer_type = kwargs.get('tf_mlp_layer_type', 'gated')
        self.tf_mlp_activation_fuction = kwargs.get('tf_mlp_activation_fuction', 'gelu')
        self.use_gated_norm = kwargs.get('use_gated_norm', False)
        self.rmsnorm_impl = kwargs.get('rmsnorm_impl', 'triton')
        self.dropout_prob = kwargs.get('dropout', 0.0)
        self.dropout = nn.Dropout(self.dropout_prob) if self.dropout_prob > 0 else None
        self.kwargs = kwargs

        if self.tf_mlp_layer_type != 'gated':
            raise ValueError(f"Unknown tf_mlp_layer_type: {self.tf_mlp_layer_type}")

        hidden_size = int(in_features * self.kwargs.get('tf_mlp_hidden_size_ratio', 4.0))
        if self.kwargs.get('tf_mlp_hidden_size_2_even', False):
            hidden_size = hidden_size + 1 if hidden_size % 2 != 0 else hidden_size
        self.gated_norm = build_rmsnorm(
            hidden_size,
            eps=kwargs.get('layer_norm_eps', 1e-5),
            elementwise_affine=False,
            device=device,
            dtype=dtype,
            norm_impl=self.rmsnorm_impl,
            recompute=kwargs.get('layer_recompute', False)) if self.use_gated_norm else None
        self.gate_proj = nn.Linear(in_features, hidden_size, bias=has_bias, device=device, dtype=dtype)
        self.up_proj = nn.Linear(in_features, hidden_size, bias=has_bias, device=device, dtype=dtype)
        self.down_proj = nn.Linear(hidden_size, out_features, bias=has_bias, device=device, dtype=dtype)
        self.act_fn = ACTIVATION_FN[self.tf_mlp_activation_fuction]
        self.dropout_after_act = self.dropout

    @nvtx.annotate('MLP')
    @autobatch(batch_dim=1)
    def forward(self, x: torch.Tensor, y_type: bool = 0) -> torch.Tensor:
        with nvtx.annotate('gate-proj'):
            gate_out = self.gate_proj(x)
        with nvtx.annotate('up-proj'):
            up_out = self.up_proj(x)
        with nvtx.annotate('left-act-and-mul'):
            activated = self.act_fn(gate_out) * up_out
            if self.gated_norm is not None:
                activated = self.gated_norm(activated).to(activated.dtype)
            if self.dropout_after_act is not None:
                activated = self.dropout_after_act(activated)
        with nvtx.annotate('down-proj'):
            out = self.down_proj(activated)
        return out


class SeparateXYFFN(nn.Module):
    """Run independent FFNs over X tokens and the trailing y/CLS tokens."""

    def __init__(
            self,
            x_ffn: nn.Module | None,
            y_ffn: nn.Module,
            num_y_tokens: int = 1,
    ):
        super().__init__()
        if num_y_tokens < 1:
            raise ValueError("num_y_tokens must be greater than or equal to 1")
        self.x_ffn = x_ffn
        self.y_ffn = y_ffn
        self.num_y_tokens = num_y_tokens

    def forward(
            self,
            x: torch.Tensor,
            y_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del y_type
        if x.ndim < 2:
            raise ValueError(
                "SeparateXYFFN input must have token and embedding dimensions"
            )
        if x.shape[-2] < self.num_y_tokens:
            raise ValueError(
                "SeparateXYFFN input has fewer tokens than the trailing "
                f"{self.num_y_tokens} y/CLS token(s): {x.shape[-2]}"
            )

        x_tokens = x[..., :-self.num_y_tokens, :]
        if self.x_ffn is None:
            if x_tokens.shape[-2] != 0:
                raise ValueError(
                    "CLS-only FFN received non-CLS tokens; crop X before the "
                    "first CLS-only layer"
                )
            out_x = x_tokens
        else:
            with nvtx.annotate('x-ffn'):
                out_x = self.x_ffn(x_tokens)
        y_tokens = x[..., -self.num_y_tokens:, :].reshape(*x.shape[:2], -1)
        with nvtx.annotate('y-fnn'):
            out_y = self.y_ffn(y_tokens)
        return torch.cat(
            (out_x, out_y.reshape(*x.shape[:2], self.num_y_tokens, x.shape[-1])),
            dim=-2,
        )


class AttentionQKScalingMixin:
    def _init_qk_norm_and_scaling(
            self,
            kwargs: dict[str, Any],
            device: Optional[torch.device],
            dtype: Optional[torch.dtype],
    ) -> None:
        self.use_qk_norm = kwargs.get(
            "use_qk_norm",
            kwargs.get("induce_use_qk_norm", kwargs.get("induce_use_qknorm", False)),
        )
        self.use_softmax_scaling_mlp = kwargs.get(
            "use_softmax_scaling_mlp",
            kwargs.get("induce_use_softmax_scaling_mlp", False),
        )
        if not self.use_qk_norm:
            raise ValueError("use_qk_norm must be True")
        if not self.use_softmax_scaling_mlp:
            raise ValueError("use_softmax_scaling_mlp must be True")
        self.qk_norm_eps = kwargs.get(
            "qk_norm_eps",
            kwargs.get("induce_qk_norm_eps", kwargs.get("layer_norm_eps", 1e-5)),
        )
        self.qk_norm_elementwise_affine = kwargs.get(
            "qk_norm_elementwise_affine",
            kwargs.get("induce_qk_norm_elementwise_affine", True),
        )

        self.q_norm = build_rmsnorm(
            self.head_dim,
            eps=self.qk_norm_eps,
            elementwise_affine=self.qk_norm_elementwise_affine,
            device=device,
            dtype=dtype,
            norm_impl=self.rmsnorm_impl,
            norm_type='qknorm',
            recompute=self.recompute,
        )
        self.k_norm = build_rmsnorm(
            self.head_dim,
            eps=self.qk_norm_eps,
            elementwise_affine=self.qk_norm_elementwise_affine,
            device=device,
            dtype=dtype,
            norm_impl=self.rmsnorm_impl,
            norm_type='qknorm',
            recompute=self.recompute,
        )
        self.softmax_scaling_mlp = SoftmaxScalingMLP(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            temp_upper_bound=kwargs.get(
                "softmax_scaling_temp_upper_bound",
                kwargs.get("induce_softmax_scaling_temp_upper_bound", 0.5),
            ),
            temp_lower_bound=kwargs.get(
                "softmax_scaling_temp_lower_bound",
                kwargs.get("induce_softmax_scaling_temp_lower_bound", 0.0),
            ),
            scale_base_bound=kwargs.get(
                "scale_base_bound",
                kwargs.get("induce_softmax_scaling_base_bound", 5.0),
            ),
        )

        if device is not None or dtype is not None:
            self.q_norm.to(device=device, dtype=dtype)
            self.k_norm.to(device=device, dtype=dtype)
            self.softmax_scaling_mlp.to(device=device, dtype=dtype)

    @staticmethod
    def _run_qk_norm(norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if (isinstance(norm, RMSNormMixedPrecision) or isinstance(norm, TritonQKNorm)) and not x.is_cuda:
            weight = norm.weight
            if weight is not None:
                weight = weight.to(device=x.device, dtype=x.dtype)
            return F.rms_norm(x, (norm.hs,), weight, norm.eps)

        return norm(x)

    def _apply_qk_norm_and_scaling(
            self,
            q: torch.Tensor,
            kv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k, v = kv.unbind(dim=2)
        with nvtx.annotate('qknorm'):
            q = self._run_qk_norm(self.q_norm, q)
            k = self._run_qk_norm(self.k_norm, k)
        with nvtx.annotate('sm-scaling'):
            q = self.softmax_scaling_mlp(q, k.size(1))
        return q, torch.stack((k, v), dim=2)


class DifferentMHA(torch.nn.Module):
    def __init__(
            self,
            x_attention: nn.Module | None,
            y_attention: nn.Module,
            num_y_tokens: int = 1,
            use_x_attention: bool = True,
            y_token_merge_method: Literal['reshape', 'linear'] = 'reshape',
            embed_dim: int | None = None,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        del embed_dim, device, dtype
        if num_y_tokens < 1:
            raise ValueError("num_y_tokens must be greater than or equal to 1")
        if not use_x_attention:
            raise ValueError("use_x_attention must be True")
        if y_token_merge_method != 'reshape':
            raise ValueError(
                "y_token_merge_method must be 'reshape', "
                f"got {y_token_merge_method!r}"
            )
        if x_attention is None:
            raise ValueError("x_attention is required")
        self.x_attention = x_attention
        self.y_attention = y_attention
        self.num_y_tokens = num_y_tokens

    def _merge_y_tokens(self, y_tokens: torch.Tensor) -> torch.Tensor:
        # [B, Ty, S, E] -> [B, 1, S, Ty*E].  Move the sample axis before
        # flattening so values from different samples are never mixed.
        return y_tokens.permute(0, 2, 1, 3).flatten(
            start_dim=-2
        ).unsqueeze(1)

    def forward(
            self,
            x: torch.Tensor,
            x_kv: torch.Tensor,
            y_type: torch.Tensor | None = None,
            **kwargs
    ) -> torch.Tensor:
        del y_type, kwargs
        if x.ndim != 4 or x_kv.ndim != 4:
            raise ValueError(
                "DifferentMHA inputs must be [batch, token, sample, embedding]"
            )
        if x.shape[1] < self.num_y_tokens or x_kv.shape[1] < self.num_y_tokens:
            raise ValueError(
                "DifferentMHA input has fewer tokens than the trailing "
                f"{self.num_y_tokens} y/CLS token(s)"
            )

        x_tokens = x[:, :-self.num_y_tokens, :, :]
        xkv_tokens = x_kv[:, :-self.num_y_tokens, :, :]
        y_tokens = self._merge_y_tokens(x[:, -self.num_y_tokens:, :, :])
        ykv_tokens = self._merge_y_tokens(x_kv[:, -self.num_y_tokens:, :, :])

        with nvtx.annotate('x-attn'):
            out_x = self.x_attention(x_tokens, xkv_tokens)[0]
        with nvtx.annotate('y-attn'):
            out_y, _, sample_attention = self.y_attention(
                y_tokens,
                ykv_tokens,
            )
        out_y = out_y.squeeze(1).reshape(
            x.shape[0],
            x.shape[2],
            self.num_y_tokens,
            x.shape[3],
        ).permute(0, 2, 1, 3).contiguous()
        return torch.cat((out_x, out_y), dim=1), None, sample_attention


class MultiheadAttentionBertType(AttentionQKScalingMixin, torch.nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            qkv_combined: bool = False,
            dropout: float = 0,
            recompute: bool = False,
            deterministic: bool = False,
            has_bias: bool = False,
            **kwargs,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        if qkv_combined:
            raise ValueError("qkv_combined must be False")
        if kwargs.get('kv_combined', False):
            raise ValueError("kv_combined must be False")
        if recompute:
            raise ValueError("recompute must be False")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.recompute = recompute
        self.device = device
        self.dtype = dtype
        self.deterministic = deterministic
        self.has_bias = has_bias
        self.flash_attention_precision = normalize_flash_attention_precision(
            kwargs.get("flash_attention_precision", "inherit")
        )
        self.rmsnorm_impl = kwargs.get('rmsnorm_impl', 'triton')
        self._init_qk_norm_and_scaling(kwargs, device, dtype)

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=self.has_bias, **factory_kwargs)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=self.has_bias, **factory_kwargs)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=self.has_bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=self.has_bias, **factory_kwargs)

    def get_cu_seqlens(self, batch_size: int, seqlen: int, device: torch.device) -> torch.Tensor:
        return torch.arange(
            0,
            (batch_size + 1) * seqlen,
            step=seqlen,
            dtype=torch.int32,
            device=device,
        )

    @autobatch(num_batched_tensor=1)
    def compute_attention_by_torch(
            self,
            q: torch.Tensor,
            kv: torch.Tensor,
            attn_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """
        Since flash attention does not support attn_mask,
        use scaled_dot_product_attention to compute attention when attn_mask is not None
        """
        k, v = kv.unbind(dim=-3)
        with sdpa_context(self.deterministic):
            attention_outputs = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attn_mask=attn_mask,
                dropout_p=self.dropout,
            )
        attention_outputs = attention_outputs.transpose(1, 2)

        return attention_outputs

    def compute_attention_by_flashattn(
            self,
            q: torch.Tensor,
            kv: torch.Tensor,
    ) -> torch.Tensor:
        "Compute attention using flash attention"
        assert HAVE_FLASH_ATTN, \
            "Flash attention is not supported. Please install/reinstall flash attention."

        B, S = q.shape[:2]
        kv_shape = kv.shape
        atten_out = flash_attn_varlen_kvpacked_func(  # type: ignore
            q.reshape(B * S, self.num_heads, self.head_dim),
            kv.reshape(B * kv_shape[1], 2, self.num_heads, self.head_dim),
            self.get_cu_seqlens(B, S, q.device),
            self.get_cu_seqlens(B, kv_shape[1], kv.device),
            S,
            kv_shape[1],
            dropout_p=self.dropout,
            causal=False,
            return_attn_probs=False,
            deterministic=self.deterministic,
        )

        return atten_out.reshape(B, S, *atten_out.shape[1:])  # type: ignore

    def core_attention(
            self,
            q: torch.Tensor,
            kv: torch.Tensor,
            attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = q.device.type
        dtype = q.dtype
        flash_dtype = resolve_flash_attention_dtype(
            self.flash_attention_precision,
            dtype,
        )

        if (
                attn_mask is None
                and HAVE_FLASH_ATTN
                and device == 'cuda'
                and flash_dtype in {torch.float16, torch.bfloat16}
        ):
            flash_q, flash_kv = cast_flash_attention_inputs(
                self.flash_attention_precision,
                q,
                kv,
            )
            attn_out = self.compute_attention_by_flashattn(
                flash_q,
                flash_kv,
            ).to(dtype=dtype)
        else:
            attn_out = self.compute_attention_by_torch(q, kv, attn_mask)
        return attn_out

    @override
    @autobatch(batch_dim=1)
    def forward(
            self,
            x: torch.Tensor,
            x_kv: Optional[torch.Tensor] = None,
            attn_mask: torch.Tensor | None = None,
            y_type: torch.Tensor = None,
            **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        x: [batch_size, seq_len, feature, embed_dim]
        x_kv: [batch_size, seq_len_kv, feature, embed_dim]
        """
        del y_type, kwargs
        B, S, _, _ = x.shape
        assert x.shape[-1] == self.embed_dim
        if x_kv is None:
            raise ValueError("x_kv is required")

        x = x.reshape(-1, *x.shape[-2:])
        BS, F, E = x.shape

        with nvtx.annotate('linear-qkv'):
            q = self.q_proj(x).view(x.size(0), x.size(1), self.num_heads, self.head_dim)
            x_kv_flat = x_kv.reshape(-1, x_kv.shape[-2], E)
            k = self.k_proj(x_kv_flat).view(
                x_kv_flat.size(0), x_kv_flat.size(1), self.num_heads, self.head_dim
            )
            v = self.v_proj(x_kv_flat).view(
                x_kv_flat.size(0), x_kv_flat.size(1), self.num_heads, self.head_dim
            )
            kv = torch.stack([k, v], dim=2)

        q, kv = self._apply_qk_norm_and_scaling(q, kv)

        with nvtx.annotate('core-attn'):
            attn_out = self.core_attention(q, kv, attn_mask)
            attn_out = attn_out.reshape(BS, F, self.num_heads, self.head_dim)

        with nvtx.annotate('linear-o'):
            out = self.out_proj(attn_out.reshape(attn_out.size(0), attn_out.size(1), -1))

        return out.reshape(B, S, *out.shape[1:]), None, None


class EncoderBaseLayer(nn.Module):
    "Base encoder layer of the Transformer model"

    def __init__(self,
                 nhead: int,
                 embed_dim: int,
                 hid_dim: int,
                 layer_idx: int,
                 dropout: float = 0,
                 pre_norm: bool = False,
                 all_norm: bool = False,
                 activation: Literal['relu', 'gelu'] = 'gelu',
                 layer_norm_eps: float = 1e-5,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None,
                 recompute_attn: bool = False,
                 layer_arch: str = 'smf',
                 deterministic: bool = False,
                 seq_attn_isolated: bool = False,
                 seq_attn_serial: bool = False,
                 self_share_all_kv_heads: bool = False,
                 cross_share_all_kv_heads: bool = True,
                 init_seed_dict: dict = {},
                 mlp_seed_mode: str = 'split',
                 layer_recompute: bool = False,
                 **layer_kwargs: Any,
                 ):
        super().__init__()
        self.logger = get_logger('ldm', 'model')
        self.nhead = nhead
        self.embed_dim = embed_dim
        self.hid_dim = hid_dim
        self.dropout = dropout
        self.all_norm = all_norm
        # All-norm extends pre-norm with a second, independent normalization on
        # the residual branch output: x = x + out_norm(layer(in_norm(x))).
        self.pre_norm = pre_norm or all_norm
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps

        self.use_separate_attention = layer_kwargs.get("use_separate_attention", False)
        self.separate_attn_kv_combined = layer_kwargs.get("separate_attn_kv_combined", False)
        #        if self.use_separate_attention:
        #            print("use_separate_attention")
        self.device = device
        self.dtype = dtype
        self.layer_arch = layer_arch
        self.head_dim = self.embed_dim // self.nhead
        self.recompute_attn = recompute_attn
        self.layer_recompute = layer_recompute
        self.self_share_all_kv_heads = self_share_all_kv_heads
        self.cross_share_all_kv_heads = cross_share_all_kv_heads
        self.seq_attn_serial = seq_attn_serial
        self.seq_attn_isolated = seq_attn_isolated
        self.init_seed_dict = init_seed_dict
        self.mlp_seed_mode = mlp_seed_mode
        self.layer_idx = layer_idx
        self.mlp_idx_info = {}
        self.layer_kwargs = layer_kwargs
        self.cls_only = bool(layer_kwargs.get("cls_only", False))
        self.y_token_k = int(layer_kwargs.get('num_cls_tokens', 1))
        for head_count_name in (
                "cls_sample_attention_num_heads",
                "cls_only_sample_attention_num_heads",
        ):
            raw_head_count = layer_kwargs.get(head_count_name, 0)
            head_count = 0 if raw_head_count is None else int(raw_head_count)
            if head_count < 0:
                raise ValueError(
                    f"{head_count_name} must be 0 (automatic) or greater, "
                    f"got {head_count}"
                )
            setattr(self, head_count_name, head_count)
        self.separate_x_y_ffn = layer_kwargs.get("separate_x_y_ffn", False)
        self.sample_attention_cls_token_merge = layer_kwargs.get(
            "sample_attention_cls_token_merge",
            "reshape",
        )
        self.feature_attention_cls_token_merge = layer_kwargs.get(
            "feature_attention_cls_token_merge",
            "none",
        )
        self.feature_attention_pipeline = layer_kwargs.get(
            "feature_attention_pipeline",
            "legacy",
        )
        self.decoupled_attn_kv_combined= layer_kwargs.get(
            "decoupled_attn_kv_combined",
            False,
        )
        self.num_y_tokens = self.y_token_k
        if self.sample_attention_cls_token_merge not in ('reshape', 'linear'):
            raise ValueError(
                "sample_attention_cls_token_merge must be 'reshape' or 'linear', "
                f"got {self.sample_attention_cls_token_merge!r}"
            )
        if self.feature_attention_cls_token_merge != 'none':
            raise ValueError(
                "feature_attention_cls_token_merge must be 'none', "
                f"got {self.feature_attention_cls_token_merge!r}"
            )
        if self.feature_attention_pipeline != "decoupled":
            raise ValueError(
                "feature_attention_pipeline must be 'decoupled', "
                f"got {self.feature_attention_pipeline!r}"
            )
        self.feature_attentions = []
        self.sequence_attentions = []
        self.mlp = []
        self.deterministic = deterministic
        self.layer_arch_dict = self.parse_arch(layer_arch)
        if self.cls_only:
            # LayerStack removes X at the boundary.  Avoid constructing Feature
            # Attention modules (and unused parameters) in all later layers.
            self.layer_arch_dict['FA'] = []
            self.layer_arch_dict['arch'] = self.layer_arch_dict['arch'].replace('F', '')
        self.tf_mlp_layer_type = layer_kwargs.get('tf_mlp_layer_type', 'gated')
        self.tf_mlp_activation_fuction = layer_kwargs.get('tf_mlp_activation_fuction', 'gelu')
        self.tf_mlp_hidden_size_ratio = layer_kwargs.get('tf_mlp_hidden_size_ratio', 4.0)
        self.tf_mlp_hidden_size_2_even = layer_kwargs.get('tf_mlp_hidden_size_2_even', False)
        self.tf_mlp_y_hidden_size_ratio = layer_kwargs.get('tf_mlp_y_hidden_size_ratio', None)
        self.tf_mlp_use_bias = layer_kwargs.get('tf_mlp_use_bias', False)
        self.tf_attention_layer_type = layer_kwargs.get('tf_attention_layer_type', 'bert')
        self.tf_attention_use_bias = layer_kwargs.get('tf_attention_use_bias', False)
        self.tf_mlp_dropout = layer_kwargs.get('tf_mlp_dropout', 0.0)
        self.tf_norm_type = layer_kwargs.get('tf_norm_type', 'layer_norm')
        self.rmsnorm_impl = layer_kwargs.get('rmsnorm_impl', 'triton')
        self.tf_layer_norm_use_elementwise_affine = layer_kwargs.get('tf_layer_norm_use_elementwise_affine', False)
        self.induce_use_qk_norm = layer_kwargs.get(
            "induce_use_qk_norm",
            layer_kwargs.get("induce_use_qknorm", False),
        )
        self.induce_qk_norm_eps = layer_kwargs.get(
            "induce_qk_norm_eps",
            self.layer_norm_eps,
        )
        self.induce_qk_norm_elementwise_affine = layer_kwargs.get(
            "induce_qk_norm_elementwise_affine",
            True,
        )
        self.induce_use_softmax_scaling_mlp = layer_kwargs.get(
            "induce_use_softmax_scaling_mlp",
            False,
        )
        self.induce_softmax_scaling_temp_upper_bound = layer_kwargs.get(
            "induce_softmax_scaling_temp_upper_bound",
            0.5,
        )
        self.induce_softmax_scaling_temp_lower_bound = layer_kwargs.get(
            "induce_softmax_scaling_temp_lower_bound",
            0.0,
        )
        self.induce_softmax_scaling_base_bound = layer_kwargs.get(
            "induce_softmax_scaling_base_bound",
            5.0,
        )

        self.sa_temp_upper_bound = layer_kwargs.get(
            "sa_temp_upper_bound",
            0.5
        )
        self.sa_temp_lower_bound = layer_kwargs.get(
            "sa_temp_lower_bound",
            0
        )
        self.sa_scale_base_bound = layer_kwargs.get(
            "sa_scale_base_bound",
            5.0
        )
        self.feature_attn_num = len(self.layer_arch_dict['FA'])  # feature attention number
        self.mlp_num = len(self.layer_arch_dict['MLP'])  # sequence attention number
        self.seq_attn_num = len(self.layer_arch_dict['SA'])  # mlp number
        if self.seq_attn_isolated:
            self.seq_attn_num *= 2
        self.layer_compress = layer_kwargs.get('layer_compress', False)

        if self.seq_attn_num <= 0 or self.mlp_num <= 0:
            raise ValueError(
                "Every encoder layer must contain Sample Attention and MLP; "
                f"got layer_arch={layer_arch!r}"
            )
        if not self.cls_only and self.feature_attn_num <= 0:
            raise ValueError(
                "A full encoder layer must contain Feature Attention; "
                f"got layer_arch={layer_arch!r}"
            )
        self.logger.debug(f"layer arch: {self.layer_arch_dict}")
        # attention+MLP
        self.feature_attentions = nn.ModuleList([self.build_FA(i) for i in range(self.feature_attn_num)])
        self.sequence_attentions = nn.ModuleList([self.build_SA(i) for i in range(self.seq_attn_num)])

        self.mlp = nn.ModuleList(
            [self.build_MLP(i, mlp_info) for i, mlp_info in enumerate(self.layer_arch_dict['MLP'])])

        self.layer_steps = []
        F_idx = 0
        S_idx = 0
        M_idx = 0
        for arch in self.layer_arch_dict['arch']:
            if arch == 'F':
                self.layer_steps.append(partial(self.call_features_attention, index=F_idx))
                F_idx += 1
            elif arch == 'S':
                self.layer_steps.append(partial(self.call_sequence_attention, index=S_idx))
                S_idx += 1
            elif arch == 'M':
                self.layer_steps.append(self.mlp[M_idx])
                M_idx += 1
            else:
                raise ValueError(f"unsupport layer arch: {self.layer_arch_dict['arch']}")

        self.feature_attention_step_idx = self.layer_arch_dict['arch'].rfind('F')
        self.sample_attention_step_idx = self.layer_arch_dict['arch'].rfind('S')

        self.layer_norms = nn.ModuleList([self.build_LN(i) for i in range(len(self.layer_steps))])
        # Keep this list empty for legacy pre/post-norm models so their state
        # dicts remain unchanged.  All-norm deliberately does not share affine
        # parameters between the input and output normalizations.
        self.output_layer_norms = nn.ModuleList(
            [self.build_LN(i, seed_name='output_LN') for i in range(len(self.layer_steps))]
            if self.all_norm else []
        )

    def build_FA(self, idx: int):
        with SetRandomSeed(self.init_seed_dict.get(f"layer{self.layer_idx}_FA{idx}_seed", None)):
            if self.feature_attention_pipeline == "decoupled":
                relation_dim = int(self.layer_kwargs.get("dsti_relation_dim", 0)) or self.embed_dim
                value_dim = int(self.layer_kwargs.get("dsti_value_dim", 0)) or self.embed_dim
                num_heads = int(self.layer_kwargs.get("dsti_num_heads", 0)) or self.nhead
                dsti_dropout = float(self.layer_kwargs.get("dsti_dropout", -1.0))
                if dsti_dropout < 0:
                    dsti_dropout = self.dropout
                return DecoupledStructuralTaskAttention(
                    embed_dim=self.embed_dim,
                    num_heads=num_heads,
                    num_y_tokens=self.num_y_tokens,
                    relation_dim=relation_dim,
                    value_dim=value_dim,
                    feature_interaction_mode=self.layer_kwargs.get(
                        "dsti_feature_interaction_mode", "sample_split"
                    ),
                    y_token_aggregation=self.layer_kwargs.get(
                        "dsti_y_token_aggregation", "concat_linear"
                    ),
                    dropout=dsti_dropout,
                    bias=self.tf_attention_use_bias,
                    use_qk_norm=self.layer_kwargs.get("dsti_use_qk_norm", False),
                    rmsnorm_impl=self.layer_kwargs.get('rmsnorm_impl', 'triton'),
                    qk_norm_eps=self.layer_kwargs.get("dsti_qk_norm_eps", 1e-5),
                    qk_norm_elementwise_affine=self.layer_kwargs.get(
                        "dsti_qk_norm_elementwise_affine", True
                    ),
                    use_softmax_scaling_mlp=self.layer_kwargs.get(
                        "dsti_use_softmax_scaling_mlp", False
                    ),
                    softmax_scaling_temp_upper_bound=self.layer_kwargs.get(
                        "dsti_softmax_scaling_temp_upper_bound", 0.4
                    ),
                    softmax_scaling_temp_lower_bound=self.layer_kwargs.get(
                        "dsti_softmax_scaling_temp_lower_bound", 0.0
                    ),
                    softmax_scaling_base_bound=self.layer_kwargs.get(
                        "dsti_softmax_scaling_base_bound", 1.0
                    ),
                    deterministic=self.deterministic,
                    recompute=self.recompute_attn,
                    attention_backend=self.layer_kwargs.get(
                        "dsti_attention_backend", "auto"
                    ),
                    flash_attention_precision=self.layer_kwargs.get(
                        "flash_attention_precision", "inherit"
                    ),
                    device=self.device,
                    dtype=self.dtype,
                    kv_combined=self.decoupled_attn_kv_combined,
                )
            raise ValueError(
                "feature_attention_pipeline must be 'decoupled', "
                f"got {self.feature_attention_pipeline!r}"
            )

    def _resolve_cls_sample_attention_num_heads(
            self,
            *,
            attention_embed_dim: int,
            automatic_num_heads: int,
    ) -> int:
        """Resolve the CLS sample-attention head count for this layer stage."""
        parameter_name = (
            "cls_only_sample_attention_num_heads"
            if self.cls_only
            else "cls_sample_attention_num_heads"
        )
        configured_num_heads = getattr(self, parameter_name)
        num_heads = configured_num_heads or automatic_num_heads
        if attention_embed_dim % num_heads != 0:
            raise ValueError(
                f"{parameter_name}={num_heads} must divide the CLS sample-attention "
                f"embedding dimension {attention_embed_dim} in layer {self.layer_idx}"
            )
        return num_heads

    def build_SA(self, idx: int):
        with SetRandomSeed(self.init_seed_dict.get(f"layer{self.layer_idx}_SA{idx}_seed", None)):
            if self.tf_attention_layer_type != 'bert':
                raise ValueError(
                    f"unknown tf_attention_layer_type: {self.tf_attention_layer_type}"
                )
            attention_kwargs = dict(
                embed_dim=self.embed_dim,
                num_heads=self.nhead,
                device=self.device,
                dtype=self.dtype,
                qkv_combined=False,
                dropout=self.dropout,
                recompute=self.recompute_attn,
                deterministic=self.deterministic,
                has_bias=self.tf_attention_use_bias,
                layer_norm_eps=self.layer_norm_eps,
                rmsnorm_impl=self.rmsnorm_impl,
                layer_recompute=self.layer_recompute,
                induce_use_qk_norm=self.induce_use_qk_norm,
                induce_qk_norm_eps=self.induce_qk_norm_eps,
                induce_qk_norm_elementwise_affine=self.induce_qk_norm_elementwise_affine,
                induce_use_softmax_scaling_mlp=self.induce_use_softmax_scaling_mlp,
                induce_softmax_scaling_temp_upper_bound=self.sa_temp_upper_bound,
                induce_softmax_scaling_temp_lower_bound=self.sa_temp_lower_bound,
                induce_softmax_scaling_base_bound=self.sa_scale_base_bound,
                kv_combined=self.separate_attn_kv_combined,
                flash_attention_precision=self.layer_kwargs.get(
                    "flash_attention_precision", "inherit"
                ),
            )

            if self.cls_only:
                y_attention_kwargs = dict(attention_kwargs)
                if self.sample_attention_cls_token_merge == 'reshape':
                    y_attention_kwargs["embed_dim"] = self.embed_dim * self.y_token_k
                    automatic_num_heads = self.nhead
                    if self.layer_kwargs.get("scale_y_sample_attention_heads", True):
                        automatic_num_heads *= self.y_token_k
                else:
                    y_attention_kwargs["embed_dim"] = self.embed_dim
                    automatic_num_heads = self.nhead
                y_attention_kwargs["num_heads"] = (
                    self._resolve_cls_sample_attention_num_heads(
                        attention_embed_dim=y_attention_kwargs["embed_dim"],
                        automatic_num_heads=automatic_num_heads,
                    )
                )
                y_attention = MultiheadAttentionBertType(**y_attention_kwargs)
                return DifferentMHA(
                    x_attention=None,
                    y_attention=y_attention,
                    num_y_tokens=self.y_token_k,
                    use_x_attention=False,
                    y_token_merge_method=self.sample_attention_cls_token_merge,
                    embed_dim=self.embed_dim,
                    device=self.device,
                    dtype=self.dtype,
                )

            if self.cls_sample_attention_num_heads > 0 and not self.use_separate_attention:
                raise ValueError(
                    "cls_sample_attention_num_heads requires a distinct CLS "
                    "sample-attention branch; enable use_separate_attention"
                )

            if self.use_separate_attention:
                x_attention = MultiheadAttentionBertType(**attention_kwargs)
                y_attention_kwargs = dict(attention_kwargs)
                if self.sample_attention_cls_token_merge == 'reshape':
                    y_attention_kwargs["embed_dim"] = self.embed_dim * self.y_token_k
                    # Flattening K Y tokens widens the Y embedding by K.  Grow
                    # the head count by the same factor so every original
                    # token retains the base per-head width instead of being
                    # compressed into the original number of heads.
                    automatic_num_heads = self.nhead
                    if self.layer_kwargs.get(
                            "scale_y_sample_attention_heads", True
                    ):
                        automatic_num_heads *= self.y_token_k
                else:
                    y_attention_kwargs["embed_dim"] = self.embed_dim
                    automatic_num_heads = self.nhead
                y_attention_kwargs["num_heads"] = (
                    self._resolve_cls_sample_attention_num_heads(
                        attention_embed_dim=y_attention_kwargs["embed_dim"],
                        automatic_num_heads=automatic_num_heads,
                    )
                )
                y_attention = MultiheadAttentionBertType(**y_attention_kwargs)
                return DifferentMHA(
                    x_attention=x_attention,
                    y_attention=y_attention,
                    num_y_tokens=self.y_token_k,
                    y_token_merge_method=self.sample_attention_cls_token_merge,
                    embed_dim=self.embed_dim,
                    device=self.device,
                    dtype=self.dtype,
                )
            return MultiheadAttentionBertType(**attention_kwargs)

    def build_MLP(self, idx: int, mlp_info_dict):
        if self.mlp_seed_mode == 'split':
            m_key = f"M{mlp_info_dict['depth']}"
            if m_key in self.mlp_idx_info:
                self.mlp_idx_info[m_key] += 1
            else:
                self.mlp_idx_info[m_key] = 0
            idx = self.mlp_idx_info[m_key]
            seed = self.init_seed_dict.get(f"layer{self.layer_idx}_MLP{mlp_info_dict['depth']}_{idx}_seed", None)
        else:
            seed = self.init_seed_dict.get(f"layer{self.layer_idx}_MLP{mlp_info_dict['depth']}_{idx}_seed", None)

        ffn_kwargs = {
            'in_features': self.embed_dim,
            'hidden_size': self.hid_dim,
            'out_features': self.embed_dim,
            'has_bias': self.tf_mlp_use_bias,
            'device': self.device,
            'dtype': self.dtype,
            'activation': self.activation,
            'depth': mlp_info_dict['depth'],
            'tf_mlp_layer_type': self.tf_mlp_layer_type,
            'tf_mlp_activation_fuction': self.tf_mlp_activation_fuction,
            "tf_mlp_hidden_size_ratio": self.tf_mlp_hidden_size_ratio,
            "tf_mlp_hidden_size_2_even": self.tf_mlp_hidden_size_2_even,
            'dropout': self.tf_mlp_dropout,
            'use_gated_norm': self.layer_kwargs.get('use_gated_norm', False),
            'rmsnorm_impl': self.layer_kwargs.get('rmsnorm_impl', 'triton'),
            'layer_norm_eps': self.layer_norm_eps,
            'layer_recompute': self.layer_kwargs.get('layer_recompute', False),
        }

        def make_ffn(kwargs: dict) -> nn.Module:
            resolved_factory = MLP
            ffn = resolved_factory(**kwargs)
            if not isinstance(ffn, nn.Module):
                raise TypeError(
                    "FFN factory must return an instance of torch.nn.Module, "
                    f"got {type(ffn).__name__}"
                )
            return ffn

        with SetRandomSeed(seed):
            if self.separate_x_y_ffn:
                x_kwargs = {
                    'in_features': self.embed_dim,
                    'hidden_size': self.hid_dim,
                    'out_features': self.embed_dim,
                    'has_bias': self.tf_mlp_use_bias,
                    'device': self.device,
                    'dtype': self.dtype,
                    'activation': self.activation,
                    'depth': mlp_info_dict['depth'],
                    'tf_mlp_layer_type': self.tf_mlp_layer_type,
                    'tf_mlp_activation_fuction': self.tf_mlp_activation_fuction,
                    "tf_mlp_hidden_size_ratio": self.tf_mlp_hidden_size_ratio,
                    "tf_mlp_hidden_size_2_even": self.tf_mlp_hidden_size_2_even,
                    'dropout': self.tf_mlp_dropout,
                    'use_gated_norm': self.layer_kwargs.get('use_gated_norm', False),
                    'rmsnorm_impl': self.rmsnorm_impl,
                    'layer_norm_eps': self.layer_norm_eps,
                    'layer_recompute': self.layer_kwargs.get('layer_recompute', False),
                }
                y_kwargs = {
                    'in_features': self.embed_dim * self.y_token_k,
                    'hidden_size': self.hid_dim,
                    'out_features': self.embed_dim * self.y_token_k,
                    'has_bias': self.tf_mlp_use_bias,
                    'device': self.device,
                    'dtype': self.dtype,
                    'activation': self.activation,
                    'depth': mlp_info_dict['depth'],
                    'tf_mlp_layer_type': self.tf_mlp_layer_type,
                    'tf_mlp_activation_fuction': self.tf_mlp_activation_fuction,
                    "tf_mlp_hidden_size_ratio": self.tf_mlp_y_hidden_size_ratio if self.tf_mlp_y_hidden_size_ratio is not None else self.tf_mlp_hidden_size_ratio,
                    "tf_mlp_hidden_size_2_even": self.tf_mlp_hidden_size_2_even,
                    'dropout': self.tf_mlp_dropout,
                    'use_gated_norm': self.layer_kwargs.get('use_gated_norm', False),
                    'rmsnorm_impl': self.rmsnorm_impl,
                    'layer_norm_eps': self.layer_norm_eps,
                    'layer_recompute': self.layer_kwargs.get('layer_recompute', False),
                }
                x_ffn = None if self.cls_only else make_ffn(x_kwargs)
                y_ffn = make_ffn(y_kwargs)
                return SeparateXYFFN(
                    x_ffn=x_ffn,
                    y_ffn=y_ffn,
                    num_y_tokens=self.y_token_k,
                )

            return make_ffn(ffn_kwargs)

    def build_LN(self, idx: int, seed_name: str = 'LN'):
        with SetRandomSeed(self.init_seed_dict.get(f"layer{self.layer_idx}_{seed_name}{idx}_seed", None)):
            if self.tf_norm_type != 'rmsnorm':
                raise ValueError(f'unkown tf_norm_type: {self.tf_norm_type}')
            return build_rmsnorm(self.embed_dim,
                           eps=self.layer_norm_eps,
                           elementwise_affine=self.tf_layer_norm_use_elementwise_affine,
                           device=self.device,
                           dtype=self.dtype,
                           norm_impl=self.rmsnorm_impl,
                           recompute=self.layer_kwargs.get('layer_recompute', False))

    def parse_arch(self, arch_str: str):
        model_arch = {}
        FA = []
        SA = []
        MLP = []
        arch = ""
        arch_str = arch_str.upper()
        for token in arch_str.split("|"):
            parts = re.findall(r"[FSM]\d*", token)
            for p in parts:
                if p.startswith("M"):
                    depth = 2
                    if len(p) > 1:
                        depth = int(p[1:])
                    MLP.append({'depth': depth, 'type': "MLP"})
                    arch += 'M'
                elif p.startswith("F"):
                    FA.append({"type": "FA"})
                    arch += 'F'
                elif p.startswith("S"):
                    SA.append({"type": "SA"})
                    arch += 'S'
                else:
                    raise ValueError(f"unsupport arch: {arch_str}, {p} is not support")

        model_arch['FA'] = FA
        model_arch['SA'] = SA
        model_arch['MLP'] = MLP
        model_arch['arch'] = arch

        return model_arch

    def create_attn_mask(self, q_mask: torch.Tensor, k_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_mask (torch.Tensor): Query mask of shape [batch_size, seq_len, q_feature_num]. True marks masked positions.
            k_mask (torch.Tensor): Key mask of shape [batch_size, seq_len, k_feature_num]. True marks masked positions.
        Returns:
            torch.Tensor: Attention mask of shape [batch_size * seq_len, nhead, q_feature_num, k_feature_num]. True allows attention, False blocks it.
        """
        k_mask = k_mask.bool()  # [batch_size, seq_len, feature_num]

        k_expanded = k_mask.unsqueeze(-2)  # [batch_size, seq_len, 1, feature_num]
        attn_mask = ~k_expanded.expand(-1, -1, q_mask.shape[-1], -1)

        attn_mask = einops.rearrange(
            attn_mask,
            "b s f1 f2 -> (b s) f1 f2",
        )
        attn_mask = attn_mask.unsqueeze(1)  # [batch_size * seq_len, 1, feature_num, feature_num]
        attn_mask = attn_mask.expand(-1, self.nhead, -1, -1)  # [batch_size * seq_len, nhead, feature_num, feature_num]

        return attn_mask

    @nvtx.annotate('FA')
    def call_features_attention(
            self,
            x: torch.Tensor,
            feature_mask: torch.Tensor | None,
            eval_pos: int,
            index: int = 0,
            calculate_feature_attention: bool = False,
            y_type: torch.Tensor = None,
    ):
        assert len(self.feature_attentions) > index
        attention = self.feature_attentions[index]
        if isinstance(attention, DecoupledStructuralTaskAttention):
            x_feature_count = x.shape[2] - attention.num_y_tokens
            feature_padding_mask = None
            if feature_mask is not None:
                feature_padding_mask = feature_mask[..., :x_feature_count]
            return attention(
                x,
                feature_padding_mask=feature_padding_mask,
                calculate_feature_attention=calculate_feature_attention,
            )
        if not self.layer_kwargs.get('enable_feature_attention_mask', False):
            feature_mask = None
        attn_mask = None
        if feature_mask is not None:
            attn_mask = self.create_attn_mask(feature_mask, feature_mask)

        return attention(
            x,
            x_kv=None,
            attn_mask=attn_mask,
            calculate_feature_attention=calculate_feature_attention
        )

    @nvtx.annotate('IA')
    def call_sequence_attention(
            self,
            x: torch.Tensor,
            feature_mask: torch.Tensor | None,
            eval_pos: int,
            index: int = 0,
            calculate_sample_attention: bool = False,
            y_type: torch.Tensor = None,
    ):
        assert len(self.sequence_attentions) > index
        # ``feature_mask`` describes invalid cells/features for the F path.
        # The optimized all-query sample-attention path below does not accept
        # the corresponding train/test mask layout.  Historically this mask
        # was not forwarded here; keep that behavior while allowing DStI's
        # cross-sample summary to consume it in ``call_features_attention``.
        del feature_mask
        index1 = index * 2 if self.seq_attn_isolated else index
        assert index1 < len(self.sequence_attentions), \
            f'Error: index1({index1}) >= len(self.sequence_attentions)({len(self.sequence_attentions)})'
        assert 0 <= eval_pos <= x.shape[1]

        if eval_pos == x.shape[1]:
            print(f"\033[30;43mWarning: eval_pos >= x.shape[1]!\033[0m")

            # self-attn (Q_all * KV_all)
            out = self.sequence_attentions[index1](
                x=x.transpose(1, 2),
                x_kv=x.transpose(1, 2),
                y_type=y_type
            )[0].transpose(1, 2)

            return out, None, None

        if not (
            self.seq_attn_serial is False
            and self.self_share_all_kv_heads is False
            and self.cross_share_all_kv_heads is True
        ):
            raise ValueError(
                "Sample attention only supports seq_attn_serial=False, "
                "self_share_all_kv_heads=False, and cross_share_all_kv_heads=True"
            )

        # cross-attn (Q_all * KV_train)
        out, _, sample_attention = self.sequence_attentions[index1](
            x=x.transpose(1, 2),
            x_kv=x[:, :eval_pos].transpose(1, 2),
            calculate_sample_attention=calculate_sample_attention,
            y_type=y_type
        )
        out = out.transpose(1, 2)

        return out, None, sample_attention

    def forward(
            self,
            x: torch.Tensor,
            # feature_mask: torch.Tensor,
            eval_pos: int,
            **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        calculate_sample_attention = kwargs.get('calculate_sample_attention', False)
        calculate_feature_attention = kwargs.get('calculate_feature_attention', False)
        layer_idx = kwargs.get('layer_idx', 11)
        last_layer_idx = kwargs.get('last_layer_idx', 11)
        feature_attention = None
        sample_attention = None
        feature_mask = kwargs.get('feature_mask', None)
        y_type = kwargs.get('y_type', torch.ones_like(x[:, :, 0, 0]))
        for idx, (sublayer, layer_norm) in enumerate(zip(self.layer_steps, self.layer_norms)):
            if self.pre_norm:
                residual = x
                with nvtx.annotate('pre-norm'):
                    x = layer_norm(x)
                if idx == self.feature_attention_step_idx and calculate_feature_attention and layer_idx == last_layer_idx:
                    x, feature_attention, _ = sublayer(x, feature_mask, eval_pos, calculate_feature_attention=True,
                                                       y_type=y_type)
                elif idx == self.sample_attention_step_idx and calculate_sample_attention and layer_idx == last_layer_idx:
                    x, _, sample_attention = sublayer(x, feature_mask, eval_pos, calculate_sample_attention=True,
                                                      y_type=y_type)
                else:
                    if isinstance(sublayer, functools.partial):
                        x = sublayer(x, feature_mask, eval_pos, y_type=y_type)
                        if isinstance(x, tuple):
                            x = x[0]
                    else:
                        x = sublayer(x, y_type=y_type)
                        if isinstance(x, tuple):
                            x = x[0]
                if self.all_norm:
                    with nvtx.annotate('out-norm'):
                        x = self.output_layer_norms[idx](x)
                x = x + residual.to(dtype=x.dtype) if self.training else x.add_(residual.to(dtype=x.dtype))
            else:
                residual = x
                if idx == self.feature_attention_step_idx and calculate_feature_attention and layer_idx == last_layer_idx:
                    x, feature_attention, _ = sublayer(x, feature_mask, eval_pos, calculate_feature_attention=True,
                                                       y_type=y_type)
                    x = x + residual.to(dtype=x.dtype)
                elif idx == self.sample_attention_step_idx and calculate_sample_attention and layer_idx == last_layer_idx:
                    x, _, sample_attention = sublayer(x, feature_mask, eval_pos, calculate_sample_attention=True,
                                                      y_type=y_type)
                    x = x + residual.to(dtype=x.dtype)
                else:
                    if isinstance(sublayer, functools.partial):
                        x = sublayer(x, feature_mask, eval_pos, y_type=y_type)
                        if isinstance(x, tuple):
                            x = x[0]
                        x = x + residual.to(dtype=x.dtype)
                    else:
                        x = sublayer(x, y_type=y_type)
                        if isinstance(x, tuple):
                            x = x[0]
                        x = x + residual.to(dtype=x.dtype)
                try:
                    x = layer_norm(x)
                except OutOfMemoryError:
                    del residual
                    gc.collect()
                    torch.cuda.empty_cache()
                    x = layer_norm(x)

        return x, feature_attention, sample_attention


class LayerStack(nn.Module):
    """
    A flexible container module similar to ``nn.Sequential`` that allows
    keyword arguments to be passed through to each layer.

    Built from ``model_structure_config``; every layer must share the same ``emsize``.
    """

    def __init__(
            self,
            model_structure_config: dict | None = None,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
            init_seed_dict: dict = {},
            **common_kwargs,
    ):
        super().__init__()
        self.logger = get_logger('ldm', 'model')
        self.layer_recompute = common_kwargs.get('layer_recompute', False)
        self.layer_recompute_start_layer = common_kwargs.get('layer_recompute_start_layer', 0)
        self.num_cls_tokens = int(common_kwargs.get('num_cls_tokens', 1))
        if self.num_cls_tokens <= 0:
            raise ValueError("num_cls_tokens must be greater than 0")

        if model_structure_config is None:
            raise ValueError("必须提供 model_structure_config")
        self.layers = self._build_layers_from_config(
            model_structure_config,
            device=device,
            dtype=dtype,
            init_seed_dict=init_seed_dict,
            **common_kwargs,
        )

        self.feature_emb_layer = common_kwargs.get("feature_emb_layer", len(self.layers) - 1)
        self.reg_y_emb_layer = common_kwargs.get("reg_y_emb_layer", len(self.layers) - 1)
        self.cls_y_emb_layer = common_kwargs.get("cls_y_emb_layer", len(self.layers) - 1)
        cls_only_start_layer = common_kwargs.get('cls_only_start_layer', -1)
        cls_only_start_layer = -1 if cls_only_start_layer is None else int(cls_only_start_layer)
        if cls_only_start_layer < -1 or cls_only_start_layer >= len(self.layers):
            raise ValueError(
                "cls_only_start_layer must be -1 (disabled) or a valid zero-based "
                f"layer index in [0, {len(self.layers) - 1}], got {cls_only_start_layer}"
            )
        self.cls_only_start_layer = cls_only_start_layer

    def _build_layers_from_config(
            self,
            config: dict,
            device: torch.device | None,
            dtype: torch.dtype | None,
            init_seed_dict: dict,
            **common_kwargs,
    ) -> nn.ModuleList:
        """Build the layer list from the model structure config.

        Args:
            config: Model structure config dict containing a 'layers' list.
            device: Device.
            dtype: Data type.
            init_seed_dict: Initialization seed dict.
            **common_kwargs: Shared kwargs passed to every layer.

        Returns:
            Built layer list.
        """
        layers_config = config.get('layers', [])
        if not layers_config:
            raise ValueError("model_structure_config中的'layers'列表不能为空")

        built_layers = []

        for idx, layer_cfg in enumerate(layers_config):
            cls_only_start_layer = common_kwargs.get('cls_only_start_layer', -1)
            cls_only_start_layer = -1 if cls_only_start_layer is None else int(cls_only_start_layer)
            cls_only = cls_only_start_layer >= 0 and idx >= cls_only_start_layer
            layer_all_norm = layer_cfg.get('all_norm', common_kwargs.get('all_norm', False))
            layer_pre_norm = (
                    layer_cfg.get('pre_norm', common_kwargs.get('pre_norm', False))
                    or layer_all_norm
            )
            layer_params = {
                'layer_idx': idx,
                'embed_dim': layer_cfg['emsize'],
                'nhead': layer_cfg['nhead'],
                'hid_dim': layer_cfg['hid_dim'],
                'layer_arch': layer_cfg['arch'],
                'cls_only': cls_only,
                'pre_norm': layer_pre_norm,
                'all_norm': layer_all_norm,
                'activation': layer_cfg.get('activation', common_kwargs.get('activation', 'gelu')),
                'dropout': layer_cfg.get('dropout', common_kwargs.get('dropout', 0.0)),
                'layer_norm_eps': layer_cfg.get('layer_norm_eps', common_kwargs.get('layer_norm_eps', 1e-5)),
                'tf_norm_type': layer_cfg.get('tf_norm_type', common_kwargs.get('tf_norm_type', 'layer_norm')),
                'rmsnorm_impl': layer_cfg.get('rmsnorm_impl', common_kwargs.get('rmsnorm_impl', 'triton')),
                'tf_layer_norm_use_elementwise_affine': layer_cfg.get(
                    'tf_layer_norm_use_elementwise_affine',
                    common_kwargs.get('tf_layer_norm_use_elementwise_affine', False),
                ),
                'device': device,
                'dtype': dtype,
                'init_seed_dict': init_seed_dict,
                'flash_attention_precision': layer_cfg.get(
                    'flash_attention_precision',
                    common_kwargs.get('flash_attention_precision', 'inherit'),
                ),
                'induce_use_qk_norm': layer_cfg.get(
                    'induce_use_qk_norm',
                    layer_cfg.get(
                        'induce_use_qknorm',
                        common_kwargs.get(
                            'induce_use_qk_norm',
                            common_kwargs.get('induce_use_qknorm', False),
                        ),
                    ),
                ),
                'induce_qk_norm_eps': layer_cfg.get(
                    'induce_qk_norm_eps',
                    common_kwargs.get('induce_qk_norm_eps', 1e-5),
                ),
                'induce_qk_norm_elementwise_affine': layer_cfg.get(
                    'induce_qk_norm_elementwise_affine',
                    common_kwargs.get('induce_qk_norm_elementwise_affine', True),
                ),
                'induce_use_softmax_scaling_mlp': layer_cfg.get(
                    'induce_use_softmax_scaling_mlp',
                    common_kwargs.get('induce_use_softmax_scaling_mlp', False),
                ),
                'induce_softmax_scaling_temp_upper_bound': layer_cfg.get(
                    'induce_softmax_scaling_temp_upper_bound',
                    common_kwargs.get('induce_softmax_scaling_temp_upper_bound', 0.5),
                ),
                'induce_softmax_scaling_temp_lower_bound': layer_cfg.get(
                    'induce_softmax_scaling_temp_lower_bound',
                    common_kwargs.get('induce_softmax_scaling_temp_lower_bound', 0.0),
                ),
                'induce_softmax_scaling_base_bound': layer_cfg.get(
                    'induce_softmax_scaling_base_bound',
                    common_kwargs.get('induce_softmax_scaling_base_bound', 5.0),
                ),
                "layer_recompute": self.layer_recompute,
                "separate_x_y_ffn": common_kwargs.get("separate_x_y_ffn", False),
                "use_separate_attention": common_kwargs.get("use_separate_attention", False),
                "separate_attn_kv_combined": common_kwargs.get("separate_attn_kv_combined", False),
                "sample_attention_cls_token_merge": layer_cfg.get(
                    "sample_attention_cls_token_merge",
                    common_kwargs.get("sample_attention_cls_token_merge", "reshape"),
                ),
                "scale_y_sample_attention_heads": layer_cfg.get(
                    "scale_y_sample_attention_heads",
                    common_kwargs.get("scale_y_sample_attention_heads", True),
                ),
                "cls_sample_attention_num_heads": layer_cfg.get(
                    "cls_sample_attention_num_heads",
                    common_kwargs.get("cls_sample_attention_num_heads", 0),
                ),
                "cls_only_sample_attention_num_heads": layer_cfg.get(
                    "cls_only_sample_attention_num_heads",
                    common_kwargs.get("cls_only_sample_attention_num_heads", 0),
                ),
                "feature_attention_cls_token_merge": layer_cfg.get(
                    "feature_attention_cls_token_merge",
                    common_kwargs.get("feature_attention_cls_token_merge", "none"),
                ),
                "feature_attention_pipeline": layer_cfg.get(
                    "feature_attention_pipeline",
                    common_kwargs.get("feature_attention_pipeline", "legacy"),
                ),
                "decoupled_attn_kv_combined": layer_cfg.get(
                    "decoupled_attn_kv_combined",
                    common_kwargs.get("decoupled_attn_kv_combined", False),
                ),
                "dsti_relation_dim": layer_cfg.get(
                    "dsti_relation_dim",
                    common_kwargs.get("dsti_relation_dim", 0),
                ),
                "dsti_value_dim": layer_cfg.get(
                    "dsti_value_dim",
                    common_kwargs.get("dsti_value_dim", 0),
                ),
                "dsti_num_heads": layer_cfg.get(
                    "dsti_num_heads",
                    common_kwargs.get("dsti_num_heads", 0),
                ),
                "dsti_feature_interaction_mode": layer_cfg.get(
                    "dsti_feature_interaction_mode",
                    common_kwargs.get(
                        "dsti_feature_interaction_mode", "sample_split"
                    ),
                ),
                "dsti_y_token_aggregation": layer_cfg.get(
                    "dsti_y_token_aggregation",
                    common_kwargs.get(
                        "dsti_y_token_aggregation", "concat_linear"
                    ),
                ),
                "dsti_dropout": layer_cfg.get(
                    "dsti_dropout",
                    common_kwargs.get("dsti_dropout", -1.0),
                ),
                "dsti_use_qk_norm": layer_cfg.get(
                    "dsti_use_qk_norm",
                    common_kwargs.get("dsti_use_qk_norm", False),
                ),
                "dsti_qk_norm_eps": layer_cfg.get(
                    "dsti_qk_norm_eps",
                    common_kwargs.get("dsti_qk_norm_eps", 1e-5),
                ),
                "dsti_qk_norm_elementwise_affine": layer_cfg.get(
                    "dsti_qk_norm_elementwise_affine",
                    common_kwargs.get(
                        "dsti_qk_norm_elementwise_affine", True
                    ),
                ),
                "dsti_use_softmax_scaling_mlp": layer_cfg.get(
                    "dsti_use_softmax_scaling_mlp",
                    common_kwargs.get("dsti_use_softmax_scaling_mlp", False),
                ),
                "dsti_softmax_scaling_temp_upper_bound": layer_cfg.get(
                    "dsti_softmax_scaling_temp_upper_bound",
                    common_kwargs.get(
                        "dsti_softmax_scaling_temp_upper_bound", 0.4
                    ),
                ),
                "dsti_softmax_scaling_temp_lower_bound": layer_cfg.get(
                    "dsti_softmax_scaling_temp_lower_bound",
                    common_kwargs.get(
                        "dsti_softmax_scaling_temp_lower_bound", 0.0
                    ),
                ),
                "dsti_softmax_scaling_base_bound": layer_cfg.get(
                    "dsti_softmax_scaling_base_bound",
                    common_kwargs.get("dsti_softmax_scaling_base_bound", 1.0),
                ),
                "dsti_attention_backend": layer_cfg.get(
                    "dsti_attention_backend",
                    common_kwargs.get("dsti_attention_backend", "auto"),
                ),
                "sa_temp_upper_bound": common_kwargs.get(
                    "sa_temp_upper_bound",
                    0.5
                ),
                "sa_temp_lower_bound": common_kwargs.get(
                    "sa_temp_lower_bound",
                    0
                ),
                "sa_scale_base_bound": common_kwargs.get(
                    "sa_scale_base_bound",
                    5.0
                ),
            }
            for key, value in common_kwargs.items():
                if key not in layer_params:
                    layer_params[key] = value

            standard_keys = {
                'layer_idx', 'arch', 'emsize', 'nhead', 'nhid_factor', 'hid_dim',
                'activation', 'dropout', 'pre_norm', 'all_norm',
                'layer_norm_eps', 'tf_norm_type', 'rmsnorm_impl', 'tf_layer_norm_use_elementwise_affine',
                'induce_use_qk_norm', 'induce_use_qknorm', 'induce_qk_norm_eps',
                'induce_qk_norm_elementwise_affine', 'induce_use_softmax_scaling_mlp',
                'induce_softmax_scaling_temp_upper_bound',
                'induce_softmax_scaling_temp_lower_bound',
                'induce_softmax_scaling_base_bound',
                'separate_attn_kv_combined',
            }
            for key, value in layer_cfg.items():
                if key not in standard_keys and key not in layer_params:
                    layer_params[key] = value

            layer = EncoderBaseLayer(**layer_params)
            built_layers.append(layer)
            self.logger.debug(
                f"Built layer {idx}: emsize={layer_cfg['emsize']}, "
                f"nhead={layer_cfg['nhead']}, arch={layer_cfg['arch']}"
            )

        return nn.ModuleList(built_layers)

    def forward(self, x, **kwargs):
        nlayers = len(self.layers)
        feature_emb = None
        reg_y_emb = None
        cls_y_emb = None
        for idx, layer in enumerate(self.layers):
            if idx == self.cls_only_start_layer:
                if x.shape[2] < self.num_cls_tokens:
                    raise ValueError(
                        "Encoder state has fewer tokens than num_cls_tokens at "
                        f"the CLS-only boundary: {x.shape[2]} < {self.num_cls_tokens}"
                    )
                # CLS tokens are trailing. Drop X and any target tokens so no
                # later operation can read or retain their representations.
                x = x[:, :, -self.num_cls_tokens:, :].contiguous()
            with nvtx.annotate('layer'):
                kwargs['layer_idx'] = idx
                kwargs['last_layer_idx'] = nlayers - 1
                if self.training and self.layer_recompute and idx >= self.layer_recompute_start_layer:
                    x, feature_attention, sample_attention = checkpoint(layer, x, **kwargs, use_reentrant=False)
                else:
                    x, feature_attention, sample_attention = layer(x, **kwargs)

            if idx == self.feature_emb_layer and idx < nlayers - 1:
                # NOTE: offload to CPU to reduce memory footprint for inference
                feature_emb = x.clone() if self.training else x.cpu()
            if idx == self.reg_y_emb_layer and idx < nlayers - 1:
                reg_y_emb = x[:, :, -self.num_cls_tokens:, :].clone()
            if idx == self.cls_y_emb_layer and idx < nlayers - 1:
                cls_y_emb = x[:, :, -self.num_cls_tokens:, :].clone()

        if feature_emb is not None:
            feature_emb = feature_emb.to(x.device)
        return x, feature_attention, sample_attention, feature_emb, reg_y_emb, cls_y_emb
