"""Decoupled structural and task interaction for tabular feature attention.

The module consumes the repository's combined token layout
``[batch, sample, x_feature + y_token, embedding]`` and runs per-sample
asymmetric feature attention (``sample_split``): X queries all [X,Y] while
each Y token queries X only.  No cross-sample score is broadcast.

This is a drop-in feature-attention sublayer and preserves the outer
residual/norm contract owned by :class:`EncoderBaseLayer`.
"""

from __future__ import annotations

import nvtx
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_precision import (
    cast_flash_attention_inputs,
    normalize_flash_attention_precision,
    resolve_flash_attention_dtype,
)
from .softmax_scaling_mlp import SoftmaxScalingMLP
from .utils import sdpa_context, simple_autobatch
from .autobatch import autobatch

try:
    from .operators.rmsnorm import build_rmsnorm, RMSNormMixedPrecision, TritonQKNorm
except (ImportError, ModuleNotFoundError):
    build_rmsnorm = None

try:
    from flash_attn import flash_attn_func as _flash_attn_func

    HAVE_FLASH_ATTN = True
except (ImportError, ModuleNotFoundError):
    _flash_attn_func = None
    HAVE_FLASH_ATTN = False


class _TorchQKNormFallback(nn.Module):
    """State-compatible CPU fallback when Triton is not importable."""

    def __init__(
        self,
        hs: int,
        eps: float,
        elementwise_affine: bool,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        self.hs = hs
        self.eps = eps
        self.weight = (
            nn.Parameter(torch.ones(hs, device=device, dtype=dtype))
            if elementwise_affine
            else None
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(value, (self.hs,), self.weight, self.eps)


class DecoupledStructuralTaskAttention(nn.Module):
    """Run per-sample asymmetric feature attention (``sample_split``).

    ``embed_dim`` is the per-token semantic width.  A wider Y semantic
    representation is carried by multiple trailing Y tokens.  Every token
    keeps an independent relation query and semantic readout; the readouts are
    aggregated only after cross-attention.

    The returned tensor is an *attention update*.  Residual addition and the
    layer norm remain outside this module.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        num_y_tokens: int = 1,
        relation_dim: int | None = None,
        value_dim: int | None = None,
        feature_interaction_mode: str = "sample_split",
        y_token_aggregation: str = "concat_linear",
        dropout: float = 0.0,
        bias: bool = False,
        use_qk_norm: bool = False,
        rmsnorm_impl: str = 'triton',
        qk_norm_eps: float = 1e-5,
        qk_norm_elementwise_affine: bool = True,
        use_softmax_scaling_mlp: bool = False,
        softmax_scaling_temp_upper_bound: float = 0.4,
        softmax_scaling_temp_lower_bound: float = 0.0,
        softmax_scaling_base_bound: float = 1.0,
        deterministic: bool = False,
        recompute: bool = False,
        attention_backend: str = "auto",
        flash_attention_precision: str = "inherit",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        kv_combined: bool = False,
    ) -> None:
        super().__init__()
        relation_dim = embed_dim if relation_dim is None else int(relation_dim)
        value_dim = embed_dim if value_dim is None else int(value_dim)

        dimensions = {
            "embed_dim": embed_dim,
            "relation_dim": relation_dim,
            "value_dim": value_dim,
        }
        for name, dimension in dimensions.items():
            if dimension <= 0:
                raise ValueError(f"{name} must be greater than 0")
        if num_heads <= 0:
            raise ValueError("num_heads must be greater than 0")
        if num_y_tokens <= 0:
            raise ValueError("num_y_tokens must be greater than 0")
        feature_interaction_mode = feature_interaction_mode.lower()
        if feature_interaction_mode != "sample_split":
            raise ValueError(
                "feature_interaction_mode must be 'sample_split', "
                f"got {feature_interaction_mode!r}"
            )
        y_token_aggregation = y_token_aggregation.lower()
        if y_token_aggregation != "concat_linear":
            raise ValueError(
                "y_token_aggregation must be 'concat_linear', "
                f"got {y_token_aggregation!r}"
            )
        for name in ("relation_dim", "value_dim"):
            if dimensions[name] % num_heads != 0:
                raise ValueError(
                    f"{name}={dimensions[name]} must be divisible by "
                    f"num_heads={num_heads}"
                )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if qk_norm_eps <= 0:
            raise ValueError("qk_norm_eps must be greater than 0")
        if softmax_scaling_temp_upper_bound <= softmax_scaling_temp_lower_bound:
            raise ValueError(
                "softmax_scaling_temp_upper_bound must be greater than "
                "softmax_scaling_temp_lower_bound"
            )
        if softmax_scaling_base_bound <= 0:
            raise ValueError("softmax_scaling_base_bound must be greater than 0")
        if not use_qk_norm:
            raise ValueError("use_qk_norm must be True")
        if not use_softmax_scaling_mlp:
            raise ValueError("use_softmax_scaling_mlp must be True")
        attention_backend = attention_backend.lower()
        if attention_backend != "auto":
            raise ValueError(
                "attention_backend must be 'auto', "
                f"got {attention_backend!r}"
            )
        if relation_dim != value_dim:
            raise ValueError(
                "relation_dim must equal value_dim, "
                f"got relation_dim={relation_dim}, value_dim={value_dim}"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_y_tokens = num_y_tokens
        self.relation_dim = relation_dim
        self.value_dim = value_dim
        self.feature_interaction_mode = feature_interaction_mode
        self.y_token_aggregation = y_token_aggregation
        self.relation_head_dim = relation_dim // num_heads
        self.value_head_dim = value_dim // num_heads
        self.dropout_p = dropout
        self.use_qk_norm = use_qk_norm
        self.rmsnorm_impl = rmsnorm_impl
        self.qk_norm_eps = qk_norm_eps
        self.qk_norm_elementwise_affine = qk_norm_elementwise_affine
        self.use_softmax_scaling_mlp = use_softmax_scaling_mlp
        self.deterministic = deterministic
        self.recompute = recompute
        self.attention_backend = attention_backend
        self.flash_attention_precision = normalize_flash_attention_precision(
            flash_attention_precision
        )
        self._flash_runtime_fallback_warned = False
        self._flash_disabled_reason: str | None = None
        if kv_combined:
            raise ValueError("kv_combined must be False")

        factory_kwargs = {"device": device, "dtype": dtype}

        # Per-sample asymmetric feature attention:
        #   X queries all [X,Y] keys/values;
        #   Y queries only X keys/values.
        self.sample_x_q_proj = nn.Linear(
            embed_dim, relation_dim, bias=bias, **factory_kwargs
        )
        self.sample_all_k_proj = nn.Linear(
            embed_dim, relation_dim, bias=bias, **factory_kwargs
        )
        self.sample_all_v_proj = nn.Linear(
            embed_dim, value_dim, bias=bias, **factory_kwargs
        )
        self.sample_x_out_proj = nn.Linear(
            value_dim, embed_dim, bias=bias, **factory_kwargs
        )

        # Each Y token retains its own representation, relation query, and X
        # readout.  The output projection concatenates all readouts and mixes
        # them jointly before restoring the Y-token axis.
        self.yx_y_q_proj = nn.Linear(
            embed_dim, relation_dim, bias=bias, **factory_kwargs
        )
        self.yx_x_k_proj = nn.Linear(
            embed_dim, relation_dim, bias=bias, **factory_kwargs
        )
        self.yx_x_v_proj = nn.Linear(
            embed_dim, value_dim, bias=bias, **factory_kwargs
        )
        self.yx_out_proj = nn.Linear(
            num_y_tokens * value_dim,
            num_y_tokens * embed_dim,
            bias=bias,
            **factory_kwargs,
        )

        self.sample_x_q_norm = self._make_qk_norm(
            self.relation_head_dim, factory_kwargs
        )
        self.sample_all_k_norm = self._make_qk_norm(
            self.relation_head_dim, factory_kwargs
        )
        self.yx_q_norm = self._make_qk_norm(
            self.relation_head_dim, factory_kwargs
        )
        self.yx_k_norm = self._make_qk_norm(
            self.relation_head_dim, factory_kwargs
        )

        scaling_kwargs = {
            "num_heads": num_heads,
            "temp_upper_bound": softmax_scaling_temp_upper_bound,
            "temp_lower_bound": softmax_scaling_temp_lower_bound,
            "scale_base_bound": softmax_scaling_base_bound,
        }
        self.sample_x_softmax_scaling = SoftmaxScalingMLP(
            head_dim=self.relation_head_dim, **scaling_kwargs
        )
        self.yx_softmax_scaling = SoftmaxScalingMLP(
            head_dim=self.relation_head_dim, **scaling_kwargs
        )
        self.sample_x_softmax_scaling.to(device=device, dtype=dtype)
        self.yx_softmax_scaling.to(device=device, dtype=dtype)

    def _make_qk_norm(
        self,
        head_dim: int,
        factory_kwargs: dict,
    ) -> nn.Module:
        if build_rmsnorm is not None:
            norm = build_rmsnorm(
                head_dim,
                eps=self.qk_norm_eps,
                elementwise_affine=self.qk_norm_elementwise_affine,
                norm_impl=self.rmsnorm_impl,
                norm_type='qknorm',
                recompute=self.recompute,
                **factory_kwargs,
            )
            norm.to(
                device=factory_kwargs.get("device"),
                dtype=factory_kwargs.get("dtype"),
            )
            return norm
        return _TorchQKNormFallback(
            hs=head_dim,
            eps=self.qk_norm_eps,
            elementwise_affine=self.qk_norm_elementwise_affine,
            **factory_kwargs,
        )

    @staticmethod
    def _run_qk_norm(norm: nn.Module, value: torch.Tensor) -> torch.Tensor:
        # TritonQKNorm is CUDA-only.  The functional branch keeps CPU tests and
        # explicit SDPA fallback numerically aligned while sharing its weight.
        if build_rmsnorm is not None and \
           (isinstance(norm, RMSNormMixedPrecision) or isinstance(norm, TritonQKNorm)):
            if not value.is_cuda:
                weight = norm.weight
                if weight is not None:
                    weight = weight.to(device=value.device, dtype=value.dtype)
                return F.rms_norm(value, (norm.hs,), weight, norm.eps)
        return norm(value)

    def _flash_eligibility(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[bool, str]:
        if self._flash_disabled_reason is not None:
            return False, self._flash_disabled_reason
        if not HAVE_FLASH_ATTN or _flash_attn_func is None:
            return False, "flash-attn is not installed"
        if attention_mask is not None:
            return False, "flash_attn_func does not support arbitrary masks"
        if not query.is_cuda:
            return False, "FlashAttention requires CUDA/ROCm tensors"
        flash_dtype = resolve_flash_attention_dtype(
            self.flash_attention_precision,
            query.dtype,
        )
        if flash_dtype not in {torch.float16, torch.bfloat16}:
            return False, "FlashAttention requires fp16 or bf16 tensors"
        if key.device != query.device or value.device != query.device:
            return False, "Q, K, and V must be on the same device"
        if (
            self.flash_attention_precision == "inherit"
            and (key.dtype != query.dtype or value.dtype != query.dtype)
        ):
            return False, "Q, K, and V must have the same dtype"
        if query.shape[-1] != key.shape[-1]:
            return False, "Q and K head dimensions must match"
        if query.shape[-1] > 256:
            return False, "FlashAttention-2 supports head dimensions up to 256"
        return True, ""

    @simple_autobatch(batch_size=65535, batch_dim=0)
    def _flash_attn_func_wrapper(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs):
        return _flash_attn_func(q, k, v, **kwargs)

    def _flash_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        if _flash_attn_func is None:
            raise RuntimeError("flash-attn is not installed")
        output_dtype = value.dtype
        query, key, value = cast_flash_attention_inputs(
            self.flash_attention_precision,
            query,
            key,
            value,
        )
        assert query is not None and key is not None and value is not None
        query_head_dim = query.shape[-1]
        if query_head_dim != value.shape[-1]:
            raise ValueError("Q and V head dimensions must match")
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        output = self._flash_attn_func_wrapper(
            query,
            key,
            value,
            dropout_p=self.dropout_p if self.training else 0.0,
            softmax_scale=query_head_dim ** -0.5,
            causal=False,
            deterministic=self.deterministic,
        )
        return output.to(dtype=output_dtype)

    @simple_autobatch(batch_size=65535, batch_dim=0)
    def _sdpa_attn_func_wrapper(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs):
        return F.scaled_dot_product_attention(q, k, v, **kwargs)

    def _sdpa_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        with sdpa_context(self.deterministic):
            output = self._sdpa_attn_func_wrapper(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                attn_mask=attention_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
            )
        return output.transpose(1, 2)

    def _attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        eligible, _reason = self._flash_eligibility(
            query, key, value, attention_mask
        )
        if eligible:
            try:
                return self._flash_attention(query, key, value)
            except (RuntimeError, AssertionError) as error:
                if isinstance(error, torch.cuda.OutOfMemoryError):
                    raise
                self._flash_disabled_reason = (
                    f"previous FlashAttention runtime failure: {error}"
                )
                if not self._flash_runtime_fallback_warned:
                    warnings.warn(
                        "DStI FlashAttention failed at runtime; falling back "
                        f"to PyTorch SDPA. Original error: {error}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._flash_runtime_fallback_warned = True
        return self._sdpa_attention(query, key, value, attention_mask)

    def _aggregate_y_readouts(self, readout: torch.Tensor) -> torch.Tensor:
        batch_size, sample_count, y_token_count, _ = readout.shape
        return self.yx_out_proj(
            readout.flatten(start_dim=-2)
        ).reshape(
            batch_size,
            sample_count,
            y_token_count,
            self.embed_dim,
        )

    def _x_route_linear_qkv(
        self,
        x_tokens: torch.Tensor,
        y_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, sample_count, feature_count, _ = x_tokens.shape
        y_token_count = y_tokens.shape[2]
        total_count = feature_count + y_token_count

        # linear-q
        flat_x = x_tokens.reshape(batch_size * sample_count, feature_count, self.embed_dim)
        with nvtx.annotate('linear-q'):
            query = self.sample_x_q_proj(flat_x).view(
                batch_size * sample_count,
                feature_count,
                self.num_heads,
                self.relation_head_dim,
            )
        del flat_x

        # linear-kv
        flat_all = torch.cat((x_tokens, y_tokens), dim=2).reshape(
            batch_size * sample_count, total_count, self.embed_dim
        )
        with nvtx.annotate('linear-k'):
            key = self.sample_all_k_proj(flat_all).view(
                batch_size * sample_count,
                total_count,
                self.num_heads,
                self.relation_head_dim,
            )
        with nvtx.annotate('linear-v'):
            value = self.sample_all_v_proj(flat_all).view(
                batch_size * sample_count,
                total_count,
                self.num_heads,
                self.value_head_dim,
            )
        del flat_all

        with nvtx.annotate('qknorm'):
            query = self._run_qk_norm(self.sample_x_q_norm, query)
            key = self._run_qk_norm(self.sample_all_k_norm, key)

        with nvtx.annotate('sm-scaling'):
            query = self.sample_x_softmax_scaling(query, total_count)

        return query, key, value

    def _x_route_core_attn(
        self,
        x_tokens: torch.Tensor,
        feature_padding_mask: torch.Tensor | None,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sample_count, feature_count, _ = x_tokens.shape
        y_token_count = key.shape[1] - feature_count

        attention_mask = None
        if feature_padding_mask is not None:
            flat_feature_mask = feature_padding_mask.reshape(
                batch_size * sample_count, feature_count
            )
            valid_keys = torch.cat(
                (
                    ~flat_feature_mask,
                    torch.ones(
                        batch_size * sample_count,
                        y_token_count,
                        dtype=torch.bool,
                        device=x_tokens.device,
                    ),
                ),
                dim=1,
            )
            attention_mask = valid_keys[:, None, None, :]

        mixed = self._attention(query, key, value, attention_mask).reshape(
            batch_size, sample_count, feature_count, self.value_dim,
        )
        return mixed

    @autobatch(batch_dim=1)
    def _chunk_sample_split_x_route(
        self,
        x_tokens: torch.Tensor,
        y_tokens: torch.Tensor,
        feature_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        with nvtx.annotate('linear-qkv'):
            query, key, value = self._x_route_linear_qkv(x_tokens, y_tokens)

        with nvtx.annotate('core-attn'):
            mixed = self._x_route_core_attn(
                x_tokens, feature_padding_mask, query, key, value
            )
        del query, key, value

        with nvtx.annotate('linear-out'):
            update = self.sample_x_out_proj(mixed)
        if feature_padding_mask is not None:
            update = update.masked_fill(feature_padding_mask[..., None], 0.0)

        return update

    def _sample_split_x_route(
        self,
        x_tokens: torch.Tensor,
        y_tokens: torch.Tensor,
        feature_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        return self._chunk_sample_split_x_route(x_tokens, y_tokens, feature_padding_mask)

    def _yx_route_linear_qkv(
        self,
        y_tokens: torch.Tensor,
        x_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, sample_count, y_token_count, _ = y_tokens.shape
        feature_count = x_tokens.shape[2]

        # linear-q
        flat_y = y_tokens.reshape(batch_size * sample_count, y_token_count, self.embed_dim)
        with nvtx.annotate('linear-q'):
            query = self.yx_y_q_proj(flat_y).view(
                batch_size * sample_count,
                y_token_count,
                self.num_heads,
                self.relation_head_dim,
            )
        del flat_y

        # linear-kv
        flat_x = x_tokens.reshape(batch_size * sample_count, feature_count, self.embed_dim)
        with nvtx.annotate('linear-k'):
            key = self.yx_x_k_proj(flat_x).view(
                batch_size * sample_count,
                feature_count,
                self.num_heads,
                self.relation_head_dim,
            )
        with nvtx.annotate('linear-v'):
            value = self.yx_x_v_proj(flat_x).view(
                batch_size * sample_count,
                feature_count,
                self.num_heads,
                self.value_head_dim,
            )
        del flat_x

        with nvtx.annotate('qknorm'):
            query = self._run_qk_norm(self.yx_q_norm, query)
            key = self._run_qk_norm(self.yx_k_norm, key)

        with nvtx.annotate('sm-scaling'):
            query = self.yx_softmax_scaling(query, feature_count)

        return query, key, value

    def _yx_route_core_attn(
        self,
        y_tokens: torch.Tensor,
        feature_padding_mask: torch.Tensor | None,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sample_count, y_token_count, _ = y_tokens.shape
        feature_count = key.shape[1]
        attention_mask = (
            (~feature_padding_mask).reshape(
                batch_size * sample_count, 1, 1, feature_count
            )
            if feature_padding_mask is not None
            else None
        )

        readout = self._attention(query, key, value, attention_mask).reshape(
            batch_size, sample_count, y_token_count, self.value_dim
        )
        return readout

    @autobatch(batch_dim=1)
    def _chunk_sample_split_yx_route(
        self,
        y_tokens: torch.Tensor,
        x_tokens: torch.Tensor,
        feature_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        with nvtx.annotate('linear-qkv'):
            query, key, value = self._yx_route_linear_qkv(y_tokens, x_tokens)

        with nvtx.annotate('core-attn'):
            readout = self._yx_route_core_attn(
                y_tokens, feature_padding_mask, query, key, value
            )
        del query, key, value

        with nvtx.annotate('linear-out'):
            update = self._aggregate_y_readouts(readout)

        return update

    def _sample_split_yx_route(
        self,
        y_tokens: torch.Tensor,
        x_tokens: torch.Tensor,
        feature_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        return self._chunk_sample_split_yx_route(y_tokens, x_tokens, feature_padding_mask)

    def forward(
        self,
        x: torch.Tensor,
        x_kv: torch.Tensor | None = None,
        *,
        feature_padding_mask: torch.Tensor | None = None,
        calculate_feature_attention: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, None]:
        del x_kv, calculate_feature_attention, kwargs
        if x.ndim != 4:
            raise ValueError(
                "DStI input must be [batch,sample,token,embedding], got "
                f"{tuple(x.shape)}"
            )
        if x.shape[-1] != self.embed_dim:
            raise ValueError(
                f"last dimension must equal embed_dim={self.embed_dim}"
            )
        if x.shape[2] <= self.num_y_tokens:
            raise ValueError(
                "DStI expects at least one X feature before the trailing "
                f"{self.num_y_tokens} Y token(s)"
            )

        feature_count = x.shape[2] - self.num_y_tokens
        x_tokens = x[:, :, :feature_count]
        y_tokens = x[:, :, feature_count:]
        if feature_padding_mask is not None:
            expected_shape = x_tokens.shape[:3]
            if tuple(feature_padding_mask.shape) != tuple(expected_shape):
                raise ValueError(
                    "feature_padding_mask must have shape "
                    f"{tuple(expected_shape)}, got "
                    f"{tuple(feature_padding_mask.shape)}"
                )
            feature_padding_mask = feature_padding_mask.bool()
            if not torch.any(feature_padding_mask):
                # Avoid disabling FlashAttention for an all-valid batch.
                feature_padding_mask = None

        with nvtx.annotate('sample-x'):
            x_update = self._sample_split_x_route(
                x_tokens,
                y_tokens,
                feature_padding_mask,
            )
        # Use the original X values so the Y branch has a strict Y->X
        # key/value path and cannot read Y indirectly through X updates.
        with nvtx.annotate('sample-yx'):
            y_update = self._sample_split_yx_route(
                y_tokens,
                x_tokens,
                feature_padding_mask,
            )

        with nvtx.annotate('concat'):
            out = torch.cat((x_update, y_update), dim=2)
        return (
            out,
            None,
            None,
        )

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"num_y_tokens={self.num_y_tokens}, "
            f"feature_interaction_mode={self.feature_interaction_mode!r}, "
            f"y_token_aggregation={self.y_token_aggregation!r}, "
            f"relation_dim={self.relation_dim}, value_dim={self.value_dim}, "
            f"use_qk_norm={self.use_qk_norm}, "
            f"use_softmax_scaling_mlp={self.use_softmax_scaling_mlp}, "
            f"attention_backend={self.attention_backend!r}"
        )
