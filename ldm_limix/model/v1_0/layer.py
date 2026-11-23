import gc
from typing import Callable, Literal, Optional, Any
import functools

import nvtx
import torch
import torch.nn as nn
from torch.cuda import OutOfMemoryError
from torch.utils.checkpoint import checkpoint
from functools import partial
from torch.amp import autocast
from .utils import SetRandomSeed, simple_autobatch, get_logger
from torch.nn.attention import SDPBackend, sdpa_kernel
import re
from .autobatch import AutobatchConfig, autobatch, _cuda_free_bytes, _is_retryable_cuda_error
import math
from .operators.triton_rmsnorm import TritonRMSNorm


try:
    from flash_attn.flash_attn_interface import (
        flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_qkvpacked_func,
        flash_attn_qkvpacked_func,
    )
    from flash_attn import flash_attn_func
    HAVE_FLASH_ATTN = True
except (ModuleNotFoundError, ImportError):
    HAVE_FLASH_ATTN = False


def _sdpa_context(deterministic: bool):
    backends = [
        SDPBackend.MATH,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.CUDNN_ATTENTION,
    ]
    if not deterministic:
        backends = [SDPBackend.FLASH_ATTENTION, *backends]
    return sdpa_kernel(backends)


from typing_extensions import override

Activation = Literal['gelu']

ACTIVATION_FN: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    'gelu': nn.GELU(),
}


class LayerNormMixedPrecision(nn.LayerNorm):
    """
    When the embedding dimension is below 512, use half precision for computation to improve performance. 
    If the embedding dimension exceeds 512, it may cause training instability.
    """
    @nvtx.annotate('LN')
    def forward(self, input: torch.Tensor):
        if input.dtype == torch.float16 and sum(self.normalized_shape) < 512:
            with autocast(device_type="cuda" if input.is_cuda else "cpu", enabled=False):
                return self._forward(input)
        else:
            return self._forward(input)
    
    @autobatch(batch_dim=1)    
    def _forward(self, input: torch.Tensor):
        if self.elementwise_affine and input.dtype != self.weight.dtype:
            input = input.to(self.weight.dtype)
        return super().forward(input)


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
         init_std: float = 0,
         residual_scale: float = 1.0,
         **kwargs,
    ):
        super().__init__()

        self.depth = depth
        self.hidden_size = hidden_size
        self.activation = activation
        self.residual_scale = residual_scale
        self.tf_mlp_layer_type = kwargs.get('tf_mlp_layer_type', 'normal')
        self.tf_mlp_activation_fuction = kwargs.get('tf_mlp_activation_fuction', 'gelu')
        self.dropout_prob = kwargs.get('dropout', 0.0)
        self.dropout = nn.Dropout(self.dropout_prob) if self.dropout_prob > 0 else None
        self.kwargs = kwargs
        

        if 'normal' == self.tf_mlp_layer_type:
            self.layers = []
            if depth == 1:
                self.layers.append(nn.Linear(in_features, out_features, bias=has_bias, device=device, dtype=dtype))
            else:
                # input layer
                self.layers.append(nn.Linear(in_features, hidden_size, bias=has_bias, device=device, dtype=dtype))
                self.layers.append(ACTIVATION_FN[self.activation])
                if self.dropout is not None:
                    self.layers.append(self.dropout)

                # hidden layers
                for _ in range(2, depth):
                    self.layers.append(nn.Linear(hidden_size, hidden_size, bias=has_bias, device=device, dtype=dtype))
                    self.layers.append(ACTIVATION_FN[self.activation])
                    if self.dropout is not None:
                        self.layers.append(self.dropout)

                # output layer
                self.layers.append(nn.Linear(hidden_size, out_features, bias=has_bias, device=device, dtype=dtype))
            self.mlp = nn.Sequential(*self.layers)
        elif 'gated' == self.tf_mlp_layer_type:
            hidden_size = int(in_features * self.kwargs.get('tf_mlp_hidden_size_ratio', 4.0))
            self.gate_proj = nn.Linear(in_features, hidden_size, bias=has_bias, device=device, dtype=dtype)
            self.up_proj = nn.Linear(in_features, hidden_size, bias=has_bias, device=device, dtype=dtype)
            self.down_proj = nn.Linear(hidden_size, out_features, bias=has_bias, device=device, dtype=dtype)
            self.act_fn = ACTIVATION_FN[self.tf_mlp_activation_fuction]
            self.dropout_after_act = self.dropout
        else:
            raise ValueError(f"Unknown tf_mlp_layer_type: {self.tf_mlp_layer_type}")


    @nvtx.annotate('MLP')
    @autobatch(batch_dim=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if 'normal' == self.tf_mlp_layer_type:
            return self.mlp(x)
        elif 'gated' == self.tf_mlp_layer_type:
            with nvtx.annotate('gate-proj'):
                gate_out = self.gate_proj(x)
            with nvtx.annotate('up-proj'):
                up_out = self.up_proj(x)
            with nvtx.annotate('left-act-and-mul'):
                activated = self.act_fn(gate_out) * up_out
                if self.dropout_after_act is not None:
                    activated = self.dropout_after_act(activated)
            with nvtx.annotate('down-proj'):
                out = self.down_proj(activated)
            return out
        else:
            raise ValueError(f"Unknown tf_mlp_layer_type: {self.tf_mlp_layer_type}")

class MultiheadAttentionBertType(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        qkv_combined: bool = True,
        dropout:float=0,
        recompute:bool=False,
        mlp_init_std:float=0.0,
        deterministic: bool = False,
        has_bias: bool = False,
        residual_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_combined = qkv_combined
        self.dropout = dropout
        self.recompute = recompute
        self.device = device
        self.dtype = dtype
        self.mlp_init_std = mlp_init_std
        self.deterministic = deterministic
        self.has_bias = has_bias
        self.residual_scale = residual_scale
        self.kwargs = kwargs
        self.kv_num_heads = kwargs.get('kv_num_heads', num_heads)
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.attn_type = self.kwargs.get('attn_type', "sequence")
        self.layer_recompute = self.kwargs.get('layer_recompute', False)

        # Use nn.Linear for better compatibility and bias support
        if self.qkv_combined:
            self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=self.has_bias, **factory_kwargs)
        else:
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=self.has_bias, **factory_kwargs)
            self.k_proj = nn.Linear(embed_dim, self.kv_num_heads * self.head_dim, bias=self.has_bias, **factory_kwargs)
            self.v_proj = nn.Linear(embed_dim, self.kv_num_heads * self.head_dim, bias=self.has_bias, **factory_kwargs)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=self.has_bias, **factory_kwargs)
        if recompute:
            self.forward = partial(checkpoint, self.forward, use_reentrant=False)  # type: ignore


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
        qkv: torch.Tensor | None,
        q: torch.Tensor | None,
        kv: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Since flash attention does not support attn_mask,
        use scaled_dot_product_attention to compute attention when attn_mask is not None
        """
        if qkv is not None:
            q, k, v = qkv.unbind(dim=-3)
        elif kv is not None and q is not None:
            k,v = kv.unbind(dim=-3)
        else:
            raise ValueError("When qkv is None, q and kv cannot both be None at the same time")
        assert q is not None and k is not None and v is not None, "q, k, and v must not be None"

        with _sdpa_context(self.deterministic):
            attention_outputs = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attn_mask=attn_mask,
                dropout_p=self.dropout,
            )
        attention_outputs = attention_outputs.transpose(1, 2)

        return attention_outputs

    @simple_autobatch(batch_size=65535, batch_dim=0)
    def flash_attn_qkvpacked_func_wrapper(self, qkv: torch.Tensor, **kwargs):
        return flash_attn_qkvpacked_func(qkv, **kwargs)

    def compute_attention_by_flashattn(
        self,
        qkv: torch.Tensor | None,
        q: torch.Tensor | None,
        kv: torch.Tensor | None,
    ) -> torch.Tensor:
        "Compute attention using flash attention"
        assert HAVE_FLASH_ATTN, \
            "Flash attention is not supported. Please install/reinstall flash attention."

        if self.qkv_combined and qkv is not None:
            atten_out = self.flash_attn_qkvpacked_func_wrapper(
                qkv,
                dropout_p=self.dropout,
                causal=False,
                return_attn_probs=False,
                deterministic=self.deterministic,
            )
            return atten_out # type: ignore
        
        elif not self.qkv_combined and q is not None and kv is not None:
            kv_num_heads = kv.size(3)
            B,S = q.shape[:2]
            kv_shape = kv.shape
            atten_out = flash_attn_varlen_kvpacked_func( # type: ignore
                q.reshape(B * S, self.num_heads, self.head_dim),
                kv.reshape(B * kv_shape[1], 2, kv_num_heads, self.head_dim),
                self.get_cu_seqlens(B, S, q.device),
                self.get_cu_seqlens(B, kv_shape[1], kv.device),
                S,
                kv_shape[1],
                dropout_p=self.dropout,
                causal=False,
                return_attn_probs=False,
                deterministic=self.deterministic,
            )

            return atten_out.reshape(B, S, *atten_out.shape[1:]) # type: ignore


    @staticmethod
    def cal_ps(q: torch.Tensor | None, k: torch.Tensor | None) -> torch.Tensor:
        # NOTE: remove einsum usage
        logits = torch.einsum("b q h d, b k h d -> b q k h", q, k)
        logits *= torch.sqrt(torch.tensor(1.0 / q.shape[-1])).to(k.device)
        ps = torch.softmax(logits.float(), dim=2).to(torch.float16).mean(dim=-1)
        return ps


    def caculate_attention_score(self, q: torch.Tensor | None, k: torch.Tensor | None) -> torch.Tensor:
        if len(q.shape) == 3:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            cal_ps = autobatch(batch_dim=1, num_batched_tensor=1)(self.cal_ps)
        else:
            cal_ps = autobatch()(self.cal_ps)
        return cal_ps(q, k)


    def core_attention(
        self,
        qkv: torch.Tensor | None,
        q: torch.Tensor | None,
        kv: torch.Tensor | None,
        attn_mask: bool | None = None,
    ) -> torch.Tensor:
        device = qkv.device.type if self.qkv_combined else q.device.type
        dtype = qkv.dtype if self.qkv_combined else q.dtype

        if attn_mask is None and HAVE_FLASH_ATTN and device == 'cuda' and dtype != torch.float32:
            attn_out = self.compute_attention_by_flashattn(qkv, q, kv)
        else:
            attn_out = self.compute_attention_by_torch(qkv, q, kv, attn_mask)
        return attn_out
    

    @staticmethod
    @autobatch(batch_dim=-2, num_batched_tensor=1)
    def batched_matmul(x, y):
        return x @ y


    @override
    @autobatch(batch_dim=1)
    def forward(
        self,
        x: torch.Tensor,
        x_kv: Optional[torch.Tensor] = None,
        copy_first_head_kv: bool = False,
        attn_mask: torch.Tensor | None = None,
        attn_mask_test: torch.Tensor | None = None,
        calculate_sample_attention: bool = False,
        calculate_feature_attention: bool = False,
        test_use_train_first_head: bool = False,
    ) -> tuple[torch.Tensor,torch.Tensor | None, torch.Tensor | None]:
        """
        x: [batch_size, seq_len, feature, embed_dim]
        kv: Optional[batch_size, seq_len_kv, feature, embed_dim] — only needed if qkv_combined=False
        copy_first_head: Reuse the results from the first attention head
        test_use_train_first_head: Testset use Trainset first head (for IA combined attention)
        """
        # feature attention: [B S F E]
        # item attention: [B F S E]
        # B, T, C = x.shape
        B, S, _, _ = x.shape
        assert x.shape[-1] == self.embed_dim

        x = x.reshape(-1, *x.shape[-2:])
        BS, F, E = x.shape

        qkv, q, kv = None, None, None
        feature_attention, sample_attention = None, None

        with nvtx.annotate('linear-qkv'):
            if self.qkv_combined:
                qkv = self.qkv_proj(x).view(x.size(0), x.size(1), 3, self.num_heads, self.head_dim)
                q,k,v = torch.unbind(qkv, dim=2)
                qkv = torch.stack((q, k, v), dim=2)
            else:
                q = self.q_proj(x).view(x.size(0), x.size(1), self.num_heads, self.head_dim)
                
                # Project K, V (from x_kv)
                x_kv_flat = x_kv.reshape(-1, x_kv.shape[-2], E)  # [BS, F_kv, E]
                k = self.k_proj(x_kv_flat).view(x_kv_flat.size(0), x_kv_flat.size(1), self.kv_num_heads, self.head_dim)  # [BS, F_kv, H, Dh]
                v = self.v_proj(x_kv_flat).view(x_kv_flat.size(0), x_kv_flat.size(1), self.kv_num_heads, self.head_dim)

                if copy_first_head_kv:
                    k = k[:, :, :1].expand(-1, -1, self.kv_num_heads, -1)
                    v = v[:, :, :1].expand(-1, -1, self.kv_num_heads, -1)

                kv = torch.stack([k, v], dim=2)  # [BS, F_kv, 2, H, Dh]

        if test_use_train_first_head:
            # q: (BF, S, nhead, head_dim)
            # kv: (BF, P, nhead, head_dim)
            eval_pos = kv.size(1)
            q_train, q_test = q[:, :eval_pos], q[:, eval_pos:]

            # self-attn (q_train + kv_train)
            with nvtx.annotate('core-attn-1'):
                attn_out_train = self.core_attention(qkv, q_train, kv, attn_mask)
                attn_out_train = attn_out_train.view(q.size(0), -1, self.num_heads, self.head_dim)

            # cross-attn (q_test + kv_train)
            with nvtx.annotate('core-attn-2'):
                kv_with_first_head = kv[:, :, :, 0:1]
                attn_out_test = self.core_attention(
                    qkv, q_test, kv_with_first_head, attn_mask_test,
                )
                attn_out_test = attn_out_test.view(q.size(0), -1, self.num_heads, self.head_dim)

            # concat
            attn_out = torch.concat([attn_out_train, attn_out_test], dim=1)
        else:
            with nvtx.annotate('core-attn'):
                attn_out = self.core_attention(qkv, q, kv, attn_mask)
                attn_out = attn_out.reshape(BS, F, self.num_heads, self.head_dim)

        with nvtx.annotate('linear-o'):
            out = self.out_proj(attn_out.reshape(attn_out.size(0), attn_out.size(1), -1))

        if qkv is not None:
            q, k, v = qkv.unbind(dim=2)
        else:
            k, v = kv.unbind(dim=2)

        if calculate_feature_attention:
            feature_attention = self.caculate_attention_score(q, k)

        if calculate_sample_attention:
            # NOTE: override q for calculating sample attention
            if test_use_train_first_head:
                q = q_test
            sample_attention = self.caculate_attention_score(q[-1], k[-1])

        return out.reshape(B, S, *out.shape[1:]), feature_attention, sample_attention


class MultiheadAttention(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        qkv_combined: bool = True,
        dropout:float=0,
        recompute:bool=False,
        mlp_init_std:float=0.0,
        deterministic: bool = False,
        residual_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_combined = qkv_combined
        self.dropout = dropout
        self.recompute = recompute
        self.device = device
        self.dtype = dtype
        self.mlp_init_std = mlp_init_std
        self.deterministic = deterministic
        self.residual_scale = residual_scale

        self.out_proj_weight = torch.nn.Parameter(torch.zeros(self.num_heads, self.head_dim, self.embed_dim, device=self.device, dtype=self.dtype))
        self.qkv_proj_weight = torch.nn.Parameter(torch.zeros(3, self.num_heads, self.head_dim, self.embed_dim, device=device, dtype=dtype))
        self.q_proj_weight = None
        self.kv_proj_weight = None
        
        if recompute:
            self.forward = partial(checkpoint, self.forward, use_reentrant=False)  # type: ignore
    
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
        qkv: torch.Tensor | None,
        q: torch.Tensor | None,
        kv: torch.Tensor | None,
        attn_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """
        Since flash attention does not support attn_mask,
        use scaled_dot_product_attention to compute attention when attn_mask is not None
        """
        if qkv is not None:
            q, k, v = qkv.unbind(dim=-3)
        elif kv is not None and q is not None:
            k,v = kv.unbind(dim=-3)
        else:
            raise ValueError("When qkv is None, q and kv cannot both be None at the same time")
        assert q is not None and k is not None and v is not None, "q, k, and v must not be None"

        with _sdpa_context(self.deterministic):
            attention_outputs = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attn_mask=attn_mask,
                dropout_p=self.dropout,
            )
        attention_outputs = attention_outputs.transpose(1, 2)

        return attention_outputs

    @simple_autobatch(batch_size=65535, batch_dim=0)
    def flash_attn_qkvpacked_func_wrapper(self, qkv: torch.Tensor, **kwargs):
        return flash_attn_qkvpacked_func(qkv, **kwargs)

    def compute_attention_by_flashattn(
        self,
        qkv: torch.Tensor | None,
        q: torch.Tensor | None,
        kv: torch.Tensor | None
    ) -> torch.Tensor:
        "Compute attention using flash attention"
        assert HAVE_FLASH_ATTN, \
            "Flash attention is not supported. Please install/reinstall flash attention."

        if self.qkv_combined and qkv is not None:
            atten_out = self.flash_attn_qkvpacked_func_wrapper(
                qkv,
                dropout_p=self.dropout,
                softmax_scale=None,
                causal=False,
                return_attn_probs=False,
                deterministic=self.deterministic,
            )
            return atten_out # type: ignore
        
        elif not self.qkv_combined and q is not None and kv is not None:
            kv_num_heads = self.num_heads if kv.size(3) == self.num_heads else 1
            B,S = q.shape[:2]
            kv_shape = kv.shape
            atten_out = flash_attn_varlen_kvpacked_func( # type: ignore
                q.reshape(B * S, self.num_heads, self.head_dim),
                kv.reshape(B * kv_shape[1], 2, kv_num_heads, self.head_dim),
                self.get_cu_seqlens(B, S, q.device),
                self.get_cu_seqlens(B, kv_shape[1], kv.device),
                S,
                kv_shape[1],
                dropout_p=self.dropout,
                causal=False,
                return_attn_probs=False,
                deterministic=self.deterministic,
            )

            return atten_out.reshape(B, S, *atten_out.shape[1:]) # type: ignore

    
    @staticmethod
    def cal_ps(q: torch.Tensor | None, k: torch.Tensor | None) -> torch.Tensor:
        # NOTE: remove einsum usage
        logits = torch.einsum("b q h d, b k h d -> b q k h", q, k)
        logits *= torch.sqrt(torch.tensor(1.0 / q.shape[-1])).to(k.device)
        ps = torch.softmax(logits.float(), dim=2).to(torch.float16).mean(dim=-1)
        return ps

    def caculate_attention_score(self, q: torch.Tensor | None, k: torch.Tensor | None) -> torch.Tensor:
        if len(q.shape) == 3:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            cal_ps = autobatch(batch_dim=1, num_batched_tensor=1)(self.cal_ps)
        else:
            cal_ps = autobatch()(self.cal_ps)
        return cal_ps(q, k)


    def core_attention(
        self,
        qkv: torch.Tensor | None,
        q: torch.Tensor | None,
        kv: torch.Tensor | None,
        attn_mask: bool = False,
    ) -> torch.Tensor:
        device = qkv.device.type if self.qkv_combined else q.device.type
        dtype = qkv.dtype if self.qkv_combined else q.dtype

        if attn_mask is None and HAVE_FLASH_ATTN and device == 'cuda' and dtype != torch.float32:
            attn_out = self.compute_attention_by_flashattn(qkv, q, kv)
        else:
            attn_out = self.compute_attention_by_torch(qkv, q, kv, attn_mask)
        return attn_out
    
    @staticmethod
    @autobatch(batch_dim=-2, num_batched_tensor=1)
    def batched_matmul(x, y):
        return x @ y

    @override
    @autobatch(batch_dim=1)
    def forward(
        self,
        x: torch.Tensor,
        x_kv: Optional[torch.Tensor] = None,
        copy_first_head_kv: bool = False,
        attn_mask: torch.Tensor | None = None,
        attn_mask_test: torch.Tensor | None = None,
        calculate_sample_attention: bool = False,
        calculate_feature_attention: bool = False,
        test_use_train_first_head: bool = False,
    ) -> tuple[torch.Tensor,torch.Tensor | None, torch.Tensor | None]:
        """
        x: [batch_size, seq_len, feature, embed_dim]
        kv: Optional[batch_size, seq_len_kv, feature, embed_dim] — only needed if qkv_combined=False
        copy_first_head: Reuse the results from the first attention head
        test_use_train_first_head: Testset use Trainset first head (for IA combined attention)
        """
        # feature attention: [B S F E]
        # item attention: [B F S E]
        # B, T, C = x.shape
        B, S, _, _ = x.shape
        assert x.shape[-1] == self.embed_dim

        x = x.reshape(-1, *x.shape[-2:])
        # TODO: remove this, IA should be BF, S, E = x.shape
        BS, F, E = x.shape

        qkv, q, kv = None, None, None
        feature_attention, sample_attention = None, None
        # batch_size = None
        # seqlen = None

        with nvtx.annotate('linear-qkv'):
            if self.qkv_combined:
                qkv_proj = self.batched_matmul(x, self.qkv_proj_weight.view(-1, E).T)
                qkv = qkv_proj.view(x.size(0), x.size(1), 3, self.num_heads, self.head_dim)
            else:
                self.q_proj_weight = self.qkv_proj_weight[0]
                self.kv_proj_weight = self.qkv_proj_weight[1:]
                assert x_kv is not None, "kv combined attention requires kv input"
                x_kv = x_kv.reshape(-1, *x_kv.shape[-2:])
                q_proj = self.batched_matmul(x, self.q_proj_weight.view(-1, E).T)
                q = q_proj.view(x.size(0), x.size(1), self.num_heads, self.head_dim)
                if copy_first_head_kv:
                    kv_weights = self.kv_proj_weight[:,:1]
                    kv_proj = self.batched_matmul(x_kv, kv_weights.reshape(-1, E).T)
                    kv = kv_proj.view(x_kv.size(0), x_kv.size(1), 2, 1, self.head_dim)
                    expand_shape = [-1 for _ in kv.shape]
                    expand_shape[-2] = self.num_heads
                    kv = kv.expand(*expand_shape)
                else:
                    kv_proj = self.batched_matmul(x_kv, self.kv_proj_weight.view(-1, E).T)
                    kv = kv_proj.view(x_kv.size(0), x_kv.size(1), 2, self.num_heads, self.head_dim)

        if test_use_train_first_head:
            # q: (BF, S, nhead, head_dim)
            # kv: (BF, P, nhead, head_dim)
            eval_pos = kv.size(1)
            q_train, q_test = q[:, :eval_pos], q[:, eval_pos:]

            # self-attn (q_train + kv_train)
            with nvtx.annotate('core-attn-1'):
                attn_out_train = self.core_attention(qkv, q_train, kv, attn_mask)
                attn_out_train = attn_out_train.view(q.size(0), -1, self.num_heads, self.head_dim)

            # cross-attn (q_test + kv_train)
            with nvtx.annotate('core-attn-2'):
                kv_with_first_head = kv[:, :, :, 0:1]
                attn_out_test = self.core_attention(qkv, q_test, kv_with_first_head, attn_mask_test)
                attn_out_test = attn_out_test.view(q.size(0), -1, self.num_heads, self.head_dim)

            # concat
            attn_out = torch.concat([attn_out_train, attn_out_test], dim=1)
        else:
            with nvtx.annotate('core-attn'):
                attn_out = self.core_attention(qkv, q, kv, attn_mask)
                attn_out = attn_out.view(BS, F, self.num_heads, self.head_dim)

        with nvtx.annotate('linear-o'):
            out_proj = self.batched_matmul(attn_out.reshape(attn_out.size(0), attn_out.size(1), -1), self.out_proj_weight.view(-1, E))
            out = out_proj.view(attn_out.size(0), attn_out.size(1), E)

        if qkv is not None:
            q, k, v = qkv.unbind(dim=2)
        else:
            k,v=kv.unbind(dim=2)

        if calculate_feature_attention:
            feature_attention = self.caculate_attention_score(q, k)

        if calculate_sample_attention:
            # NOTE: override q for calculating sample attention
            if test_use_train_first_head:
                q = q_test
            sample_attention = self.caculate_attention_score(q[-1], k[-1])

        return out.reshape(B, S, *out.shape[1:]), feature_attention, sample_attention


class EncoderBaseLayer(nn.Module):
    "Base encoder layer of the Transformer model"
    def __init__(self, 
                 nhead: int, 
                 embed_dim: int, 
                 hid_dim:int, 
                 layer_idx:int,
                 dropout: float=0,
                 pre_norm: bool=False,
                 activation: Literal['gelu']='gelu',
                 layer_norm_eps: float=1e-5,
                 device: torch.device|None=None,
                 dtype: torch.dtype|None=None,
                 recompute_attn: bool=False,
                 mlp_init_std:float=0,
                 mlp_use_residual:bool=False,
                 layer_arch: str = 'smf',
                 deterministic: bool = False,
                 seq_attn_isolated: bool = False,
                 seq_attn_serial: bool = False,
                 self_share_all_kv_heads: bool = False,
                 cross_share_all_kv_heads: bool = True,
                 init_seed_dict: dict = {},
                 mlp_seed_mode: str = 'split',
                 residual_scale: float = 1.0,
                 **layer_kwargs:Any,
                 ):
        super().__init__()
        self.logger = get_logger('ldm', 'model')
        self.nhead = nhead
        self.embed_dim = embed_dim
        self.hid_dim = hid_dim
        self.dropout = dropout
        self.pre_norm = pre_norm
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.device = device
        self.dtype = dtype
        self.layer_arch = layer_arch
        self.head_dim = self.embed_dim // self.nhead
        self.recompute_attn = recompute_attn
        self.mlp_init_std = mlp_init_std
        self.mlp_use_residual = mlp_use_residual
        self.self_share_all_kv_heads = self_share_all_kv_heads
        self.cross_share_all_kv_heads = cross_share_all_kv_heads
        self.seq_attn_serial = seq_attn_serial
        self.seq_attn_isolated = seq_attn_isolated
        self.init_seed_dict = init_seed_dict
        self.mlp_seed_mode = mlp_seed_mode
        self.layer_idx = layer_idx
        self.mlp_idx_info = {}
        self.residual_scale = residual_scale

        self.feature_attentions = []
        self.sequence_attentions = []
        self.mlp = []            
        self.deterministic = deterministic
        self.layer_arch_dict = self.parse_arch(layer_arch)
        self.profile_stage = layer_kwargs.get('profile_stage', False)
        self.tf_mlp_layer_type = layer_kwargs.get('tf_mlp_layer_type', 'normal')
        self.tf_mlp_activation_fuction = layer_kwargs.get('tf_mlp_activation_fuction', 'gelu')
        self.tf_mlp_use_bias = layer_kwargs.get('tf_mlp_use_bias', False)
        self.tf_attention_layer_type = layer_kwargs.get('tf_attention_layer_type', False)
        self.tf_attention_use_bias = layer_kwargs.get('tf_attention_use_bias', False)
        self.tf_mlp_dropout = layer_kwargs.get('tf_mlp_dropout', 0.0)
        self.tf_norm_type = layer_kwargs.get('tf_norm_type', 'layer_norm')
        self.tf_layer_norm_use_elementwise_affine = layer_kwargs.get('tf_layer_norm_use_elementwise_affine', False)
        self.layer_kwargs = layer_kwargs

        self.feature_attn_num = len(self.layer_arch_dict['FA'])     # feature attention number
        self.mlp_num = len(self.layer_arch_dict['MLP'])             # sequence attention number
        self.seq_attn_num = len(self.layer_arch_dict['SA'])         # mlp number
        if self.seq_attn_isolated:
            self.seq_attn_num *= 2

        assert (self.feature_attn_num + self.seq_attn_num + self.mlp_num) > 0, f"One of the numbers of FA, SA, and MLP must be greater than 0! layr_arch: {layer_arch}"
        # disbale logs in profile_stage
        if not self.profile_stage:
            self.logger.debug(f"layer arch: {self.layer_arch_dict}")
        # attention+MLP
        if 0 < self.feature_attn_num:
            self.feature_attentions = nn.ModuleList([self.build_FA(i) for i in range(self.feature_attn_num)])

        if 0 < self.seq_attn_num:
            self.sequence_attentions = nn.ModuleList([self.build_SA(i) for i in range(self.seq_attn_num)])

        self.mlp = nn.ModuleList([self.build_MLP(i, mlp_info) for i, mlp_info in enumerate(self.layer_arch_dict['MLP'])])
        
        self.layer_steps = []
        self.layer_step_kinds = []
        F_idx = 0
        S_idx = 0
        M_idx = 0
        for arch in self.layer_arch_dict['arch']:
            if arch == 'F':
                self.layer_steps.append(partial(self.call_features_attention, index=F_idx))
                self.layer_step_kinds.append('F')
                F_idx += 1
            elif arch == 'S':
                self.layer_steps.append(partial(self.call_sequence_attention, index=S_idx))
                self.layer_step_kinds.append('S')
                S_idx += 1
            elif arch == 'M':
                self.layer_steps.append(self.mlp[M_idx])
                self.layer_step_kinds.append('M')
                M_idx += 1
            else:
                raise ValueError(f"unsupport layer arch: {self.layer_arch_dict['arch']}")
        
        self.layer_norms = nn.ModuleList([self.build_LN(i)for i in range(len(self.layer_steps))])


    def build_FA(self, idx:int):
        with SetRandomSeed(self.init_seed_dict.get(f"layer{self.layer_idx}_FA{idx}_seed", None)):
            if 'original' == self.tf_attention_layer_type:
                return MultiheadAttention(
                        embed_dim=self.embed_dim,
                        num_heads=self.nhead,
                        device=self.device,
                        dtype=self.dtype,
                        qkv_combined=True,
                        dropout=self.dropout,
                        recompute=self.recompute_attn,
                        mlp_init_std=self.mlp_init_std,
                        deterministic=self.deterministic,
                        residual_scale=self.residual_scale,
                )
            elif 'bert' == self.tf_attention_layer_type:
                return MultiheadAttentionBertType(
                        embed_dim=self.embed_dim,
                        num_heads=self.nhead,
                        device=self.device,
                        dtype=self.dtype,
                        qkv_combined=True,
                        dropout=self.dropout,
                        recompute=self.recompute_attn,
                        mlp_init_std=self.mlp_init_std,
                        deterministic=self.deterministic,
                        has_bias=self.tf_attention_use_bias,
                        residual_scale=self.residual_scale,
                        layer_recompute=self.layer_kwargs.get('layer_recompute', False),
                        attn_type = "feature",
                )
            else:
                raise f'unkown tf_attention_layer_type: {self.tf_attention_layer_type}'


    def build_SA(self, idx:int):
        with SetRandomSeed(self.init_seed_dict.get(f"layer{self.layer_idx}_SA{idx}_seed", None)):
            if 'original' == self.tf_attention_layer_type:
                return MultiheadAttention(
                        embed_dim=self.embed_dim,
                        num_heads=self.nhead,
                        device=self.device,
                        dtype=self.dtype,
                        qkv_combined=False,
                        dropout=self.dropout,
                        recompute=self.recompute_attn,
                        mlp_init_std=self.mlp_init_std,
                        deterministic=self.deterministic,
                        residual_scale=self.residual_scale,
                    )
            elif 'bert' == self.tf_attention_layer_type:
                return MultiheadAttentionBertType(
                        embed_dim=self.embed_dim,
                        num_heads=self.nhead,
                        device=self.device,
                        dtype=self.dtype,
                        qkv_combined=False,
                        dropout=self.dropout,
                        recompute=self.recompute_attn,
                        mlp_init_std=self.mlp_init_std,
                        deterministic=self.deterministic,
                        has_bias=self.tf_attention_use_bias,
                        residual_scale=self.residual_scale,
                        kv_num_heads=self.layer_kwargs.get('seq_att_kv_num_heads', self.nhead),
                        layer_recompute=self.layer_kwargs.get('layer_recompute', False),
                        attn_type = "sequence",
                )
            else:
                raise f'unkown tf_attention_layer_type: {self.tf_attention_layer_type}'


    def build_MLP(self, idx:int, mlp_info_dict):
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

        with SetRandomSeed(seed):
            return MLP(
                    in_features=self.embed_dim,
                    hidden_size=self.hid_dim,
                    out_features=self.embed_dim,
                    has_bias=self.tf_mlp_use_bias,
                    device=self.device,
                    dtype=self.dtype,
                    activation=self.activation,
                    depth=mlp_info_dict['depth'],
                    init_std=self.mlp_init_std,
                    residual_scale=self.residual_scale,
                    tf_mlp_layer_type=self.tf_mlp_layer_type,
                    tf_mlp_activation_fuction=self.tf_mlp_activation_fuction,
                    dropout = self.tf_mlp_dropout,
                    tf_mlp_hidden_size_ratio=self.layer_kwargs.get('tf_mlp_hidden_size_ratio', 4.0),
                    tf_mlp_hidden_size_2_even=self.layer_kwargs.get('tf_mlp_hidden_size_2_even', False),
                )


    def build_LN(self, idx:int):
        with SetRandomSeed(self.init_seed_dict.get(f"layer{self.layer_idx}_LN{idx}_seed", None)):
            if self.layer_kwargs.get('use_rmsnorm', False):
                return TritonRMSNorm(
                    hs=(self.embed_dim,),
                    eps=self.layer_norm_eps,
                    elementwise_affine=self.tf_layer_norm_use_elementwise_affine,
                    recompute=self.layer_kwargs.get('layer_recompute', False),
                )
            if 'layer_norm' != self.tf_norm_type:
                raise ValueError(f'unkown tf_norm_type: {self.tf_norm_type}')
            return LayerNormMixedPrecision(normalized_shape=self.embed_dim,
                                           eps=self.layer_norm_eps,
                                           elementwise_affine=self.tf_layer_norm_use_elementwise_affine,
                                           device=self.device,
                                           dtype=self.dtype)


    def parse_arch(self, arch_str:str):
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
                    MLP.append({'depth':depth, 'type':"MLP"})
                    arch += 'M'
                elif p.startswith("F"):
                    FA.append({"type":"FA"})
                    arch += 'F'
                elif p.startswith("S"):
                    SA.append({"type":"SA"})
                    arch += 'S'
                else:
                    raise ValueError(f"unsupport arch: {arch_str}, {p} is not support")

        model_arch['FA'] = FA
        model_arch['SA'] = SA
        model_arch['MLP'] = MLP
        model_arch['arch'] = arch

        return model_arch


    @nvtx.annotate('FA')
    def call_features_attention(
        self,
        x: torch.Tensor,
        feature_attention_mask: torch.Tensor | None,
        eval_pos: int,
        index: int = 0,
        calculate_feature_attention: bool = False,
    ):
        assert len(self.feature_attentions) > index

        return self.feature_attentions[index](
            x,
            x_kv=None,
            attn_mask=feature_attention_mask,
            calculate_feature_attention=calculate_feature_attention
        )


    @nvtx.annotate('IA')
    def call_sequence_attention(
        self,
        x: torch.Tensor,
        feature_attention_mask: torch.Tensor | None,
        eval_pos: int,
        index: int = 0,
        calculate_sample_attention: bool = False,
    ):
        assert len(self.sequence_attentions) > index
        index1 = index * 2 if self.seq_attn_isolated else index
        index2 = index1 + 1 if self.seq_attn_isolated else index1
        assert index2 < len(self.sequence_attentions), \
            f'Error: index2({index2}) >= len(self.sequence_attentions)({len(self.sequence_attentions)})'
        assert 0 <= eval_pos <= x.shape[1]

        attn_mask_test = None
        attn_mask_train = None

        if eval_pos == x.shape[1]:
            print(f"\033[30;43mWarning: eval_pos >= x.shape[1]!\033[0m")

            # self-attn (Q_all * KV_all)
            out = self.sequence_attentions[index2](
                x = x.transpose(1, 2),
                x_kv = x.transpose(1, 2),
            )[0].transpose(1, 2)

            return out, None, None

        # TODO: acceleration ONLY supports self_share_all_kv_heads=False & cross_share_all_kv_heads=True currently,
        # other configurations are not yet optimized and will fallback to the naive implementation
        if self.seq_attn_serial is False and \
            (self.self_share_all_kv_heads is False and self.cross_share_all_kv_heads is True):
            test_use_train_first_head  = not self.layer_kwargs.get('disable_test_use_train_first_head', False)
            # cross-attn (Q_all * KV_train)
            out, _, sample_attention = self.sequence_attentions[index1](
                x = x.transpose(1, 2),
                x_kv = x[:, :eval_pos].transpose(1, 2),
                calculate_sample_attention = calculate_sample_attention,
                test_use_train_first_head = test_use_train_first_head,
                attn_mask = attn_mask_train,
                attn_mask_test = attn_mask_test
            )
            out = out.transpose(1, 2)

            return out, None, sample_attention
        else:
            # TODO: support combined linear-qkv and linear-o
            # first self-attn (Q_train * KV_train)
            x_train = self.sequence_attentions[index1](
                x = x[:, :eval_pos].transpose(1, 2),
                x_kv = x[:, :eval_pos].transpose(1, 2),
                copy_first_head_kv = self.self_share_all_kv_heads,
                attn_mask = attn_mask_train
            )[0].transpose(1, 2)

            if self.seq_attn_serial:
                x[:, :eval_pos] = x_train

            # then cross-attn (Q_test * KV_train)
            x_test, _, sample_attention = self.sequence_attentions[index2](
                x = x[:, eval_pos:].transpose(1, 2),
                x_kv = x[:, :eval_pos].transpose(1, 2),
                copy_first_head_kv = self.cross_share_all_kv_heads,
                calculate_sample_attention = calculate_sample_attention,
                attn_mask = attn_mask_test
            )
            x_test = x_test.transpose(1, 2)

            x_all = torch.cat([x_train, x_test], dim=1)
            return x_all, None, sample_attention

    def _autobatch_chunk_size(self, x: torch.Tensor, dim: int) -> int:
        """Chunk FA/MLP along seq (dim=1) and SA along group (dim=2) so per-chunk activations stay much smaller than the full tensor."""
        n = int(x.shape[dim])
        forced_key = '_force_seq_chunk' if dim == 1 else '_force_group_chunk'
        forced = getattr(self, forced_key, None)
        if forced is not None:
            return max(1, min(int(forced), n))
        other = x.numel() // max(n, 1)
        per = max(other * x.element_size() * 20, 1)
        free = _cuda_free_bytes(x.device)
        if free is None:
            return n
        n_fit = int(0.25 * free / per)
        if dim == 1:
            n_fit = min(n_fit, max(1, 65535 // max(int(x.shape[0]), 1)))
        return max(1, min(n, n_fit))

    def _needs_residual_chunking(self, x: torch.Tensor) -> bool:
        for kind in getattr(self, 'layer_step_kinds', []):
            chunk_dim = 2 if kind == 'S' else 1
            n = int(x.shape[chunk_dim])
            if self._autobatch_chunk_size(x, chunk_dim) < n:
                return True
        return False

    def _run_sublayer(self, sublayer, h, feature_attention_mask, eval_pos, sublayer_kwargs):
        if isinstance(sublayer, functools.partial):
            out = sublayer(h, feature_attention_mask, eval_pos, **sublayer_kwargs)
        else:
            out = sublayer(h)
        if isinstance(out, tuple):
            return out[0]
        return out

    def _residual_sublayer_chunked(
        self,
        x: torch.Tensor,
        sublayer,
        layer_norm,
        feature_attention_mask,
        eval_pos: int,
        sublayer_kwargs: dict,
        kind: str,
    ) -> torch.Tensor:
        """Apply LN + sublayer + residual onto slices of x to avoid keeping residual, LN, and attention full tensors at once."""
        chunk_dim = 2 if kind == 'S' else 1
        chunk = self._autobatch_chunk_size(x, chunk_dim)
        n = int(x.shape[chunk_dim])
        # if chunk < n:
        #     print(
        #         f"EncoderBaseLayer[{self.layer_idx}] residual chunk dim={chunk_dim} "
        #         f"size={chunk}/{n} pre_norm={self.pre_norm}"
        #     )
        start = 0
        while start < n:
            end = min(start + chunk, n)
            sl = [slice(None)] * x.dim()
            sl[chunk_dim] = slice(start, end)
            sl = tuple(sl)
            try:
                inp = x[sl]
                if self.pre_norm:
                    h = layer_norm(inp)
                    if h.dtype != inp.dtype:
                        h = h.to(dtype=inp.dtype)
                    if not h.is_contiguous():
                        h = h.contiguous()
                    y = self._run_sublayer(sublayer, h, feature_attention_mask, eval_pos, sublayer_kwargs)
                    del h
                    inp.add_(y.to(dtype=inp.dtype))
                    del y
                else:
                    h = inp if inp.is_contiguous() else inp.contiguous()
                    y = self._run_sublayer(sublayer, h, feature_attention_mask, eval_pos, sublayer_kwargs)
                    y = inp + y.to(dtype=inp.dtype)
                    y = layer_norm(y)
                    x[sl] = y.to(dtype=x.dtype)
                    del y
                start = end
            except Exception as e:
                if not _is_retryable_cuda_error(e) or chunk == 1:
                    raise
                print(
                    f"EncoderBaseLayer[{self.layer_idx}] chunk_size={chunk} OOM, retry with half"
                )
                chunk = max(1, chunk // 2)
                gc.collect()
                if x.device.type == 'cuda':
                    torch.cuda.empty_cache()
        return x

    def forward(
        self,
        x: torch.Tensor,
        feature_attention_mask: torch.Tensor,
        eval_pos: int,
        **kwargs,
    ) -> tuple[torch.Tensor,torch.Tensor | None,torch.Tensor | None]:
        calculate_sample_attention = kwargs.get('calculate_sample_attention', False)
        calculate_feature_attention = kwargs.get('calculate_feature_attention', False)
        layer_idx = kwargs.get('layer_idx', 11)

        use_chunked = (
            AutobatchConfig.ENABLE_AUTOBATCH
            and (not self.training)
            and (not calculate_sample_attention)
            and (not calculate_feature_attention)
            and self._needs_residual_chunking(x)
        )

        feature_attention = None
        sample_attention = None
        for idx, (sublayer, layer_norm) in enumerate(zip(self.layer_steps, self.layer_norms)):
            kind = self.layer_step_kinds[idx] if idx < len(getattr(self, 'layer_step_kinds', [])) else '?'
            sublayer_kwargs = {}
            if idx == 2 and calculate_feature_attention and layer_idx == 11:
                sublayer_kwargs['calculate_feature_attention'] = True
            elif ((self.pre_norm and idx == 4) or ((not self.pre_norm) and idx == 0)) and calculate_sample_attention and layer_idx == 11:
                sublayer_kwargs['calculate_sample_attention'] = True

            if use_chunked:
                x = self._residual_sublayer_chunked(
                    x, sublayer, layer_norm, feature_attention_mask, eval_pos, sublayer_kwargs, kind
                )
                continue

            if self.pre_norm:
                residual = x
                x = layer_norm(x)
                if idx == 2 and calculate_feature_attention and layer_idx == 11:
                    x, feature_attention, _ = sublayer(x, feature_attention_mask, eval_pos,calculate_feature_attention=True)
                elif idx == 4 and calculate_sample_attention and layer_idx == 11:
                    x, _, sample_attention = sublayer(x, feature_attention_mask, eval_pos,calculate_sample_attention=True)
                else:
                    if isinstance(sublayer, functools.partial):
                        x = sublayer(x, feature_attention_mask, eval_pos)
                        if isinstance(x, tuple):
                            x = x[0]
                    else:
                        x = sublayer(x)
                        if isinstance(x, tuple):
                            x = x[0]
                x = x + residual.to(dtype=x.dtype) if self.training else x.add_(residual.to(dtype=x.dtype))
            else:
                residual = x
                if idx == 2 and calculate_feature_attention and layer_idx == 11:
                    x, feature_attention, _ = sublayer(x, feature_attention_mask, eval_pos, calculate_feature_attention=True)
                elif idx == 0 and calculate_sample_attention and layer_idx == 11:
                    x, _, sample_attention = sublayer(x, feature_attention_mask, eval_pos, calculate_sample_attention=True)
                else:
                    if isinstance(sublayer, functools.partial):
                        x = sublayer(x, feature_attention_mask, eval_pos)
                        if isinstance(x, tuple):
                            x = x[0]
                    else:
                        x = sublayer(x)
                        if isinstance(x, tuple):
                            x = x[0]

                x = x + residual.to(dtype=x.dtype) if self.training else x.add_(residual.to(dtype=x.dtype))
                
                try:
                    x=layer_norm(x)
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
    
    Can be built in two ways:
    1. Pass a ``layers`` list directly (backward compatible).
    2. Build from ``model_structure_config`` (every layer must share the same ``emsize``).
    """
    def __init__(
        self, 
        layers: list[nn.Module] | None = None,
        model_structure_config: dict | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        init_seed_dict: dict = {},
        residual_scale: float = 1.0,
        **common_kwargs,
    ):
        super().__init__()
        self.logger = get_logger('ldm', 'model')
        self.layer_recompute = common_kwargs.get('layer_recompute', False)
        self.layer_recompute_start_layer = common_kwargs.get('layer_recompute_start_layer', 0)
        
        if layers is not None:
            self.layers = nn.ModuleList(layers)
        elif model_structure_config is not None:
            self.layers = self._build_layers_from_config(
                model_structure_config,
                device=device,
                dtype=dtype,
                init_seed_dict=init_seed_dict,
                residual_scale=residual_scale,
                **common_kwargs,
            )
        else:
            raise ValueError("必须提供layers或model_structure_config参数之一")
        
        self.feature_emb_layer = common_kwargs.get('feature_emb_layer', len(self.layers)-1)
        self.reg_y_emb_layer = common_kwargs.get('reg_y_emb_layer', len(self.layers)-1)
        self.cls_y_emb_layer = common_kwargs.get('cls_y_emb_layer', len(self.layers)-1)
    
    def _build_layers_from_config(
        self,
        config: dict,
        device: torch.device | None,
        dtype: torch.dtype | None,
        init_seed_dict: dict,
        residual_scale: float,
        **common_kwargs,
    ) -> nn.ModuleList:
        """Build the layer list from the model structure config.

        Args:
            config: Model structure config dict containing a 'layers' list.
            device: Device.
            dtype: Data type.
            init_seed_dict: Initialization seed dict.
            residual_scale: Residual scaling factor.
            **common_kwargs: Shared kwargs passed to every layer.

        Returns:
            Built layer list.
        """
        layers_config = config.get('layers', [])
        if not layers_config:
            raise ValueError("model_structure_config中的'layers'列表不能为空")
        
        built_layers = []
        
        for idx, layer_cfg in enumerate(layers_config):
            layer_params = {
                'layer_idx': idx,
                'embed_dim': layer_cfg['emsize'],
                'nhead': layer_cfg['nhead'],
                'hid_dim': layer_cfg['hid_dim'],
                'layer_arch': layer_cfg['arch'],
                'pre_norm': layer_cfg.get('pre_norm', common_kwargs.get('pre_norm', False)),
                'activation': layer_cfg.get('activation', common_kwargs.get('activation', 'gelu')),
                'dropout': layer_cfg.get('dropout', common_kwargs.get('dropout', 0.0)),
                'device': device,
                'dtype': dtype,
                'init_seed_dict': init_seed_dict,
                'residual_scale': residual_scale,
            }
            
            for key, value in common_kwargs.items():
                if key not in layer_params:
                    layer_params[key] = value
            
            standard_keys = {
                'layer_idx', 'arch', 'emsize', 'nhead', 'nhid_factor', 'hid_dim',
                'activation', 'dropout', 'pre_norm'
            }
            for key, value in layer_cfg.items():
                if key not in standard_keys and key not in layer_params:
                    layer_params[key] = value
            
            layer = EncoderBaseLayer(**layer_params)
            built_layers.append(layer)
            
            if not layer_params.get('profile_stage', False):
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
            with nvtx.annotate('layer'):
                kwargs['layer_idx'] = idx
                if self.layer_recompute and idx >= self.layer_recompute_start_layer:
                    x, feature_attention, sample_attention = checkpoint(layer, x, **kwargs, use_reentrant=False)
                else:
                    x, feature_attention, sample_attention = layer(x, **kwargs)

            if idx == self.feature_emb_layer and idx < nlayers-1:
                feature_emb = x.clone()
            if idx == self.reg_y_emb_layer and idx < nlayers-1:
                reg_y_emb = x.clone()
            if idx == self.cls_y_emb_layer and idx < nlayers-1:
                cls_y_emb = x.clone()

        return x, feature_attention, sample_attention, feature_emb, reg_y_emb, cls_y_emb
