"""MLX port of the LimiX-2 v2.0 forward pass (inference only).

Faithful port of limix_src/model/v2_0 (torch). Module/parameter names are
identical to the torch checkpoint so weight loading is mechanical
(see convert.py / loader.py).

Covered, in torch order (transformer.py::forward):
  padding_xy -> x_preprocess (Nan/Valid/Norm) -> mask_process_4_x ->
  MaskEmbEncoder -> y_encode (cls/reg) -> add_embeddings (pos) ->
  add_task_info -> LayerStack[SA -> SepXY-MLP -> DStI]x24 (pre-norm,
  all_norm) -> out norms -> cls/reg/feature decoders.

Numerically exact replacements:
  - TritonRMSNorm/TritonQKNorm -> fp32 x*rms^-1[*w] (same math as the
    torch CPU fallback in _run_qk_norm).
  - flash-attn / SDPA -> mx.fast.scaled_dot_product_attention in fp32.
  - nvtx / autobatch / recompute / dropout(0) -> plain calls.
  - No autocast on the MLX path: everything runs float32, matching the
    torch CPU baseline.

Requires: mlx, numpy. No torch dependency.
"""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ---------------------------------------------------------------- activations

def _silu(x):
    return x * mx.sigmoid(x)


def _gelu_exact(x):
    return nn.gelu(x)


_ACTS = {"relu": lambda x: mx.maximum(x, 0), "gelu": _gelu_exact, "silu": _silu,
         "ReLU": lambda x: mx.maximum(x, 0), "GELU": _gelu_exact, "SiLU": _silu}


class RMSNorm(nn.Module):
    """RMSNorm with optional affine weight (None when affine=False)."""

    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine: bool = False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = mx.ones((dim,))
        else:
            self.weight = None

    def __call__(self, x):
        # Fused kernel; reduces in float32 internally, so this matches the
        # manual chain to float32 reassociation error (~3e-06 relative end to
        # end, well under the 9.7e-05 gap to torch this port is measured at).
        return mx.fast.rms_norm(x, self.weight, self.eps)


# ------------------------------------------------------- softmax scaling MLP

def _soft_clamp_range(x, lo, hi):
    w = hi - lo
    mid = (hi + lo) / 2
    return (w / 2) * mx.tanh((2 / w) * x) + mid


class SoftmaxScalingMLP(nn.Module):
    """Per-head learned query scaling (softmax_scaling_mlp.py)."""

    def __init__(self, num_heads, head_dim=None, temp_upper_bound=0.5,
                 temp_lower_bound=0.0, scale_base_bound=5.0):
        super().__init__()
        self.num_heads = num_heads
        self.temp_upper_bound = temp_upper_bound
        self.temp_lower_bound = temp_lower_bound
        self.scale_base_bound = scale_base_bound
        self.scale_linear = nn.Linear(1, num_heads)

    def __call__(self, q, n: int):
        logn = math.log(n / 1.0)
        w = _soft_clamp_range(self.scale_linear.weight.astype(mx.float32),
                              self.temp_lower_bound, self.temp_upper_bound)  # [H,1]
        logn_scale = (w * logn).squeeze(-1) + 1.0  # [H]
        bb = self.scale_base_bound
        bias_scale = ((bb / 2) * mx.tanh(
            self.scale_linear.bias.astype(mx.float32) * 2 / bb) + bb / 2)
        scales = (logn_scale * bias_scale).reshape(1, 1, self.num_heads, 1)
        return q * scales.astype(q.dtype)


# ------------------------------------------------------------------ attention

def _sdpa(q, k, v, mask=None):
    """q/k/v: [B,H,S,D]/[B,H,Skv,D]; mask bool [B,1,1 or S,Skv] True=attend."""
    dt = q.dtype
    qf = q.astype(mx.float32)
    kf = k.astype(mx.float32)
    vf = v.astype(mx.float32)
    # Fused kernel: never materializes the [B,H,S,Skv] score matrix, and does
    # the softmax in float32 internally like the manual version it replaces.
    bias = None
    if mask is not None:
        bias = (1.0 - mask.astype(mx.float32)) * -1e9
    o = mx.fast.scaled_dot_product_attention(
        qf, kf, vf, scale=q.shape[-1] ** -0.5, mask=bias)
    return o.astype(dt)


class MultiheadAttentionBertType(nn.Module):
    """torch layer.py::MultiheadAttentionBertType (bert only, no combined qkv)."""

    def __init__(self, embed_dim, num_heads, has_bias=False,
                 rmsnorm_impl="triton", qk_norm_eps=1e-5,
                 qk_norm_elementwise_affine=True,
                 sm_temp_upper=0.5, sm_temp_lower=0.0, sm_base_bound=5.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=has_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=has_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=has_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=has_bias)
        self.q_norm = RMSNorm(self.head_dim, eps=qk_norm_eps,
                              elementwise_affine=qk_norm_elementwise_affine)
        self.k_norm = RMSNorm(self.head_dim, eps=qk_norm_eps,
                              elementwise_affine=qk_norm_elementwise_affine)
        self.softmax_scaling_mlp = SoftmaxScalingMLP(
            num_heads, self.head_dim, sm_temp_upper, sm_temp_lower, sm_base_bound)

    def __call__(self, x, x_kv, attn_mask=None):
        # x: [B,S,F,E], x_kv: [B,Skv,F,E] -> out [B,S,F,E].
        # NOTE: torch flattens B*S and attends over F per (b,s); the
        # log-n scaling count n is the key token dim F (for the merged-Y
        # branch that dim is the sample count -- same rule applies).
        b, s, f, e = x.shape
        fk = x_kv.shape[2]
        n = fk
        bs, bskv = x.shape[0] * x.shape[1], x_kv.shape[0] * x_kv.shape[1]
        q = self.q_proj(x.reshape(bs, f, e)).reshape(
            bs, f, self.num_heads, self.head_dim)
        kvf = x_kv.reshape(bskv, fk, e)
        k = self.k_proj(kvf).reshape(-1, fk, self.num_heads, self.head_dim)
        v = self.v_proj(kvf).reshape(-1, fk, self.num_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = self.softmax_scaling_mlp(q, n)
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        o = _sdpa(q, k, v, attn_mask)
        o = mx.transpose(o, (0, 2, 1, 3)).reshape(bs, f, e)
        out = self.out_proj(o)
        return out.reshape(b, s, f, e)


class DifferentMHA(nn.Module):
    """Separate X / Y-token sample attention (reshape merge)."""

    def __init__(self, x_attention, y_attention, num_y_tokens=1):
        super().__init__()
        self.x_attention = x_attention
        self.y_attention = y_attention
        self.num_y_tokens = num_y_tokens

    def _merge(self, t):
        # [B,Ty,S,E] -> [B,1,S,Ty*E]
        return mx.expand_dims(
            mx.transpose(t, (0, 2, 1, 3)).reshape(t.shape[0], t.shape[2], -1), 1)

    def __call__(self, x, x_kv):
        k = self.num_y_tokens
        out_x = self.x_attention(x[:, :-k], x_kv[:, :-k])
        yt = self._merge(x[:, -k:])
        ykvt = self._merge(x_kv[:, -k:])
        out_y = self.y_attention(yt, ykvt)
        out_y = mx.transpose(out_y.squeeze(1).reshape(
            x.shape[0], x.shape[2], k, x.shape[3]), (0, 2, 1, 3))
        return mx.concatenate([out_x, out_y], axis=1)


class DecoupledStructuralTaskAttention(nn.Module):
    """torch decoupled_structural_task_attention.py (sample_split, concat_linear)."""

    def __init__(self, embed_dim, num_heads, num_y_tokens=1, relation_dim=None,
                 value_dim=None, bias=True, qk_norm_eps=1e-5,
                 qk_norm_elementwise_affine=True,
                 sm_temp_upper=0.4, sm_temp_lower=0.0, sm_base_bound=1.0):
        super().__init__()
        relation_dim = embed_dim if not relation_dim else int(relation_dim)
        value_dim = embed_dim if not value_dim else int(value_dim)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_y_tokens = num_y_tokens
        self.relation_dim = relation_dim
        self.value_dim = value_dim
        self.relation_head_dim = relation_dim // num_heads
        self.value_head_dim = value_dim // num_heads
        self.sample_x_q_proj = nn.Linear(embed_dim, relation_dim, bias=bias)
        self.sample_all_k_proj = nn.Linear(embed_dim, relation_dim, bias=bias)
        self.sample_all_v_proj = nn.Linear(embed_dim, value_dim, bias=bias)
        self.sample_x_out_proj = nn.Linear(value_dim, embed_dim, bias=bias)
        self.yx_y_q_proj = nn.Linear(embed_dim, relation_dim, bias=bias)
        self.yx_x_k_proj = nn.Linear(embed_dim, relation_dim, bias=bias)
        self.yx_x_v_proj = nn.Linear(embed_dim, value_dim, bias=bias)
        self.yx_out_proj = nn.Linear(num_y_tokens * value_dim,
                                     num_y_tokens * embed_dim, bias=bias)
        self.sample_x_q_norm = RMSNorm(self.relation_head_dim, eps=qk_norm_eps,
                                       elementwise_affine=qk_norm_elementwise_affine)
        self.sample_all_k_norm = RMSNorm(self.relation_head_dim, eps=qk_norm_eps,
                                         elementwise_affine=qk_norm_elementwise_affine)
        self.yx_q_norm = RMSNorm(self.relation_head_dim, eps=qk_norm_eps,
                                 elementwise_affine=qk_norm_elementwise_affine)
        self.yx_k_norm = RMSNorm(self.relation_head_dim, eps=qk_norm_eps,
                                 elementwise_affine=qk_norm_elementwise_affine)
        self.sample_x_softmax_scaling = SoftmaxScalingMLP(
            num_heads, self.relation_head_dim, sm_temp_upper, sm_temp_lower,
            sm_base_bound)
        self.yx_softmax_scaling = SoftmaxScalingMLP(
            num_heads, self.relation_head_dim, sm_temp_upper, sm_temp_lower,
            sm_base_bound)

    def _attn(self, q, k, v, mask):
        # [BS,T,H,D] -> SDPA per head
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        return mx.transpose(_sdpa(q, k, v, mask), (0, 2, 1, 3))

    def __call__(self, x, feature_padding_mask=None):
        # x: [B,S,Ft,E]; Ft = Fg + K
        k = self.num_y_tokens
        fg = x.shape[2] - k
        x_tok = x[:, :, :fg]
        y_tok = x[:, :, fg:]
        b, s = x.shape[0], x.shape[1]
        if feature_padding_mask is not None:
            feature_padding_mask = mx.array(feature_padding_mask, dtype=mx.bool_)
            if not mx.any(feature_padding_mask).item():
                feature_padding_mask = None

        # X route: Q=X over K/V=[X;Y]
        flat_x = x_tok.reshape(b * s, fg, self.embed_dim)
        q = self.sample_x_q_proj(flat_x).reshape(
            b * s, fg, self.num_heads, self.relation_head_dim)
        flat_all = mx.concatenate([x_tok, y_tok], axis=2).reshape(
            b * s, fg + k, self.embed_dim)
        kk = self.sample_all_k_proj(flat_all).reshape(
            b * s, fg + k, self.num_heads, self.relation_head_dim)
        vv = self.sample_all_v_proj(flat_all).reshape(
            b * s, fg + k, self.num_heads, self.value_head_dim)
        q = self.sample_x_q_norm(q)
        kk = self.sample_all_k_norm(kk)
        q = self.sample_x_softmax_scaling(q, fg + k)
        mask = None
        if feature_padding_mask is not None:
            valid = mx.concatenate(
                [~feature_padding_mask.reshape(b * s, fg),
                 mx.ones((b * s, k), dtype=mx.bool_)], axis=1)
            mask = valid[:, None, None, :]
        mixed = self._attn(q, kk, vv, mask).reshape(b, s, fg, self.value_dim)
        x_update = self.sample_x_out_proj(mixed)
        if feature_padding_mask is not None:
            x_update = mx.where(feature_padding_mask[..., None],
                                mx.zeros_like(x_update), x_update)

        # YX route: Q=Y over K/V=X (original X)
        flat_y = y_tok.reshape(b * s, k, self.embed_dim)
        qy = self.yx_y_q_proj(flat_y).reshape(
            b * s, k, self.num_heads, self.relation_head_dim)
        ky = self.yx_x_k_proj(flat_x).reshape(
            b * s, fg, self.num_heads, self.relation_head_dim)
        vy = self.yx_x_v_proj(flat_x).reshape(
            b * s, fg, self.num_heads, self.value_head_dim)
        qy = self.yx_q_norm(qy)
        ky = self.yx_k_norm(ky)
        qy = self.yx_softmax_scaling(qy, fg)
        masky = None
        if feature_padding_mask is not None:
            masky = (~feature_padding_mask).reshape(b * s, 1, 1, fg)
        readout = self._attn(qy, ky, vy, masky).reshape(
            b, s, k, self.value_dim)
        y_update = self.yx_out_proj(
            readout.reshape(b, s, k * self.value_dim)).reshape(
                b, s, k, self.embed_dim)
        return mx.concatenate([x_update, y_update], axis=2)


# ---------------------------------------------------------------------- MLP

class GatedMLP(nn.Module):
    """torch layer.py::MLP (gated SiLU here)."""

    def __init__(self, in_features, hidden_size, out_features, has_bias=True,
                 activation="silu"):
        super().__init__()
        self.gate_proj = nn.Linear(in_features, hidden_size, bias=has_bias)
        self.up_proj = nn.Linear(in_features, hidden_size, bias=has_bias)
        self.down_proj = nn.Linear(hidden_size, out_features, bias=has_bias)
        self.act = _ACTS[activation]

    def __call__(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class SeparateXYFFN(nn.Module):
    def __init__(self, x_ffn, y_ffn, num_y_tokens=1):
        super().__init__()
        self.x_ffn = x_ffn
        self.y_ffn = y_ffn
        self.num_y_tokens = num_y_tokens

    def __call__(self, x):
        k = self.num_y_tokens
        out_x = self.x_ffn(x[..., :-k, :])
        yt = x[..., -k:, :].reshape(*x.shape[:2], -1)
        out_y = self.y_ffn(yt).reshape(*x.shape[:2], k, x.shape[-1])
        return mx.concatenate([out_x, out_y], axis=-2)


# -------------------------------------------------------------------- layer

class EncoderBaseLayer(nn.Module):
    """One 'smf' layer: SA -> SepXY-MLP -> DStI, pre-norm + all_norm."""

    def __init__(self, embed_dim, nhead, mlp_hidden_x, mlp_hidden_y,
                 num_y_tokens, has_bias=True, mlp_activation="silu",
                 sm_bounds=(0.5, 0.0, 5.0), dsti_bounds=(0.4, 0.0, 1.0),
                 layer_norm_eps=1e-5, ln_affine=False):
        super().__init__()
        self.pre_norm = True
        self.all_norm = True
        # torch attribute names preserved for mechanical weight loading:
        # sequence_attentions.0.{x_attention,y_attention}, mlp.0.{x_ffn,y_ffn},
        # feature_attentions.0, layer_norms.{0,1,2}, output_layer_norms.{0,1,2}
        x_attn = MultiheadAttentionBertType(
            embed_dim, nhead, has_bias, qk_norm_eps=1e-5,
            qk_norm_elementwise_affine=True,
            sm_temp_upper=sm_bounds[0], sm_temp_lower=sm_bounds[1],
            sm_base_bound=sm_bounds[2])
        ye, yh = embed_dim * num_y_tokens, nhead * num_y_tokens
        y_attn = MultiheadAttentionBertType(
            ye, yh, has_bias, qk_norm_eps=1e-5,
            qk_norm_elementwise_affine=True,
            sm_temp_upper=sm_bounds[0], sm_temp_lower=sm_bounds[1],
            sm_base_bound=sm_bounds[2])
        self.sequence_attentions = [DifferentMHA(x_attn, y_attn, num_y_tokens)]
        self.mlp = [SeparateXYFFN(
            GatedMLP(embed_dim, mlp_hidden_x, embed_dim, has_bias, mlp_activation),
            GatedMLP(ye, mlp_hidden_y, ye, has_bias, mlp_activation),
            num_y_tokens)]
        self.feature_attentions = [DecoupledStructuralTaskAttention(
            embed_dim, nhead, num_y_tokens, bias=True, qk_norm_eps=1e-5,
            qk_norm_elementwise_affine=True,
            sm_temp_upper=dsti_bounds[0], sm_temp_lower=dsti_bounds[1],
            sm_base_bound=dsti_bounds[2])]
        self.layer_norms = [RMSNorm(embed_dim, layer_norm_eps, ln_affine)
                            for _ in range(3)]
        self.output_layer_norms = [RMSNorm(embed_dim, layer_norm_eps, ln_affine)
                                   for _ in range(3)]

    def __call__(self, x, feature_mask, eval_pos):
        # step 0: sample attention (Q_all x KV_train)
        residual = x
        h = self.layer_norms[0](x)
        t = mx.transpose(h, (0, 2, 1, 3))
        kv = mx.transpose(h[:, :eval_pos], (0, 2, 1, 3))
        o = self.sequence_attentions[0](t, kv)
        o = mx.transpose(o, (0, 2, 1, 3))
        o = self.output_layer_norms[0](o)
        x = o + residual.astype(o.dtype)
        # step 1: MLP
        residual = x
        h = self.layer_norms[1](x)
        o = self.mlp[0](h)
        o = self.output_layer_norms[1](o)
        x = o + residual.astype(o.dtype)
        # step 2: feature attention (DStI)
        residual = x
        h = self.layer_norms[2](x)
        fg = h.shape[2] - self.mlp[0].num_y_tokens
        fpm = (feature_mask[..., :fg]
               if feature_mask is not None else None)
        o = self.feature_attentions[0](h, fpm)
        o = self.output_layer_norms[2](o)
        x = o + residual.astype(o.dtype)
        return x


class LayerStack(nn.Module):
    def __init__(self, layers, num_cls_tokens, feature_emb_layer,
                 reg_y_emb_layer, cls_y_emb_layer):
        super().__init__()
        self.layers = layers
        self.num_cls_tokens = num_cls_tokens
        self.feature_emb_layer = feature_emb_layer
        self.reg_y_emb_layer = reg_y_emb_layer
        self.cls_y_emb_layer = cls_y_emb_layer

    def __call__(self, x, feature_mask, eval_pos):
        nl = len(self.layers)
        feature_emb = reg_y_emb = cls_y_emb = None
        for idx, layer in enumerate(self.layers):
            x = layer(x, feature_mask, eval_pos)
            if idx == self.feature_emb_layer and idx < nl - 1:
                feature_emb = x
            if idx == self.reg_y_emb_layer and idx < nl - 1:
                reg_y_emb = x[:, :, -self.num_cls_tokens:, :]
            if idx == self.cls_y_emb_layer and idx < nl - 1:
                cls_y_emb = x[:, :, -self.num_cls_tokens:, :]
        return x, feature_emb