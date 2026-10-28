"""Introspect live LimiX-2 model structure."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shim import stub_cuda_only  # noqa: F401  (installs nvtx/triton stubs)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from model.v2_0.loading import load_model
from baseline_torch import CKPT

model, config = load_model(CKPT)
model.eval()
print("feature_attention_pipeline:", config.get("feature_attention_pipeline"))
print("layer_arch:", config.get("layer_arch"))
msc = config.get("model_structure_config")
if msc:
    print("msc layers[0]:", msc["layers"][0])
    print("msc keys:", list(msc.keys()))
L = model.transformer_encoder.layers[0]
print("layer children:", [n for n, _ in L.named_children()])
for n, m in L.named_children():
    print(" ", n, type(m).__name__, [c for c, _ in m.named_children()])
print("x_ffn act:", type(L.mlp[0].x_ffn.act_fn).__name__)
print("y_ffn act:", type(L.mlp[0].y_ffn.act_fn).__name__)
xa = L.sequence_attentions[0].x_attention
ya = L.sequence_attentions[0].y_attention
print("x_attn E/H/D:", xa.embed_dim, xa.num_heads, xa.head_dim if hasattr(xa, "head_dim") else "?")
print("y_attn E/H/D:", ya.embed_dim, ya.num_heads, ya.head_dim if hasattr(ya, "head_dim") else "?")
print("cls_y_decoder pattern:", config.get("cls_y_decoder_mlp_pattern"))
print("reg_y_decoder pattern:", config.get("reg_y_decoder_mlp_pattern"))
print("reg borders:", config.get("border_range"), "learnable:", config.get("learnable_borders"))
fa = L.feature_attentions[0]
print("DStI:", {k: v for k, v in vars(fa).items() if isinstance(v, (int, float, str, bool))})
print("encoder_out_norm affine:", model.encoder_out_norm.elementwise_affine
      if hasattr(model.encoder_out_norm, "elementwise_affine") else "Identity")
print("add_task_info:", hasattr(model, "add_task_info"))
print("use_feature_post_adapter:", config.get("use_feature_post_adapter"),
      config.get("feature_post_adapter_pattern"))
print("use_cls_post_adapter:", config.get("use_cls_post_adapter"),
      config.get("cls_post_adapter_pattern"))
print("use_reg_post_adapter:", config.get("use_reg_post_adapter"),
      config.get("reg_post_adapter_pattern"))
