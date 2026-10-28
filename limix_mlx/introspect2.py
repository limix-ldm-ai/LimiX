"""Deeper introspection: ratios, steps order, SeparateXYFFN, DStI details."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shim  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from model.v2_0.loading import load_model
from baseline_torch import CKPT

model, config = load_model(CKPT)
model.eval()
for k in ["tf_mlp_hidden_size_ratio", "tf_mlp_y_hidden_size_ratio",
          "tf_mlp_hidden_size_2_even", "tf_mlp_use_bias", "separate_x_y_ffn",
          "layer_norm_eps", "tf_layer_norm_use_elementwise_affine",
          "num_cls_tokens", "pre_norm", "all_norm", "cls_only_start_layer",
          "feature_attention_cls_token_merge", "use_separate_attention",
          "separate_attn_kv_combined", "cross_share_all_kv_heads",
          "self_share_all_kv_heads", "seq_attn_isolated", "seq_attn_serial",
          "scale_y_sample_attention_heads", "cls_sample_attention_num_heads",
          "induce_use_qk_norm", "induce_qk_norm_eps",
          "induce_qk_norm_elementwise_affine",
          "induce_use_softmax_scaling_mlp",
          "induce_softmax_scaling_temp_upper_bound",
          "induce_softmax_scaling_temp_lower_bound",
          "induce_softmax_scaling_base_bound",
          "seq_att_use_q_norm", "seq_att_use_k_norm",
          "seq_att_q_norm_use_elementwise_affine",
          "seq_att_k_norm_use_elementwise_affine",
          "fea_att_use_q_norm", "fea_att_use_k_norm",
          "fea_att_q_norm_use_elementwise_affine",
          "fea_att_k_norm_use_elementwise_affine",
          "add_task_type", "mask_feature_filling_type",
          "enable_feature_attention_mask", "feature_attention_pipeline",
          "decoupled_attn_kv_combined", "dsti_relation_dim", "dsti_value_dim",
          "dsti_num_heads", "dsti_feature_interaction_mode",
          "dsti_y_token_aggregation", "dsti_dropout", "dsti_use_qk_norm",
          "dsti_qk_norm_eps", "dsti_qk_norm_elementwise_affine",
          "dsti_use_softmax_scaling_mlp",
          "dsti_softmax_scaling_temp_upper_bound",
          "dsti_softmax_scaling_temp_lower_bound",
          "dsti_softmax_scaling_base_bound", "dsti_attention_backend",
          "deterministic", "flash_attention_precision",
          "encoder_out_norm_use_affine", "decoder_hidden_dim",
          "feature_emb_layer", "reg_y_emb_layer", "cls_y_emb_layer",
          "use_gated_norm", "dropout", "tf_mlp_dropout"]:
    print(f"{k} = {config.get(k)!r}")

L = model.transformer_encoder.layers[0]
print("layer_steps types:", [type(s).__name__ for s in L.layer_steps])
print("n layer_norms:", len(L.layer_norms), "n out_norms:", len(L.output_layer_norms))
print("feature_attn_step_idx:", L.feature_attention_step_idx,
      "sample_attn_step_idx:", L.sample_attention_step_idx)
print("separate_x_y_ffn:", L.separate_x_y_ffn, "cls_only:", L.cls_only)
print("x_ffn hidden:", L.mlp[0].x_ffn.gate_proj.weight.shape,
      "bias:", L.mlp[0].x_ffn.gate_proj.bias is not None)
print("y_ffn hidden:", L.mlp[0].y_ffn.gate_proj.weight.shape)
print("x_attn bias:", L.sequence_attentions[0].x_attention.q_proj.bias is not None)
print("sa qk affine:", L.sequence_attentions[0].x_attention.q_norm.elementwise_affine
      if hasattr(L.sequence_attentions[0].x_attention.q_norm, "elementwise_affine") else "?")
fa = L.feature_attentions[0]
print("DStI submods:", [n for n, _ in fa.named_children()])
print("DStI projs:", [(n, tuple(m.weight.shape)) for n, m in fa.named_modules()
      if isinstance(m, torch.nn.Linear)])
print("add_task_info:", type(model.add_task_info).__name__,
      model.add_task_info.token_type_embedding.weight.shape)
print("enc_x type:", type(model.encoder_x).__name__)
print("cls_y_enc:", type(model.cls_y_encoder).__name__,
      [type(m).__name__ for m in model.cls_y_encoder])
print("reg_y_enc:", type(model.reg_y_encoder).__name__,
      [type(m).__name__ for m in model.reg_y_encoder])
