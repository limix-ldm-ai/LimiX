import torch
from .transformer import FeaturesTransformer


def build_model(config: dict):
    return FeaturesTransformer(
        preprocess_config_x=config['preprocess_config_x'],
        encoder_config_x=config['encoder_config_x'],
        encoder_config_y=config['encoder_config_y'],
        decoder_config=config['decoder_config'],
        feature_positional_embedding_type=config.get(
            'feature_positional_embedding_type', "subspace"
        ),
        nlayers=config['nlayers'],
        nhead=config['nhead'],
        embed_dim=config['embed_dim'],
        hid_dim=config['hid_dim'],
        tf_mlp_layer_type=config['tf_mlp_layer_type'],
        tf_mlp_activation_fuction=config['tf_mlp_activation_fuction'],
        use_gated_norm=config.get('use_gated_norm', False),
        tf_mlp_use_bias=config['tf_mlp_use_bias'],
        tf_attention_layer_type=config['tf_attention_layer_type'],
        tf_attention_use_bias=config['tf_attention_use_bias'],
        tf_norm_type=config['tf_norm_type'],
        rmsnorm_impl=config.get('rmsnorm_impl', 'triton'),
        tf_layer_norm_use_elementwise_affine=config['tf_layer_norm_use_elementwise_affine'],
        features_per_group=config['features_per_group'],
        dropout=config.get('dropout', 0.0),
        tf_mlp_dropout=config.get('tf_mlp_dropout', 0.0),
        decoder_dropout=config.get('decoder_dropout', 0.0),
        pre_norm=config.get('pre_norm', False),
        all_norm=config.get('all_norm', False),
        activation=config.get('activation', 'gelu'),
        recompute_attn=config.get('recompute_attn', False),
        layer_recompute=config.get('layer_recompute', False),
        layer_recompute_start_layer=config.get('layer_recompute_start_layer', 0),
        self_share_all_kv_heads=config.get('self_share_all_kv_heads', False),
        cross_share_all_kv_heads=config.get('cross_share_all_kv_heads', True),
        seq_attn_isolated=config.get('seq_attn_isolated', False),
        seq_attn_serial=config.get('seq_attn_serial', False),
        num_cls_tokens=config.get('num_cls_tokens', 1),
        cls_only_start_layer=config.get('cls_only_start_layer', -1),
        use_separate_attention=config.get('use_separate_attention', False),
        separate_attn_kv_combined=config.get('separate_attn_kv_combined', False),
        sample_attention_cls_token_merge=config.get(
            'sample_attention_cls_token_merge', 'reshape'
        ),
        scale_y_sample_attention_heads=config.get(
            'scale_y_sample_attention_heads', False
        ),
        cls_sample_attention_num_heads=config.get('cls_sample_attention_num_heads', 0),
        cls_only_sample_attention_num_heads=config.get(
            'cls_only_sample_attention_num_heads', 0
        ),
        feature_attention_cls_token_merge=config.get(
            'feature_attention_cls_token_merge', 'none'
        ),
        feature_attention_pipeline=config.get('feature_attention_pipeline', 'legacy'),
        decoupled_attn_kv_combined=config.get('decoupled_attn_kv_combined', False),
        dsti_relation_dim=config.get('dsti_relation_dim', 0),
        dsti_value_dim=config.get('dsti_value_dim', 0),
        dsti_num_heads=config.get('dsti_num_heads', 0),
        dsti_feature_interaction_mode=config.get(
            'dsti_feature_interaction_mode', 'sample_split'
        ),
        dsti_y_token_aggregation=config.get('dsti_y_token_aggregation', 'none'),
        dsti_dropout=config.get('dsti_dropout', -1.0),
        dsti_use_qk_norm=config.get('dsti_use_qk_norm', False),
        dsti_qk_norm_eps=config.get('dsti_qk_norm_eps', 1e-5),
        dsti_qk_norm_elementwise_affine=config.get(
            'dsti_qk_norm_elementwise_affine', True
        ),
        dsti_use_softmax_scaling_mlp=config.get('dsti_use_softmax_scaling_mlp', False),
        dsti_softmax_scaling_temp_upper_bound=config.get(
            'dsti_softmax_scaling_temp_upper_bound', 0.4
        ),
        dsti_softmax_scaling_temp_lower_bound=config.get(
            'dsti_softmax_scaling_temp_lower_bound', 0.0
        ),
        dsti_softmax_scaling_base_bound=config.get(
            'dsti_softmax_scaling_base_bound', 1.0
        ),
        dsti_attention_backend=config.get('dsti_attention_backend', 'auto'),
        sa_temp_upper_bound=config.get('sa_temp_upper_bound', 0.5),
        sa_temp_lower_bound=config.get('sa_temp_lower_bound', 0.0),
        sa_scale_base_bound=config.get('sa_scale_base_bound', 5.0),
        enable_mask_feature_pred=config.get('enable_mask_feature_pred', True),
        enable_mask_indicator_pred=config.get('enable_mask_indicator_pred', False),
        mask_feature_filling_type=config.get('mask_feature_filling_type', 'mask_token'),
        enable_classification_y_pred=config.get('enable_classification_y_pred', True),
        enable_regression_y_pred=config.get('enable_regression_y_pred', True),
        num_buckets=config.get('num_buckets', 0),
        deterministic=config.get('deterministic', False),
        flash_attention_precision=config.get('flash_attention_precision', 'inherit'),
        init_seed_dict=config.get('init_seed_dict', {}),
        mlp_seed_mode=config.get('init_seed_mlp_mode', 'split'),
        use_feature_post_adapter=config.get('use_feature_post_adapter', False),
        feature_post_adapter_pattern=config.get(
            'feature_post_adapter_pattern', 'linear-192'
        ),
        use_feature_post_adapter_residual=config.get(
            'use_feature_post_adapter_residual', False
        ),
        use_cls_post_adapter=config.get('use_cls_post_adapter', False),
        cls_post_adapter_pattern=config.get('cls_post_adapter_pattern', 'linear-192'),
        use_cls_post_adapter_residual=config.get('use_cls_post_adapter_residual', False),
        use_reg_post_adapter=config.get('use_reg_post_adapter', False),
        reg_post_adapter_pattern=config.get('reg_post_adapter_pattern', 'linear-192'),
        use_reg_post_adapter_residual=config.get('use_reg_post_adapter_residual', False),
        decoder_hidden_dim=config.get('decoder_hidden_dim', 192 * 4),
        cls_y_decoder_type=config.get('cls_y_decoder_type', 'original'),
        cls_y_decoder_mlp_pattern=config.get(
            'cls_y_decoder_mlp_pattern', 'linear-768_GELU_linear-10'
        ),
        reg_y_decoder_type=config.get('reg_y_decoder_type', 'original'),
        reg_y_decoder_mlp_pattern=config.get(
            'reg_y_decoder_mlp_pattern', 'linear-768_GELU_linear-10'
        ),
        feature_mse_decoder_type=config.get('feature_mse_decoder_type', 'original'),
        model_structure_config=config.get('model_structure_config', None),
        separate_x_y_ffn=config.get('separate_x_y_ffn', False),
        feature_emb_layer=config.get('feature_emb_layer', config['nlayers'] - 1),
        reg_y_emb_layer=config.get('reg_y_emb_layer', config['nlayers'] - 1),
        cls_y_emb_layer=config.get('cls_y_emb_layer', config['nlayers'] - 1),
        encoder_out_norm_use_affine=config.get('encoder_out_norm_use_affine', False),
        add_task_type=config.get('add_task_type', False),
        tf_mlp_hidden_size_ratio=config.get('tf_mlp_hidden_size_ratio', 4.0),
        tf_mlp_hidden_size_2_even=config.get('tf_mlp_hidden_size_2_even', False),
        tf_mlp_y_hidden_size_ratio=config.get('tf_mlp_y_hidden_size_ratio', None),
        induce_use_qk_norm=config.get(
            'induce_use_qk_norm', config.get('seq_att_use_q_norm', False)
        ),
        induce_qk_norm_eps=config.get('induce_qk_norm_eps', 1e-5),
        induce_qk_norm_elementwise_affine=config.get(
            'induce_qk_norm_elementwise_affine', True
        ),
        induce_use_softmax_scaling_mlp=config.get(
            'induce_use_softmax_scaling_mlp',
            config.get('seq_att_use_softmax_scaling_mlp_for_q', False),
        ),
        induce_softmax_scaling_temp_upper_bound=config.get(
            'induce_softmax_scaling_temp_upper_bound', 0.5
        ),
        induce_softmax_scaling_temp_lower_bound=config.get(
            'induce_softmax_scaling_temp_lower_bound', 0.0
        ),
        induce_softmax_scaling_base_bound=config.get(
            'induce_softmax_scaling_base_bound', 5.0
        ),
    )


def load_from_checkpoint(state_dict, mask_prediction: bool = False, deterministic: bool = False):
    config = state_dict['config']
    if 'encoder_config_x' not in config:
        raise KeyError("checkpoint config missing encoder_config_x")
    config['mask_prediction'] = mask_prediction
    config['deterministic'] = deterministic
    model = build_model(config)
    model.load_state_dict(state_dict['state_dict'])
    model.eval()
    return model, config


def load_model(model_path, mask_prediction: bool = False, deterministic: bool = False):
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    return load_from_checkpoint(
        state_dict, mask_prediction=mask_prediction, deterministic=deterministic
    )
