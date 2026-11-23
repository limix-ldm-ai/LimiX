import torch
from .transformer import FeaturesTransformer


def build_model(config: dict):
    return FeaturesTransformer(
        preprocess_config_x=config['preprocess_config_x'],
        encoder_config_x=config['encoder_config_x'],
        encoder_config_y=config['encoder_config_y'],
        decoder_config=config['decoder_config'],
        feature_positional_embedding_type=config.get('feature_positional_embedding_type', "subortho"),
        nlayers=config['nlayers'],
        nhead=config['nhead'],
        embed_dim=config['embed_dim'],
        hid_dim=config['hid_dim'],
        mask_feature_embedding_type=config.get('mask_feature_embedding_type', 'mask_embedding'),
        enable_mask_feature_pred=config.get('enable_mask_feature_pred', True),
        features_per_group=config['features_per_group'],
        pre_norm=config.get('pre_norm', True),
        device=config.get('device', None),
        dtype=config.get('dtype', None),
        init_seed_dict=config.get('init_seed_dict', {}),
        tf_mlp_layer_type=config['tf_mlp_layer_type'],
        tf_mlp_activation_fuction=config['tf_mlp_activation_fuction'],
        tf_mlp_use_bias=config['tf_mlp_use_bias'],
        tf_attention_layer_type=config['tf_attention_layer_type'],
        tf_attention_use_bias=config['tf_attention_use_bias'],
        tf_norm_type=config.get('tf_norm_type', 'layer_norm'),
        tf_layer_norm_use_elementwise_affine=config.get('tf_layer_norm_use_elementwise_affine', False),
        dropout=config.get('dropout', 0.0),
        tf_mlp_dropout=config.get('tf_mlp_dropout', 0.0),
        decoder_dropout=config.get('decoder_dropout', 0.0),
        activation=config.get('activation', 'gelu'),
        recompute_attn=config.get('recompute_attn', False),
        layer_recompute=config.get('layer_recompute', False),
        layer_recompute_start_layer=config.get('layer_recompute_start_layer', 0),
        layer_arch=config.get('layer_arch', 'fmfmsm'),
        self_share_all_kv_heads=config.get('self_share_all_kv_heads', False),
        cross_share_all_kv_heads=config.get('cross_share_all_kv_heads', True),
        seq_attn_isolated=config.get('seq_attn_isolated', False),
        seq_attn_serial=config.get('seq_attn_serial', False),
        enable_classification_y_pred=config.get('enable_classification_y_pred', True),
        enable_regression_y_pred=config.get('enable_regression_y_pred', True),
        num_buckets=config.get('num_buckets', 0),
        use_rmsnorm=config.get('use_rmsnorm', False),
        deterministic=config.get('deterministic', False),
        profile_stage=config.get('profile_stage', False),
        mlp_seed_mode=config.get('init_seed_mlp_mode', 'split'),
        decoder_hidden_dim=config.get('decoder_hidden_dim', 768),
        cls_y_decoder_type=config.get('cls_y_decoder_type', 'original'),
        cls_y_decoder_mlp_pattern=config.get('cls_y_decoder_mlp_pattern', 'linear-768_GELU_linear-10'),
        reg_y_mse_decoder_type=config.get('reg_y_mse_decoder_type', 'original'),
        feature_mse_decoder_type=config.get('feature_mse_decoder_type', 'original'),
        use_shared_encoder=config.get('use_shared_encoder', False),
        model_structure_config=config.get('model_structure_config', None),
        disable_test_use_train_first_head=config.get('disable_test_use_train_first_head', False),
        feature_emb_layer=config.get('feature_emb_layer', config['nlayers'] - 1),
        reg_y_emb_layer=config.get('reg_y_emb_layer', config['nlayers'] - 1),
        cls_y_emb_layer=config.get('cls_y_emb_layer', config['nlayers'] - 1),
        seq_att_kv_num_heads=config.get('seq_att_kv_num_heads', config['nhead']),
        encoder_out_norm_use_affine=config.get('encoder_out_norm_use_affine', False),
        num_target_tokens=config.get('num_target_tokens', 0),
        tf_mlp_hidden_size_ratio=config.get('tf_mlp_hidden_size_ratio', 4.0),
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
    return load_from_checkpoint(state_dict, mask_prediction=mask_prediction, deterministic=deterministic)
