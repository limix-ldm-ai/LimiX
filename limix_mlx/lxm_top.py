"""MLX port, part 2: encoders, preprocessing, FeaturesTransformer, builder.

Imports core blocks from model.py. Module/parameter names mirror the torch
checkpoint (encoder_x.0.*, cls_y_encoder.2.*, reg_y_encoder.0.*,
feature_positional_embedding, add_task_info.*, *_decoder.*, *_post_adapter.*,
_reg_borders, reg_log_widths).
"""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from lxm_model import RMSNorm, _ACTS


class Seq(nn.Module):
    """Sequential with torch-style numeric child names ('0','1',...)."""

    def __init__(self, *mods):
        super().__init__()
        for i, m in enumerate(mods):
            setattr(self, str(i), m)
        self._n = len(mods)

    def __call__(self, x):
        for i in range(self._n):
            x = getattr(self, str(i))(x)
        return x


# ------------------------------------------------- parameter-free preprocessing

def _calc_mean(x, axis=1):
    cnt = np.sum(~np.isnan(x), axis=axis)
    cnt = np.clip(cnt, 1.0, None)
    return np.nansum(x, axis=axis) / cnt, cnt


def _calc_std(x, axis=1, mean_v=None, value_num=None):
    if mean_v is None or value_num is None:
        mean_v, value_num = _calc_mean(x, axis=axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        diff = np.expand_dims(mean_v, axis) - x
        var = np.nansum(diff ** 2, axis=axis) / (value_num - 1)
    return np.sqrt(var)


class PreprocessPipeline:
    """torch encoders.PreprocessPipeline (Nan -> Norm(train-only) -> Valid)."""

    def __init__(self, num_features, normalize_on_train_only=True,
                 normalize_x=True, remove_outliers=False,
                 normalize_by_used_features=True):
        self.num_features = num_features
        self.train_only = normalize_on_train_only
        self.normalize_x = normalize_x
        self.normalize_by_used = normalize_by_used_features
        self.mean = None
        self.std = None
        self.valid_feature_num = None

    def __call__(self, data, mask, eval_pos):
        # NanEncoder: fill nan/inf with train mean
        with np.errstate(all="ignore"):
            m, _ = _calc_mean(data[:, :eval_pos], axis=1)  # [B,Fg,G]
        m = np.expand_dims(m, 1)
        bad = np.isnan(data) | np.isinf(data)
        data = np.where(bad, m, data)
        # NormalizationEncoder (train-only stats)
        pos = eval_pos if self.train_only else -1
        seg = data[:, :pos] if pos != -1 else data
        with np.errstate(all="ignore"):
            mean, num = _calc_mean(seg, axis=1)
            std = _calc_std(seg, axis=1, mean_v=mean, value_num=num) + 1e-20
        if data.shape[1] == 1 or eval_pos == 1:
            std[:] = 1.0
        if self.normalize_x:
            data = (data - np.expand_dims(mean, 1)) / np.expand_dims(std, 1)
            data = np.clip(data, -100, 100)
        self.mean, self.std = mean, std
        # ValidFeatureEncoder
        with np.errstate(all="ignore"):
            valid = ~(data == data[:, 0:1, :]).all(axis=1)
        num = np.clip(valid.sum(-1, keepdims=True), 1, None)
        self.valid_feature_num = num
        if self.normalize_by_used:
            data = data * np.sqrt(self.num_features / num)[:, None, :, :]
        return data


# ------------------------------------------------------------------ encoders

class MaskEmbEncoder(nn.Module):
    """torch encoders.MaskEmbEncoder (numeric MLP + mask emb + fusion)."""

    def __init__(self, num_features, emsize):
        super().__init__()
        self.mask_embedding = nn.Embedding(1, emsize)
        self.numeric_mlp = Seq(
            nn.Linear(1, emsize // 2),
            RMSNorm(emsize // 2, 1e-5, True),
            _ACTS["relu"],
            nn.Linear(emsize // 2, emsize),
            RMSNorm(emsize, 1e-5, True),
            _ACTS["relu"],
        )
        self.fusion_network = Seq(
            nn.Linear(num_features * emsize, emsize),
            RMSNorm(emsize, 1e-5, True),
            _ACTS["relu"],
            nn.Linear(emsize, emsize),
            RMSNorm(emsize, 1e-5, True),
        )

    def __call__(self, x, mask):
        # x: [B,S,Fg,G] float32, mask bool (numpy or mx)
        x = mx.array(np.asarray(x, dtype=np.float32))
        mask = mx.array(np.asarray(mask, dtype=bool))
        b, s, fg, g = x.shape
        h = self.numeric_mlp(x.reshape(-1, 1)).reshape(b, s, fg, g, -1)
        mv = self.mask_embedding.weight[0].reshape(1, 1, 1, 1, -1)
        h = mx.where(mask.reshape(b, s, fg, g, 1), mv, h)
        h = h.reshape(b, s, fg, g * h.shape[-1])
        return self.fusion_network(h)


class EncoderX(nn.Module):
    def __init__(self, num_features, emsize):
        super().__init__()
        setattr(self, "0", MaskEmbEncoder(num_features, emsize))

    def __call__(self, x, mask):
        return getattr(self, "0")(x, mask)


class ClsYEncoder(nn.Module):
    """torch Sequential[NanEncoder, MulticlassTargetEncoder, EmbYEncoderStep]."""

    def __init__(self, emsize, n_classes=10, num_tokens=4):
        super().__init__()
        # torch layout: cls_y_encoder.0=NanEncoder, .1=Multiclass (no params),
        # .2=EmbYEncoderStep.
        self.num_tokens = num_tokens
        setattr(self, "2", EmbYEncoderStep(emsize, n_classes, num_tokens))

    def _encode(self, y, eval_pos):
        y = y.copy()
        with np.errstate(all="ignore"):
            m, _ = _calc_mean(y[:, :eval_pos], axis=1)
        bad = np.isnan(y) | np.isinf(y)
        y = np.where(bad, np.expand_dims(m, 1), y)
        # MulticlassTargetEncoder: rank-map train labels
        for b in range(y.shape[0]):
            u = np.unique(y[b, :eval_pos])
            y[b, :eval_pos] = (y[b, :eval_pos, None] > u[None, :]).sum(-1)
        return getattr(self, "2")(y, eval_pos)


class EmbYEncoderStep(nn.Module):
    def __init__(self, emsize, n_classes=10, num_tokens=4):
        super().__init__()
        self.num_tokens = num_tokens
        self.y_embedding = nn.Embedding(n_classes, emsize)
        self.y_mask = nn.Embedding(1, emsize)

    def __call__(self, y, eval_pos):
        yt = y[:, :eval_pos].astype(np.int32)
        k = self.num_tokens
        b, sq = y.shape[0], y.shape[1] - eval_pos
        e = self.y_embedding.weight.shape[1] // k
        tr = self.y_embedding(mx.array(yt)).reshape(b, -1, k, e)
        te = mx.broadcast_to(self.y_mask(mx.zeros((1,), dtype=mx.int32)),
                             (b, sq, k * e)).reshape(b, sq, k, e)
        return mx.concatenate([tr, te], axis=1)


class RegYEncoder(nn.Module):
    """torch Sequential[YNoneEmbeddingEncoder] with MLP_GELU_postnorm."""

    def __init__(self, emsize):
        super().__init__()
        # torch: reg_y_encoder.0.{none_embedding, numeric_encoder.{0,2,3}}
        setattr(self, "0", RegYNoneEncoder(emsize))

    def __call__(self, y):
        return getattr(self, "0")(y)


class RegYNoneEncoder(nn.Module):
    def __init__(self, emsize):
        super().__init__()
        hidden = emsize // 2
        self.none_embedding = nn.Embedding(1, emsize)
        self.numeric_encoder = Seq(
            nn.Linear(1, hidden),
            _ACTS["gelu"],
            nn.Linear(hidden, emsize),
            RMSNorm(emsize, 1e-5, True),
        )

    def __call__(self, y):
        y = np.asarray(y)
        if y.ndim == 3 and y.shape[-1] == 1:
            y = y[..., 0]
        none = np.isnan(y) | np.isinf(y)
        filled = np.where(none, 0.0, y)
        h = self.numeric_encoder(mx.array(filled[..., None].astype(np.float32)))
        ne = self.none_embedding(mx.zeros((1,), dtype=mx.int32)).reshape(
            (1,) * (h.ndim - 1) + (-1,))
        ne = mx.broadcast_to(ne, h.shape)
        cond = none.reshape((1,) * 0 + none.shape + (1,) * (h.ndim - none.ndim))
        return mx.where(mx.array(cond), ne, h)


# ------------------------------------------------------------- misc modules

def _free_mlp(pattern, in_dim, out_dim, y_token, bias=True):
    """torch utils.create_mlp_4_free_type."""
    mods = []
    last = in_dim
    for el in pattern.split("_"):
        parts = el.split("-")
        if parts[0] == "linear":
            d = int(parts[1])
            cur = d if (y_token == 0 or d in (10, 5000)) else d * y_token
            mods.append(nn.Linear(last, cur, bias=bias))
            last = cur
        elif parts[0] in _ACTS:
            mods.append(_ACTS[parts[0]])
        elif parts[0] in ("Norm", "RMSNorm"):
            mods.append(RMSNorm(last, 1e-5, True))
        else:
            raise ValueError(f"bad pattern element {el}")
    assert last == out_dim, f"{pattern}: {last} != {out_dim}"
    return Seq(*mods)


class AdapterWithResidual(nn.Module):
    def __init__(self, adapter, dim, use_residual=False):
        super().__init__()
        self.adapter = adapter
        self.layer_norm = RMSNorm(dim, 1e-5, True)
        self.use_residual = use_residual

    def __call__(self, x, **kwargs):
        a = self.adapter(x)
        if self.use_residual:
            return self.layer_norm(x + a)
        return self.layer_norm(a)


class AddTaskInfo(nn.Module):
    def __init__(self, embedding_size, num_y_tokens):
        super().__init__()
        self.num_y_tokens = num_y_tokens
        self.token_type_embedding = nn.Embedding(3, embedding_size)

    def __call__(self, all_emb, y_type):
        k = self.num_y_tokens
        x, y = all_emb[..., :-k, :], all_emb[..., -k:, :]
        t = self.token_type_embedding(
            mx.array(y_type.astype(np.int32)))[:, :, None, :]
        return mx.concatenate([x, y + t], axis=-2)


class FeatureDecoderMLP(nn.Module):
    """torch feature_decoder: Linear -> RMSNorm -> GELU -> Linear."""

    def __init__(self, embed_dim, hidden, out_dim):
        super().__init__()
        setattr(self, "0", nn.Linear(embed_dim, hidden))
        setattr(self, "1", RMSNorm(hidden, 1e-5, True))
        setattr(self, "2", _ACTS["gelu"])
        setattr(self, "3", nn.Linear(hidden, out_dim))

    def __call__(self, x):
        return getattr(self, "3")(
            getattr(self, "2")(
                getattr(self, "1")(getattr(self, "0")(x))))


# ------------------------------------------------------- FeaturesTransformer

class FeaturesTransformer(nn.Module):
    def __init__(self, *, embed_dim=256, num_cls_tokens=4, features_per_group=2,
                 num_classes=10, num_buckets=5000,
                 layers, feature_emb_layer, reg_y_emb_layer, cls_y_emb_layer,
                 cls_decoder_pattern, reg_decoder_pattern,
                 feat_post_pattern=None, cls_post_pattern=None,
                 reg_post_pattern=None, use_feat_post=False,
                 use_cls_post=False, use_reg_post=False,
                 feat_dec_hidden=1024):
        super().__init__()
        from lxm_model import LayerStack
        self.embed_dim = embed_dim
        self.y_token_k = num_cls_tokens
        self.uses_decoupled = True
        self.features_per_group = features_per_group
        self.num_classes = num_classes
        yd = embed_dim * num_cls_tokens
        self.x_preprocess = PreprocessPipeline(num_features=features_per_group)
        self.encoder_x = EncoderX(features_per_group, embed_dim)
        self.cls_y_encoder = ClsYEncoder(yd, num_classes)
        self.reg_y_encoder = RegYEncoder(yd)
        self.transformer_encoder = LayerStack(
            layers, num_cls_tokens, feature_emb_layer, reg_y_emb_layer,
            cls_y_emb_layer)
        self.encoder_out_norm = RMSNorm(embed_dim, 1e-5, False)
        self.feature_encoder_out_norm = self.encoder_out_norm
        self.reg_y_encoder_out_norm = self.encoder_out_norm
        self.cls_y_encoder_out_norm = self.encoder_out_norm
        if use_cls_post:
            self.cls_post_adapter = AdapterWithResidual(
                _free_mlp(cls_post_pattern, yd, yd, num_cls_tokens), yd, False)
        if use_reg_post:
            self.reg_post_adapter = AdapterWithResidual(
                _free_mlp(reg_post_pattern, yd, yd, num_cls_tokens), yd, False)
        if use_feat_post:
            self.feature_post_adapter = AdapterWithResidual(
                _free_mlp(feat_post_pattern, embed_dim, embed_dim, 0),
                embed_dim, False)
        self.cls_y_decoder = _free_mlp(cls_decoder_pattern, yd, num_classes,
                                      num_cls_tokens)
        self.reg_y_decoder = _free_mlp(reg_decoder_pattern, yd, num_buckets,
                                      num_cls_tokens)
        self.feature_decoder = FeatureDecoderMLP(
            embed_dim, feat_dec_hidden, features_per_group)
        self.feature_positional_embedding = nn.Linear(embed_dim // 4, embed_dim)
        self.add_task_info = AddTaskInfo(embed_dim, num_cls_tokens)
        self._reg_borders = mx.zeros((num_buckets + 1,))
        self.reg_log_widths = mx.zeros((num_buckets,))

    # -- helpers ------------------------------------------------------
    def _pad_xy(self, x, y):
        b, s, f = x.shape
        add = (self.features_per_group - f % self.features_per_group
               ) % self.features_per_group
        if add:
            x = np.concatenate([x, np.zeros((b, s, add), np.float32)], -1)
        yy = y[:, :, None].astype(np.float32)
        if yy.shape[1] < x.shape[1]:
            pad = np.full((b, x.shape[1] - yy.shape[1], 1), np.nan,
                          np.float32)
            yy = np.concatenate([yy, pad], axis=1)
        return x, yy, add

    def _encode_x(self, data, mask, eval_pos):
        # mask_process_4_x (inference masks are 0/1 -> NaN + bool)
        with np.errstate(all="ignore"):
            col_mean = np.nanmean(data, axis=1, keepdims=True)
        col_mean = np.where(np.isnan(col_mean), 0, col_mean)
        data = np.where(mask == 1, np.nan, data)
        data = np.where(mask == 2, col_mean, data)
        mask = mask.astype(bool)
        pre = self.x_preprocess(data, mask, eval_pos)
        real_x = pre.copy()
        return self.encoder_x(pre, mask), real_x

    def _encode_y(self, y1, eval_pos, task_type):
        k = self.y_token_k

        def _tok(emb):
            a = np.array(emb)
            if a.ndim == 4 and a.shape[2] == 1:
                a = a.reshape(a.shape[0], a.shape[1], k, -1)
                return mx.array(a)
            if a.ndim == 3:
                return mx.array(a.reshape(a.shape[0], a.shape[1], k, -1))
            return emb

        if task_type in ("Classification", "Classification-Feature_imputation"):
            yt = np.zeros(y1.shape[:2], np.float32)
            en_c = en_r = None
            if task_type == "Classification":
                en_c = True
            else:
                en_c, en_r = True, True
            emb = None
            if en_c:
                emb_c = self.cls_y_encoder._encode(y1, eval_pos)
            if en_r:
                emb_r = _tok(self.reg_y_encoder(y1))
                yt = np.ones(y1.shape[:2], np.float32)
            if en_c and en_r:
                m = yt[..., None, None].astype(bool)
                emb = mx.where(mx.array(m), mx.array(np.array(emb_r)),
                               mx.array(np.array(emb_c)))
            elif en_c:
                emb = emb_c
                yt = np.zeros(y1.shape[:2], np.float32)
            else:
                emb = emb_r
                yt = np.ones(y1.shape[:2], np.float32)
            return emb, yt, en_c, en_r, False
        elif task_type in ("Regression", "Regression-Feature_imputation"):
            emb = _tok(self.reg_y_encoder(y1))
            yt = np.ones(y1.shape[:2], np.float32)
            feat = task_type == "Regression-Feature_imputation"
            return emb, yt, False, True, feat
        else:  # Feature_imputation
            emb = _tok(self.reg_y_encoder(y1))
            yt = np.ones(y1.shape[:2], np.float32)
            return emb, yt, False, True, True

    def __call__(self, x_in, y_in, eval_pos, task_type="Classification",
                 pos_emb=None):
        x, y1, feature_to_add = self._pad_xy(
            np.asarray(x_in, dtype=np.float32),
            np.asarray(y_in, dtype=np.float32))
        b, s, f = x.shape
        g = self.features_per_group
        fg = f // g
        xg = x.reshape(b, s, fg, g)
        x_mask = np.isnan(x).astype(np.int32).reshape(b, s, fg, g)
        y1[:, eval_pos:] = np.nan
        x_emb, real_x = self._encode_x(xg, x_mask, eval_pos)
        emb_y, y_type, en_c, en_r, en_feat = self._encode_y(
            y1, eval_pos, task_type)
        xe = np.array(x_emb) + np.array(self.feature_positional_embedding(
            mx.array(pos_emb)))
        all_emb = mx.concatenate([mx.array(xe), emb_y], axis=2)
        if int(np.isnan(np.array(all_emb)).sum()) != 0:
            raise ValueError("embedded_all contains NaN")
        # feature mask (padding only; enable_feature_attention_mask is off)
        valid = np.ones((b, fg), bool)
        fm_pad = np.broadcast_to(~valid[:, None, :], (b, s, fg))
        xfm = None
        if self.uses_decoupled:
            xfm = np.concatenate(
                [fm_pad, np.zeros((b, s, self.y_token_k), bool)], axis=-1)
        all_emb = self.add_task_info(all_emb, y_type)
        enc_out, feature_emb = self.transformer_encoder(
            all_emb, xfm, eval_pos)
        sq = s - eval_pos
        out = {}
        if en_feat:
            fe = feature_emb if feature_emb is not None else enc_out
            fe = self.feature_encoder_out_norm(fe)
            fe = fe[:, :, :-self.y_token_k, :]
            if hasattr(self, "feature_post_adapter"):
                fe = self.feature_post_adapter(fe)
            gm = x_mask.any(axis=-1)
            pred = np.array(real_x)
            cells = np.array(fe)[gm]
            if cells.shape[0]:
                filled = np.array(self.feature_decoder(mx.array(cells)))
                m = np.broadcast_to(gm[..., None], x_mask.shape)
                rx = np.array(real_x)
                rx[m] = np.where(x_mask[m], filled.ravel(), rx[m])
                pred = rx
            out["feature_pred"] = pred.reshape(b, s, fg, g)
            out["feature_process_config"] = {
                "n_x_padding": feature_to_add,
                "mean_for_normalization": self.x_preprocess.mean,
                "std_for_normalization": self.x_preprocess.std,
                "valid_feature_num": self.x_preprocess.valid_feature_num,
            }
        if en_r:
            re_ = enc_out[:, eval_pos:, -self.y_token_k:, :]
            re_ = self.reg_y_encoder_out_norm(re_)
            re_ = re_.reshape(b, sq, -1)
            if hasattr(self, "reg_post_adapter"):
                re_ = self.reg_post_adapter(re_, task_type="reg")
            m = (y_type[:, eval_pos:] == 1)[..., None]
            re_ = mx.where(mx.array(m), re_, mx.zeros_like(re_))
            out["reg_output"] = [self.reg_y_decoder(re_)]
        if en_c:
            ce = enc_out[:, eval_pos:, -self.y_token_k:, :]
            ce = self.cls_y_encoder_out_norm(ce)
            ce = ce.reshape(b, sq, -1)
            if hasattr(self, "cls_post_adapter"):
                ce = self.cls_post_adapter(ce, task_type="cls")
            m = (y_type[:, eval_pos:] == 0)[..., None]
            ce = mx.where(mx.array(m), ce, mx.zeros_like(ce))
            out["cls_output"] = self.cls_y_decoder(ce)
        return out


def build_model(config: dict):
    from lxm_model import EncoderBaseLayer
    E = config["embed_dim"]
    K = int(config.get("num_cls_tokens", 1))
    NH = config["nhead"]
    ratio = config.get("tf_mlp_hidden_size_ratio", 2.672)
    even_fix = config.get("tf_mlp_hidden_size_2_even", False)

    def hid(d, r):
        h = int(d * r)
        return h + 1 if (even_fix and h % 2) else h

    y_ratio = config.get("tf_mlp_y_hidden_size_ratio") or ratio
    mlp_act = {"silu": "silu", "gelu": "gelu", "relu": "relu"}.get(
        config.get("tf_mlp_activation_fuction", "silu"), "silu")
    has_bias = config.get("tf_mlp_use_bias", True)
    sa_bounds = (config.get("sa_temp_upper_bound", 0.5),
                 config.get("sa_temp_lower_bound", 0.0),
                 config.get("sa_scale_base_bound", 5.0))
    fa_bounds = (config.get("dsti_softmax_scaling_temp_upper_bound", 0.4),
                 config.get("dsti_softmax_scaling_temp_lower_bound", 0.0),
                 config.get("dsti_softmax_scaling_base_bound", 1.0))
    msc = config.get("model_structure_config")
    layer_cfgs = msc["layers"] if msc else [{"arch": "smf"}] * config["nlayers"]
    layers = []
    for lc in layer_cfgs:
        assert lc.get("arch", "smf").lower() == "smf", lc
        layers.append(EncoderBaseLayer(
            E, lc.get("nhead", NH), hid(E, ratio), hid(E * K, y_ratio),
            K, has_bias, mlp_act, sa_bounds, fa_bounds,
            lc.get("layer_norm_eps", config.get("layer_norm_eps") or 1e-5),
            lc.get("tf_layer_norm_use_elementwise_affine",
                   config.get("tf_layer_norm_use_elementwise_affine", False))))
    return FeaturesTransformer(
        embed_dim=E, num_cls_tokens=K,
        features_per_group=config.get("features_per_group", 2),
        num_classes=config["decoder_config"]["num_classes"],
        num_buckets=config.get("num_buckets", 0),
        layers=layers,
        feature_emb_layer=config.get("feature_emb_layer", config["nlayers"] - 1),
        reg_y_emb_layer=config.get("reg_y_emb_layer", config["nlayers"] - 1),
        cls_y_emb_layer=config.get("cls_y_emb_layer", config["nlayers"] - 1),
        cls_decoder_pattern=config.get(
            "cls_y_decoder_mlp_pattern", "linear-768_GELU_linear-10"),
        reg_decoder_pattern=config.get(
            "reg_y_decoder_mlp_pattern", "linear-768_GELU_linear-10"),
        feat_post_pattern=config.get("feature_post_adapter_pattern", "linear-192"),
        cls_post_pattern=config.get("cls_post_adapter_pattern", "linear-192"),
        reg_post_pattern=config.get("reg_post_adapter_pattern", "linear-192"),
        use_feat_post=config.get("use_feature_post_adapter", False),
        use_cls_post=config.get("use_cls_post_adapter", False),
        use_reg_post=config.get("use_reg_post_adapter", False),
        feat_dec_hidden=config.get("decoder_hidden_dim", 192 * 4),
    )
