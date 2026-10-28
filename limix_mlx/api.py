"""LimiX-2 on MLX with the reference sklearn-style API.

Architecture: reuse the ENTIRE reference v2 predictor stack (pipeline
preprocessing, class permutations, temperature scaling, ensemble averaging,
bucket decoding, imputation inversion) and swap only the compute backend:
self.model is an MLX-backed shim with the same call signature as the torch
model. Ensemble/parse/decode logic is untouched, so numerics match the
reference by construction (up to the 1e-5 forward diff).

Usage:
    from api import LimiXMLXPredictor
    clf = LimiXMLXPredictor(
        model_path="<LimiX-2.ckpt>",
        inference_config="<cls_default_noretrieval_v2.json>")
    proba = clf.predict(X_train, y_train, X_test, task_type="Classification")
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shim  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

import mlx.core as mx
from loader import build_model, load_weights_flat

import inference.v2_0.predictor as _pred_mod
from inference.v2_0.predictor import LimiXPredictor


class MLXModelShim:
    """Callable with the torch model's forward signature, computed by MLX."""

    def __init__(self, mlx_model, config, reg_borders_torch, seed=None):
        self.mlx_model = mlx_model
        self.config = config
        self._reg_borders = reg_borders_torch
        self._embed_dim = config["embed_dim"]
        self._seed = seed

    # -- torch nn.Module API surface used by the predictor --
    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self

    def parameters(self):
        return []

    def buffers(self):
        return []

    def __call__(self, x, y, eval_pos, task_type="Classification", **kwargs):
        # A missing seed fails loudly: silently falling back to the global
        # stream would make outputs depend on unrelated RNG consumption.
        seed = getattr(self, '_seed', None)
        if seed is None:
            raise ValueError(
                'MLXModelShim requires a seed for the feature positional '
                'code (the reference reseeds torch with its predictor seed '
                'before each forward). Pass seed= when constructing the shim.')
        xb = x.detach().cpu().numpy().astype(np.float32)
        yb = y.detach().cpu().numpy().astype(np.float32)
        b, s, f = xb.shape
        g = self.mlx_model.features_per_group
        fg = (f + (g - f % g) % g) // g
        # First randn of the torch forward is the feature positional code.
        # Draw it from a DEDICATED generator seeded like the reference, so
        # MLX outputs don't depend on / pollute global torch RNG state.
        gen = torch.Generator()
        gen.manual_seed(seed)
        with torch.no_grad():
            pos = torch.randn((fg, self._embed_dim // 4),
                              generator=gen).numpy().astype(np.float32)
        out = self.mlx_model(xb, yb, eval_pos=eval_pos, task_type=task_type,
                             pos_emb=pos)
        torch_out = {}
        if "cls_output" in out:
            torch_out["cls_output"] = torch.from_numpy(
                np.array(out["cls_output"]).astype(np.float32))
        if "reg_output" in out:
            torch_out["reg_output"] = [
                torch.from_numpy(np.array(t).astype(np.float32))
                for t in out["reg_output"]]
        if "feature_pred" in out:
            fp = np.array(out["feature_pred"]).astype(np.float32)
            torch_out["feature_pred"] = torch.from_numpy(fp)
            pc = out["feature_process_config"]
            mean = np.asarray(pc["mean_for_normalization"], dtype=np.float32)
            std = np.asarray(pc["std_for_normalization"], dtype=np.float32)
            num = np.asarray(pc["valid_feature_num"], dtype=np.float32)
            torch_out["feature_process_config"] = {
                "n_x_padding": int(pc["n_x_padding"]),
                "features_per_group": int(g),
                "num_used_features": torch.from_numpy(num),
                "mean_for_normalization": torch.from_numpy(mean),
                "std_for_normalization": torch.from_numpy(std),
            }
        return torch_out


class LimiXMLXPredictor(LimiXPredictor):
    """LimiXPredictor with the forward pass executed on MLX."""

    # Reference CacheManager defaults to /mnt/... (read-only on Mac).
    CacheManager = None  # set below after class creation

    def __init__(self, model_path, inference_config,
                 npz_path=None, **kwargs):
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        config = ckpt["config"]
        reg_borders = torch.from_numpy(
            np.asarray(ckpt["state_dict"]["_reg_borders"]).astype(np.float32))
        del ckpt

        if npz_path is None:
            npz_path = os.path.join(os.path.expanduser("~"), ".cache",
                                    "limix_mlx", "LimiX-2.npz")
        if not os.path.exists(npz_path):
            from convert import convert_ckpt
            convert_ckpt(model_path, npz_path)

        mlx_model = build_model(config)
        weights = mx.load(npz_path)
        if isinstance(weights, (list, tuple)):
            weights = dict(weights)
        from mlx.utils import tree_flatten
        have = {k for k, _ in tree_flatten(mlx_model.parameters())}
        load_weights_flat(
            mlx_model, {k: weights[k] for k in have if k in weights})
        mx.eval(mlx_model.parameters())

        orig_loader = _pred_mod._load_model_for_predictor

        def _stub(model_path, reuse_frozen_model, ckpt=None,
                  deterministic=False):
            return None, config

        _pred_mod._load_model_for_predictor = _stub
        try:
            kwargs.setdefault("device", torch.device("cpu"))
            super().__init__(model_path=model_path,
                             inference_config=inference_config, **kwargs)
        finally:
            _pred_mod._load_model_for_predictor = orig_loader
        self.model = MLXModelShim(mlx_model, config, reg_borders,
                                    seed=self.seed)
        self._mlx_model = mlx_model


class _MacCacheManager(LimiXPredictor.CacheManager):
    def __init__(self, cache_dir=None):
        super().__init__(cache_dir=os.path.join(
            os.path.expanduser("~"), ".cache", "limix_mlx", "infe_cache"))


LimiXMLXPredictor.CacheManager = _MacCacheManager
