"""Builds the MLX LimiX-2 model from the checkpoint config and loads weights.

Mirrors limix_src/model/v2_0/loading.build_model so structure matches the
torch model by construction; load_weights asserts the match.

Usage:
  .venv/bin/python limix_mlx/loader.py   # smoke test
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shim  # noqa: F401

import mlx.core as mx

from lxm_model import EncoderBaseLayer
from lxm_top import FeaturesTransformer, build_model

CKPT = (str(Path.home()) + "/.cache/huggingface/hub/models--stable-ai--LimiX-2/"
        "snapshots/de07b679e74a41b50b9de18251a8fa245e537440/LimiX-2.ckpt")


def load_weights_flat(model, weights: dict):
    """Assigns flat dotted-key weights (torch naming) onto the MLX module tree.

    Needed because mlx's load_weights round-trips numeric path components
    through lists, while torch nn.Sequential names children '0','1',... as
    attributes (mirrored here via setattr).
    """
    for key, arr in weights.items():
        parts = key.split(".")
        obj = model
        for p in parts[:-1]:
            if isinstance(obj, list):
                obj = obj[int(p)]
            else:
                obj = getattr(obj, p)
        setattr(obj, parts[-1], mx.array(arr) if not isinstance(arr, mx.array)
                else arr)


def load(ckpt_path: str = CKPT, npz_path: str | None = None):
    """Builds the model, converts/loads weights, returns (model, config)."""
    if npz_path is None:
        npz_path = os.path.join(os.path.expanduser("~"), ".cache",
                                "limix_mlx", "LimiX-2.npz")
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[1]))
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = build_model(config)

    if not os.path.exists(npz_path):
        os.makedirs(os.path.dirname(npz_path), exist_ok=True)
        from convert import convert_ckpt
        convert_ckpt(ckpt_path, npz_path)

    weights = mx.load(npz_path)
    if isinstance(weights, (list, tuple)):
        weights = dict(weights)
    from mlx.utils import tree_flatten
    have = {k for k, _ in tree_flatten(model.parameters())}
    # buffers stored in ckpt (set directly below, not via load_weights)
    extra_buffers = {"_reg_borders"}
    missing = have - set(weights) - extra_buffers
    extra = set(weights) - have - extra_buffers
    assert not missing, f"missing weights: {sorted(missing)[:10]}"
    assert not extra, f"extra weights: {sorted(extra)[:10]}"
    load_weights_flat(model, {k: weights[k] for k in have if k in weights})
    sd = ckpt["state_dict"]
    for name in ("_reg_borders", "reg_log_widths"):
        if name in sd:
            setattr(model, name, mx.array(sd[name].numpy()))
    mx.eval(model.parameters())
    return model, config


if __name__ == "__main__":
    import torch
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[1]))
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    model = build_model(ckpt["config"])
    from mlx.utils import tree_flatten
    mine = {k for k, _ in tree_flatten(model.parameters())} | {"_reg_borders"}
    theirs = {k for k, v in sd.items() if isinstance(v, torch.Tensor)}
    print("torch tensors:", len(theirs), "mlx params:", len(mine))
    print("missing in mlx:", sorted(theirs - mine)[:20])
    print("extra in mlx:", sorted(mine - theirs)[:20])
