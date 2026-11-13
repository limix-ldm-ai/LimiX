"""Converts the LimiX-2 torch checkpoint to MLX (.npz) weights.

Module/param names are identical between limix_src/model/v2_0 and
limix_mlx, so conversion is a pure format translation (no key remapping).
Needs torch (for unpickling the .ckpt) + mlx + numpy.

Usage:
  .venv/bin/python limix_mlx/convert.py <LimiX-2.ckpt> <out.npz>
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shim  # noqa: F401

import mlx.core as mx
import numpy as np


def convert_ckpt(ckpt_path: str, npz_path: str) -> str:
    import torch

    npz_path = os.path.expanduser(npz_path)
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)

    sys.path.insert(
        0, str(Path(__file__).resolve().parents[1]))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    weights = {}
    skipped = []
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            skipped.append(k)
            continue
        weights[k] = v.detach().cpu().numpy()
    # NOTE: mx.savez caps kwargs at 1024; numpy savez has no such cap and
    # mx.load reads numpy .npz files directly.
    np.savez(npz_path, **weights)
    n_params = sum(int(np.prod(v.shape)) for v in weights.values())
    print(f"converted {len(weights)} tensors, {n_params / 1e6:.1f}M params "
          f"-> {npz_path} ({os.path.getsize(npz_path) / 1e9:.2f} GB)")
    if skipped:
        print("skipped non-tensors:", skipped)
    return npz_path


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ckpt_path")
    p.add_argument("npz_path")
    args = p.parse_args(argv)
    convert_ckpt(args.ckpt_path, args.npz_path)


if __name__ == "__main__":
    main()
