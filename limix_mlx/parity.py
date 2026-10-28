"""Parity: torch CPU baseline vs MLX port (same weights, same inputs).

Controls the per-forward random positional embedding by fixing torch.randn.
Usage: .venv/bin/python limix_mlx/parity.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shim  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from model.v2_0.loading import load_model
from baseline_torch import CKPT

import mlx.core as mx
from loader import build_model, load as mlx_load


def main():
    rng = np.random.default_rng(0)
    N, F, EVAL = 60, 6, 40
    X = rng.normal(size=(1, N, F)).astype(np.float32)
    X[:, 50, 0] = np.nan  # a missing value exercises the mask path
    y_cls = rng.integers(0, 2, size=(1, N)).astype(np.float32)
    y_reg = (rng.normal(size=(1, N)) * 10).astype(np.float32)

    # fixed positional-embedding noise, shared by both backends
    fg = (F + 1) // 2
    pos_np = rng.standard_normal((fg, 64)).astype(np.float32)

    torch_model, config = load_model(CKPT)
    torch_model.eval()

    orig_randn = torch.randn

    def fixed_randn(*shape, **kw):
        shape = tuple(shape[0]) if len(shape) == 1 and isinstance(
            shape[0], (list, tuple)) else tuple(shape)
        if tuple(shape) == (fg, 64):
            g = kw.get("generator", None)
            dev = kw.get("device", "cpu")
            dt = kw.get("dtype", torch.float32)
            return torch.from_numpy(pos_np).to(device=dev, dtype=dt)
        return orig_randn(*shape, **kw)

    mlx_model, _ = mlx_load()

    for task, y in [("Classification", y_cls), ("Regression", y_reg),
                    ("Feature_imputation", y_reg)]:
        torch.randn = fixed_randn
        try:
            with torch.no_grad():
                tout = torch_model(torch.from_numpy(X), torch.from_numpy(y),
                                   eval_pos=EVAL, task_type=task)
        finally:
            torch.randn = orig_randn
        mout = mlx_model(X, y, eval_pos=EVAL, task_type=task,
                         pos_emb=pos_np)
        print(f"--- {task} ---")
        for k in tout:
            tv = tout[k]
            if isinstance(tv, list):
                for i, t in enumerate(tv):
                    mv = np.array(mout[k][i])
                    d = np.abs(t.numpy() - mv).max()
                    print(f"  {k}[{i}]: shape {tuple(t.shape)} max|diff| {d:.3e}")
            elif isinstance(tv, dict):
                continue
            else:
                mv = np.array(mout[k])
                d = np.abs(tv.numpy() - mv).max()
                print(f"  {k}: shape {tuple(tv.shape)} max|diff| {d:.3e}")
    mx.eval(mx.array([1]))


if __name__ == "__main__":
    main()
