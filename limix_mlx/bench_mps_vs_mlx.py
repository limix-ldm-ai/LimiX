"""Head-to-head: LimiX-2 torch/MPS vs MLX-native on Apple silicon.

Same data, same inference config, same seeds; interleaved A/B trials.
Compares the reference LimiXPredictor (torch backend on MPS) against
LimiXMLXPredictor (MLX backend), end to end through preprocessing.
"""
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shim  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

import inference.v2_0.predictor as _pred_mod
from inference.v2_0.predictor import LimiXPredictor
from api import LimiXMLXPredictor, _MacCacheManager

CKPT = (str(Path.home()) + "/.cache/huggingface/hub/models--stable-ai--LimiX-2/"
        "snapshots/de07b679e74a41b50b9de18251a8fa245e537440/LimiX-2.ckpt")
SRC = str(Path(__file__).resolve().parents[1])

WARMUP = 1
TRIALS = 3


def make_data():
    rng = np.random.default_rng(0)
    N, Nt = 150, 30
    Xtr = rng.normal(size=(N, 5))
    ytr = rng.integers(0, 2, size=N)
    Xte = rng.normal(size=(Nt, 5))
    return Xtr, ytr, Xte


def main():
    _pred_mod.LimiXPredictor.CacheManager = _MacCacheManager
    cfg = f"{SRC}/config/cls_default_noretrieval_v2.json"
    Xtr, ytr, Xte = make_data()
    print("loading mps predictor...", flush=True)
    mps = LimiXPredictor(device=torch.device("mps"), model_path=CKPT,
                         inference_config=cfg, seed=0)
    print("loading mlx predictor...", flush=True)
    mlx = LimiXMLXPredictor(model_path=CKPT, inference_config=cfg, seed=0)

    for _ in range(WARMUP):
        mlx.predict(Xtr, ytr, Xte, "Classification")
    for _ in range(WARMUP):
        np_out = mps.predict(Xtr, ytr, Xte, "Classification")
    t_mlx, t_mps, out_mlx, out_mps = [], [], None, None
    for _ in range(TRIALS):
        t = time.perf_counter()
        out_mlx = np.asarray(mlx.predict(Xtr, ytr, Xte, "Classification"))
        t_mlx.append(time.perf_counter() - t)
        t = time.perf_counter()
        out_mps = np.asarray(mps.predict(Xtr, ytr, Xte, "Classification"))
        t_mps.append(time.perf_counter() - t)
    f_mlx, f_mps = float(np.median(t_mlx)), float(np.median(t_mps))
    print(f"predict | MLX {f_mlx:6.2f}s | MPS {f_mps:6.2f}s | "
          f"speedup MLX/MPS: {f_mps / f_mlx:5.2f}x")
    print(f"class agreement "
          f"{(out_mlx.argmax(1) == out_mps.argmax(1)).mean() * 100:.1f}%, "
          f"proba max|diff| {np.abs(out_mlx - out_mps).max():.2e}")
    print("done.")


if __name__ == "__main__":
    main()
