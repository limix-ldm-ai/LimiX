"""End-to-end parity: reference torch-CPU predictor vs MLX predictor."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shim  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import inference.v2_0.predictor as _pred_mod
from inference.v2_0.predictor import LimiXPredictor
from api import LimiXMLXPredictor, _MacCacheManager

CKPT = (str(Path.home()) + "/.cache/huggingface/hub/models--stable-ai--LimiX-2/"
        "snapshots/de07b679e74a41b50b9de18251a8fa245e537440/LimiX-2.ckpt")
SRC = str(Path(__file__).resolve().parents[1])


def main():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    Xtr, Xte, ytr, yte = Xtr[:120], Xte[:20], ytr[:120], yte[:20]
    cfg = f"{SRC}/config/cls_default_noretrieval_v2.json"

    _pred_mod.LimiXPredictor.CacheManager = _MacCacheManager
    torch_pred = LimiXPredictor(
        device=torch.device("cpu"), model_path=CKPT,
        inference_config=cfg, seed=0)
    p_torch = np.asarray(torch_pred.predict(Xtr, ytr, Xte, "Classification"))
    mlx_pred = LimiXMLXPredictor(model_path=CKPT, inference_config=cfg, seed=0)
    p_mlx = np.asarray(mlx_pred.predict(Xtr, ytr, Xte, "Classification"))
    print("torch acc:", (p_torch.argmax(1) == yte).mean(),
          "mlx acc:", (p_mlx.argmax(1) == yte).mean())
    print("proba max|diff|:", np.abs(p_torch - p_mlx).max())
    print("class agreement:", (p_torch.argmax(1) == p_mlx.argmax(1)).mean())


if __name__ == "__main__":
    main()
