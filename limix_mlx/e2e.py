"""End-to-end check: reference preprocessing + MLX backend on sklearn data."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split

from api import LimiXMLXPredictor

CKPT = (str(Path.home()) + "/.cache/huggingface/hub/models--stable-ai--LimiX-2/"
        "snapshots/de07b679e74a41b50b9de18251a8fa245e537440/LimiX-2.ckpt")
SRC = str(Path(__file__).resolve().parents[1])


def main():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    Xtr, Xte = Xtr[:200], Xte[:50]
    ytr, yte = ytr[:200], yte[:50]
    clf = LimiXMLXPredictor(
        model_path=CKPT,
        inference_config=f"{SRC}/config/cls_default_noretrieval_v2.json",
        seed=0)
    proba = clf.predict(Xtr, ytr, Xte, task_type="Classification")
    proba = np.asarray(proba)
    print("cls proba shape:", proba.shape, "rowsum:", proba.sum(1)[:3])
    pred = proba.argmax(1)
    print("cls accuracy:", (pred == yte).mean())

    Xr, yr = load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.2, random_state=0)
    Xtr, Xte = Xtr[:200], Xte[:50]
    ytr, yte = ytr[:200], yte[:50]
    reg = LimiXMLXPredictor(
        model_path=CKPT,
        inference_config=f"{SRC}/config/reg_default_noretrieval_v2.json",
        seed=0)
    pred = np.asarray(reg.predict(Xtr, ytr, Xte, task_type="Regression"))
    print("reg pred shape:", pred.shape)
    rmse = float(np.sqrt(np.mean((pred - yte) ** 2)))
    print("reg RMSE:", rmse)


if __name__ == "__main__":
    main()
