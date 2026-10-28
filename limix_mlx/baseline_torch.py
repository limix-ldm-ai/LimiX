"""Torch CPU baseline for LimiX-2 (Mac has no triton/nvtx/CUDA).

Stubs triton + nvtx at import so the reference implementation runs on CPU.
Usage: .venv/bin/python baseline_torch.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shim  # noqa: F401  (installs nvtx/triton stubs for Mac CPU)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from model.v2_0.loading import load_model

CKPT = (str(Path.home()) + "/.cache/huggingface/hub/models--stable-ai--LimiX-2/"
        "snapshots/de07b679e74a41b50b9de18251a8fa245e537440/LimiX-2.ckpt")


def main():
    torch.manual_seed(0)
    model, config = load_model(CKPT)
    model.eval()
    print("config: nlayers=%s embed=%s nhead=%s cls_tokens=%s buckets=%s" % (
        config["nlayers"], config["embed_dim"], config["nhead"],
        config.get("num_cls_tokens"), config.get("num_buckets")))

    rng = np.random.default_rng(0)
    N, F = 60, 6
    X = rng.normal(size=(1, N, F)).astype(np.float32)
    y = rng.integers(0, 2, size=(1, N)).astype(np.float32)
    eval_pos = 40
    with torch.no_grad():
        out = model(torch.from_numpy(X), torch.from_numpy(y),
                    eval_pos=eval_pos, task_type="Classification")
    print("out keys:", list(out.keys()))
    print("cls_output:", tuple(out["cls_output"].shape))
    logits = out["cls_output"][0, -20:].float().numpy()
    print("logits[0]:", logits[0])
    print("proba[0]:", torch.softmax(out["cls_output"][0, -20:], -1)[0].numpy())


if __name__ == "__main__":
    main()
