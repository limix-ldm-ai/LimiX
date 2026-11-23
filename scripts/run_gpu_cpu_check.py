#!/usr/bin/env python3
"""Run GPU/CPU inference subsets for all available LimiX checkpoints and compare."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = Path("/mnt/datagen/rg/05内部开源模型优化版")
DATA = ROOT / ".tmp_eval_data"
RESULT_ROOT = ROOT / "result" / "gpu_cpu_check"


@dataclass(frozen=True)
class ModelSpec:
    save_stem: str
    ckpt: str
    cls_cfg: str
    reg_cfg: str
    v2: bool


MODELS = [
    ModelSpec("LimiX-V2.0", "LimiX-2.ckpt", "ldm_limix/config/cls_default_noretrieval_v2.json", "ldm_limix/config/reg_default_noretrieval_v2.json", True),
    ModelSpec("LimiX-V1.5-7M", "LimiX-V1.5-7M.ckpt", "ldm_limix/config/cls_default_noretrieval.json", "ldm_limix/config/reg_default_noretrieval.json", False),
    ModelSpec("LimiX-V1.5-16M", "LimiX-V1.5-16M.ckpt", "ldm_limix/config/cls_default_noretrieval.json", "ldm_limix/config/reg_default_noretrieval.json", False),
    ModelSpec("LimiX-V1.5-64M", "LimiX-V1.5-64M.ckpt", "ldm_limix/config/cls_default_noretrieval.json", "ldm_limix/config/reg_default_noretrieval.json", False),
    ModelSpec("LimiX-V1.1_64M", "LimiX-V1.1_64M.ckpt", "ldm_limix/config/cls_default_noretrieval.json", "ldm_limix/config/reg_default_noretrieval.json", False),
    ModelSpec("LimiX-2M", "LimiX-2M.ckpt", "ldm_limix/config/cls_default_noretrieval.json", "ldm_limix/config/reg_default_noretrieval.json", False),
    ModelSpec("LimiX-16M", "LimiX-16M.ckpt", "ldm_limix/config/cls_default_noretrieval.json", "ldm_limix/config/reg_default_noretrieval.json", False),
]


def job_cmd(model: ModelSpec, task: str, device: str, gpuid: int | None) -> tuple[list[str], Path]:
    is_cls = task == "Classification"
    suffix = "cls" if is_cls else "reg"
    cfg = model.cls_cfg if is_cls else model.reg_cfg
    if device == "cpu":
        data = DATA / ("cls_cpu4" if is_cls else "reg_cpu4")
        save = f"gpu_cpu_check/cpu/{model.save_stem}_{suffix}"
    else:
        data = DATA / ("cls_gpu10" if is_cls else "reg_gpu10")
        save = f"gpu_cpu_check/gpu/{model.save_stem}_{suffix}"
    cmd = [
        sys.executable, "-m", "ldm_limix.infer",
        "--task_type", task,
        "--model_path", str(CKPT_DIR / model.ckpt),
        "--data_dir", str(data),
        "--inference_config_path", str(ROOT / cfg),
        "--device", device,
        "--autobatch",
        "--show_progress",
        "--seed", "0",
        "--save_name", save,
        "--preprocess_num_jobs", "4",
    ]
    if device == "cuda":
        cmd.extend(["--gpuid", str(gpuid)])
        if model.v2:
            cmd.extend(["--gpu_num_per_predictor", "1"])
    return cmd, RESULT_ROOT / "logs" / f"{device}_{model.save_stem}_{suffix}.log"


def run_one(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[start] {' '.join(cmd)}", flush=True)
    with log_path.open("w") as log:
        log.write("CMD: " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        return proc.wait()


def job_already_done(save_name: str, expected_n: int) -> bool:
    csv_path = ROOT / "result" / save_name / "all_rst.csv"
    if not csv_path.is_file():
        return False
    try:
        n = max(sum(1 for _ in csv_path.open()) - 1, 0)
    except OSError:
        return False
    return n >= expected_n


def run_gpu(gpus: list[int]) -> list[tuple[str, int]]:
    pending: list[tuple[ModelSpec, str, str]] = []
    statuses: list[tuple[str, int]] = []
    for model in MODELS:
        for task in ("Classification", "Regression"):
            suffix = "cls" if task == "Classification" else "reg"
            save = f"gpu_cpu_check/gpu/{model.save_stem}_{suffix}"
            label = f"{model.save_stem} {task}"
            expected = 10
            if job_already_done(save, expected):
                print(f"[skip] {label} already has {expected} GPU results", flush=True)
                statuses.append((label, 0))
            else:
                pending.append((model, task, label))

    inflight: list[tuple[subprocess.Popen, Path, str, int]] = []
    free = list(gpus)
    idx = 0
    while idx < len(pending) or inflight:
        while idx < len(pending) and free:
            model, task, label = pending[idx]
            idx += 1
            gpuid = free.pop(0)
            cmd, log = job_cmd(model, task, "cuda", gpuid)
            log.parent.mkdir(parents=True, exist_ok=True)
            print(f"[gpu {gpuid}] start {label}", flush=True)
            fh = log.open("w")
            fh.write("CMD: " + " ".join(cmd) + "\n\n")
            fh.flush()
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
            inflight.append((proc, log, label, gpuid))
        time.sleep(2)
        still = []
        for proc, log, label, gpuid in inflight:
            code = proc.poll()
            if code is None:
                still.append((proc, log, label, gpuid))
            else:
                print(f"[{'ok' if code == 0 else 'FAIL'}] {label} exit={code} log={log}", flush=True)
                statuses.append((label, code))
                free.append(gpuid)
        inflight = still
    return statuses


def run_cpu(parallel: int = 4) -> list[tuple[str, int]]:
    pending: list[tuple[list[str], Path, str]] = []
    statuses: list[tuple[str, int]] = []
    for model in MODELS:
        for task in ("Classification", "Regression"):
            suffix = "cls" if task == "Classification" else "reg"
            save = f"gpu_cpu_check/cpu/{model.save_stem}_{suffix}"
            label = f"CPU {model.save_stem} {task}"
            if job_already_done(save, 4):
                print(f"[skip] {label} already has 4 CPU results", flush=True)
                statuses.append((label, 0))
                continue
            cmd, log = job_cmd(model, task, "cpu", None)
            pending.append((cmd, log, label))

    inflight: list[tuple[subprocess.Popen, Path, str]] = []
    idx = 0
    while idx < len(pending) or inflight:
        while idx < len(pending) and len(inflight) < parallel:
            cmd, log, label = pending[idx]
            idx += 1
            log.parent.mkdir(parents=True, exist_ok=True)
            print(f"[start] {label}", flush=True)
            fh = log.open("w")
            fh.write("CMD: " + " ".join(cmd) + "\n\n")
            fh.flush()
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
            inflight.append((proc, log, label))
        time.sleep(3)
        still = []
        for proc, log, label in inflight:
            code = proc.poll()
            if code is None:
                still.append((proc, log, label))
            else:
                print(f"[{'ok' if code == 0 else 'FAIL'}] {label} exit={code} log={log}", flush=True)
                statuses.append((label, code))
        inflight = still
    return statuses


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["gpu", "cpu", "all"], default="all")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    missing = [m.ckpt for m in MODELS if not (CKPT_DIR / m.ckpt).is_file()]
    if missing:
        print("missing checkpoints:", missing, file=sys.stderr)
        return 1
    statuses = []
    if args.stage in ("gpu", "all"):
        statuses.extend(run_gpu(gpus))
    if args.stage in ("cpu", "all"):
        statuses.extend(run_cpu())
    failed = [s for s in statuses if s[1] != 0]
    print("\n=== summary ===")
    for label, code in statuses:
        print(f"{code:3d}  {label}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
