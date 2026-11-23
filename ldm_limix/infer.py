#!/usr/bin/env python3
"""Unified LimiX inference entry.

Routes by --task_type and checkpoint arch_version (V1.0 / V2.0).
V2.0 keeps inner ensemble multi-GPU via --gpu_num_per_predictor.
V1.0 only uses outer process scheduling (one GPU per worker).

Do not name this file inference.py: that would shadow the ldm_limix.inference package.
"""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse

from ldm_limix.utils.inference_unified import (
    MASK_RATIO_CHOICES,
    normalize_task_type,
    run_unified_inference,
)
from ldm_limix.utils.categorical_encoding import FEATURE_ENCODER_MODES

_REGRESSION_DECODER_TYPES = ("mse", "bucket", "bucket_4_tabpfn")


def _existing_file(path: str) -> str:
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"file not found: {path}")
    return path


def _existing_dir(path: str) -> str:
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"directory not found: {path}")
    return path


def _task_type(value: str) -> str:
    try:
        return normalize_task_type(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _nonneg_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid int: {value}") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {parsed}")
    return parsed


def _positive_int(value: str) -> int:
    parsed = _nonneg_int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {parsed}")
    return parsed


def _gpu_id(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid GPU id: {value}") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"GPU id must be >= 0, got {parsed}")
    return parsed


def build_parser():
    """Build the CLI parser for unified LimiX inference.

    Input:
        None. Flags are defined on the returned parser; callers pass sys.argv
        via parse_args(). Required flags: --task_type, --data_dir, --model_path.
        --task_type must be Classification, Regression, Feature_imputation, or
        the aliases cls / reg / imputation.

    Output:
        argparse.ArgumentParser whose namespace is consumed by
        run_unified_inference.
    """
    parser = argparse.ArgumentParser(description="Run LimiX inference")
    parser.add_argument("--task_type", type=_task_type, required=True, help="Classification | Regression | Feature_imputation (aliases: cls, reg, imputation)")
    parser.add_argument("--data_dir", type=_existing_dir, required=True, help="Specify the local storage directory of the dataset")
    parser.add_argument("--model_path", type=_existing_file, required=True, help="path to you model")
    parser.add_argument("--save_name", default=None, type=str, help="folder name to save result")
    parser.add_argument("--inference_config_path", type=_existing_file, default=None, help="path to example config; default depends on task_type and ckpt version")
    parser.add_argument("--inference_with_DDP", default=False, action="store_true", help="Inference with DDP")
    parser.add_argument("--debug", default=False, action="store_true", help="debug mode")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--deterministic", default=False, action="store_true", help="deterministic mode")
    parser.add_argument("--search_space_sample_num", type=_nonneg_int, default=0, help="number of samples to search in the search space")
    parser.add_argument("--autobatch", default=False, action="store_true", help="Enable autobatch")
    parser.add_argument("--benchmark_time", default=False, action="store_true", help="Run inference twice (warmup + benchmark); default runs once")
    parser.add_argument("--regression_decoder_type", type=str, choices=_REGRESSION_DECODER_TYPES, default="mse", help="regression decoder type")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda", help="Inference device. cpu skips CUDA and runs the predictor on host.")
    parser.add_argument("--gpuid", type=_gpu_id, nargs="+", default=None, help="GPU id(s) to use for inference, default 0. Ignored when --device cpu.")
    parser.add_argument("--gpu_num_per_predictor", type=_positive_int, default=1, help="GPUs given to each predictor for inner ensemble parallel. Outer worker count is len(gpuid) / gpu_num_per_predictor. Ignored for V1.0 (always 1 GPU per outer worker).")
    parser.add_argument("--disable_mix_precision", default=False, action="store_true", help="Disable autocast mixed precision (CPU already disables it)")
    parser.add_argument("--print_result", default=False, action="store_true", help="print result mode")
    parser.add_argument("--show_progress", default=False, action="store_true", help="show progress mode")
    parser.add_argument("--disable_preprocess_parallel", action="store_false", dest="enable_preprocess_parallel", default=True)
    parser.add_argument("--preprocess_num_jobs", type=_positive_int, default=16, help="preprocess (QTx) with num_jobs CPUs")
    parser.add_argument("--feature_encoder_mode", type=str, choices=FEATURE_ENCODER_MODES, default="current", help="feature encoder mode")
    parser.add_argument("--mask_ratio", type=str, choices=MASK_RATIO_CHOICES, default="0.05", help="feature mask ratio (Feature_imputation only)")
    parser.add_argument("--use_data_cache", default=False, action="store_true", help="cache preprocess results on disk")
    return parser


def main():
    args = build_parser().parse_args()
    run_unified_inference(args)


if __name__ == "__main__":
    main()
