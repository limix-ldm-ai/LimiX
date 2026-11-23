"""Unified LimiX inference: version routing, GPU layout, workers, and eval.

Spawn workers import this module. CUDA allocator env must be set before torch.
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import gc
import pickle as pkl
import time
import zipfile
from functools import partial
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from sklearn.metrics import accuracy_score, f1_score, log_loss, r2_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

try:
    from sklearn.metrics import root_mean_squared_error as mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error

    mean_squared_error = partial(mean_squared_error, squared=False)

import ldm_limix.config as config_pkg
from ldm_limix.inference.predictor import LimiXPredictor
from ldm_limix.model.version import resolve_arch_line, resolve_arch_version
from ldm_limix.utils.inference_subprocess_v2 import (
    ModelInitError,
    enrich_worker_result,
    is_cuda_context_corrupted,
    is_cuda_related_error,
    is_fatal_cuda_error,
    parse_gpu_ids,
    run_parallel_inference,
    worker_exit_on_fatal,
)

CPU_DEVICE_SLOT = -1
from ldm_limix.utils.inference_utils import auc_metric, sample_inferece_params
from ldm_limix.utils.categorical_encoding import FEATURE_ENCODER_MODES, encode_categorical_features

try:
    import nvtx
except ImportError:
    from contextlib import nullcontext

    class _Nvtx:
        def annotate(self, *_args, **_kwargs):
            return nullcontext()

    nvtx = _Nvtx()


TASK_TYPE_ALIASES = {
    "classification": "Classification",
    "cls": "Classification",
    "class": "Classification",
    "regression": "Regression",
    "reg": "Regression",
    "feature_imputation": "Feature_imputation",
    "imputation": "Feature_imputation",
    "mvi": "Feature_imputation",
}

MASK_RATIO_CHOICES = ("0.05", "0.10", "0.15", "0.20", "0.25", "0.30")
_REGRESSION_DECODER_TYPES = ("mse", "bucket", "bucket_4_tabpfn")
_DEVICE_CHOICES = ("cuda", "cpu")

_SEARCH_SPACE_REPEAT = {
    "Classification": 2,
    "Regression": 4,
    "Feature_imputation": 4,
}


def data_dir_format_help(task_type: str, mask_ratio: str = "0.05") -> str:
    task_type = normalize_task_type(task_type)
    mask_line = ""
    if task_type == "Feature_imputation":
        mask_line = (
            f"    <dataset_name>_{mask_ratio}.csv    "
            "# required for Feature_imputation\n"
        )
    return (
        "--data_dir must be a benchmark root with one subdirectory per dataset:\n"
        "\n"
        "<data_dir>/\n"
        "  <dataset_name>/\n"
        "    <dataset_name>_train.csv\n"
        "    <dataset_name>_test.csv\n"
        f"{mask_line}"
        "\n"
        "Each CSV needs a header. The last column is the target; remaining columns are features.\n"
        "See https://huggingface.co/datasets/stableai-org/bcco_cls and "
        "https://huggingface.co/datasets/stableai-org/bcco_reg"
    )


def _required_dataset_csvs(dataset_name: str, task_type: str, mask_ratio: str) -> list[str]:
    names = [f"{dataset_name}_train.csv", f"{dataset_name}_test.csv"]
    if task_type == "Feature_imputation":
        names.append(f"{dataset_name}_{mask_ratio}.csv")
    return names


def _csv_has_feature_and_target(path: str) -> bool:
    try:
        frame = pd.read_csv(path, nrows=0)
    except Exception:
        return False
    return frame.shape[1] >= 2


def _exit_bad_data_dir(reason: str, task_type: str, mask_ratio: str) -> None:
    raise SystemExit(
        f"{reason}\n\nRequired dataset layout:\n{data_dir_format_help(task_type, mask_ratio)}"
    )


def normalize_task_type(task_type: str) -> str:
    if task_type in ("Classification", "Regression", "Feature_imputation"):
        return task_type
    mapped = TASK_TYPE_ALIASES.get(str(task_type).strip().lower())
    if mapped is None:
        raise ValueError(
            f"Unknown task_type={task_type!r}; expected Classification, "
            f"Regression, Feature_imputation (or cls/reg/imputation)"
        )
    return mapped


def default_inference_config_path(task_type: str, arch_line: str) -> str:
    task_type = normalize_task_type(task_type)
    if arch_line not in ("v1_0", "v2_0"):
        raise ValueError(f"Unsupported arch_line={arch_line!r}; expected v1_0 or v2_0")
    if task_type == "Classification":
        filename = (
            "cls_default_noretrieval.json"
            if arch_line == "v1_0"
            else "cls_default_noretrieval_v2.json"
        )
    elif task_type == "Regression":
        filename = (
            "reg_default_noretrieval.json"
            if arch_line == "v1_0"
            else "reg_default_noretrieval_v2.json"
        )
    else:
        filename = (
            "reg_default_noretrieval_MVI.json"
            if arch_line == "v1_0"
            else "reg_default_noretrieval_MVI_v2.json"
        )
    return str(Path(config_pkg.__file__).with_name(filename))


def default_data_repo(task_type: str) -> Tuple[str, str]:
    task_type = normalize_task_type(task_type)
    if task_type == "Regression":
        return "stableai-org/bcco_reg", "./cache/bcco_reg"
    return "stableai-org/bcco_cls", "./cache/bcco_cls"


def group_gpus_for_predictors(gpu_ids: Sequence[int], gpu_num_per_predictor: int) -> List[List[int]]:
    """Split visible GPU ids into equal groups, one group per outer predictor process."""
    if gpu_num_per_predictor < 1:
        raise ValueError(
            f"gpu_num_per_predictor must be >= 1, got {gpu_num_per_predictor}"
        )
    if len(gpu_ids) % gpu_num_per_predictor != 0:
        raise ValueError(
            f"number of GPU ids ({len(gpu_ids)}) must be divisible by "
            f"gpu_num_per_predictor ({gpu_num_per_predictor})"
        )
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"gpu ids must be unique, got {list(gpu_ids)}")
    return [
        list(gpu_ids[index:index + gpu_num_per_predictor])
        for index in range(0, len(gpu_ids), gpu_num_per_predictor)
    ]


def resolve_predictor_gpu_layout(
    gpu_ids: Sequence[int],
    gpu_num_per_predictor: int,
    arch_line: str,
) -> Tuple[List[List[int]], int]:
    """Group GPUs for outer workers. V1.0 has no inner ensemble, so each GPU is one worker."""
    effective = gpu_num_per_predictor
    if arch_line == "v1_0" and gpu_num_per_predictor != 1:
        print(
            f"Warning: V1.0 does not support ensemble multi-GPU "
            f"(gpu_num_per_predictor={gpu_num_per_predictor}); "
            f"using outer process scheduling with 1 GPU per predictor"
        )
        effective = 1
    groups = group_gpus_for_predictors(gpu_ids, effective)
    return groups, effective


def resolve_inference_device_layout(
    device: str,
    gpu_ids: Sequence[int],
    gpu_num_per_predictor: int,
    arch_line: str,
) -> Tuple[List[List[int]], int]:
    """Build worker slots. CPU uses a single sentinel group [[-1]]."""
    if str(device).lower() == "cpu":
        return [[CPU_DEVICE_SLOT]], 1
    return resolve_predictor_gpu_layout(gpu_ids, gpu_num_per_predictor, arch_line)


def is_cpu_gpu_group(gpu_group: Sequence[int] | None) -> bool:
    return not gpu_group or int(gpu_group[0]) < 0


def predictor_gpu_group(gpu_id: int, worker_config: dict) -> List[int]:
    groups = worker_config.get("gpu_groups") or {}
    return list(groups.get(gpu_id, [gpu_id]))


def _require_file(path, flag: str) -> str:
    if path is None or str(path).strip() == "":
        raise SystemExit(f"{flag} is required")
    if not os.path.isfile(path):
        raise SystemExit(f"{flag} is not an existing file: {path}")
    return os.path.abspath(path)


def _require_dir(path, flag: str) -> str:
    if path is None or str(path).strip() == "":
        raise SystemExit(f"{flag} is required")
    if not os.path.isdir(path):
        raise SystemExit(f"{flag} is not an existing directory: {path}")
    return os.path.abspath(path)


def _require_json_file(path, flag: str):
    path = _require_file(path, flag)
    try:
        with open(path, "r") as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{flag} is not valid JSON: {path}\n{exc}") from exc
    except OSError as exc:
        raise SystemExit(f"{flag} cannot be read: {path}\n{exc}") from exc
    return path, payload


def validate_inference_inputs(args) -> None:
    """Fail with SystemExit if CLI paths or numeric flags cannot run inference."""
    try:
        normalize_task_type(args.task_type)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    args.data_dir = _require_dir(args.data_dir, "--data_dir")
    args.model_path = _require_file(args.model_path, "--model_path")
    if getattr(args, "inference_config_path", None):
        args.inference_config_path, _ = _require_json_file(
            args.inference_config_path, "--inference_config_path",
        )

    device = str(getattr(args, "device", "cuda")).lower()
    if device not in _DEVICE_CHOICES:
        raise SystemExit(
            f"--device must be one of {_DEVICE_CHOICES}, got {device!r}"
        )
    decoder = getattr(args, "regression_decoder_type", "mse")
    if decoder not in _REGRESSION_DECODER_TYPES:
        raise SystemExit(
            f"--regression_decoder_type must be one of {_REGRESSION_DECODER_TYPES}, "
            f"got {decoder!r}"
        )
    encoder_mode = getattr(args, "feature_encoder_mode", "current")
    if encoder_mode not in FEATURE_ENCODER_MODES:
        raise SystemExit(
            f"--feature_encoder_mode must be one of {FEATURE_ENCODER_MODES}, "
            f"got {encoder_mode!r}"
        )
    mask_ratio = str(getattr(args, "mask_ratio", "0.05"))
    if mask_ratio not in MASK_RATIO_CHOICES:
        raise SystemExit(
            f"--mask_ratio must be one of {MASK_RATIO_CHOICES}, got {mask_ratio!r}"
        )

    gpu_num = int(getattr(args, "gpu_num_per_predictor", 1))
    if gpu_num < 1:
        raise SystemExit(f"--gpu_num_per_predictor must be >= 1, got {gpu_num}")
    if device != "cpu":
        gpu_ids = parse_gpu_ids(getattr(args, "gpuid", None))
        if any(int(gid) < 0 for gid in gpu_ids):
            raise SystemExit(f"--gpuid must be >= 0, got {list(gpu_ids)}")
        if len(set(gpu_ids)) != len(gpu_ids):
            raise SystemExit(f"--gpuid must be unique, got {list(gpu_ids)}")
        if len(gpu_ids) % gpu_num != 0:
            raise SystemExit(
                f"number of GPU ids ({len(gpu_ids)}) must be divisible by "
                f"--gpu_num_per_predictor ({gpu_num}), got --gpuid {list(gpu_ids)}"
            )
    jobs = int(getattr(args, "preprocess_num_jobs", 16))
    if jobs < 1:
        raise SystemExit(f"--preprocess_num_jobs must be >= 1, got {jobs}")
    search_n = int(getattr(args, "search_space_sample_num", 0))
    if search_n < 0:
        raise SystemExit(
            f"--search_space_sample_num must be >= 0, got {search_n}"
        )


def load_xy(data_path, y_as_float: bool = False):
    data = pd.read_csv(data_path)
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    if y_as_float:
        y = y.astype(float)
    return x, y


def build_inference_tasks(
    data_root: str,
    task_type: str,
    search_space_sample_num: int = 0,
    mask_ratio: str = "0.05",
) -> list:
    task_type = normalize_task_type(task_type)
    data_root = os.path.abspath(data_root)
    if not os.path.isdir(data_root):
        _exit_bad_data_dir(
            f"--data_dir is not a directory: {data_root}",
            task_type,
            mask_ratio,
        )

    own_name = os.path.basename(data_root.rstrip(os.sep))
    nested_csvs = _required_dataset_csvs(own_name, task_type, mask_ratio)
    if all(os.path.isfile(os.path.join(data_root, name)) for name in nested_csvs):
        _exit_bad_data_dir(
            f"--data_dir looks like a single dataset folder ({own_name}), "
            f"not a benchmark root. Pass the parent directory that contains "
            f"dataset subfolders.",
            task_type,
            mask_ratio,
        )

    benchmark_name = os.path.basename(data_root)
    rng = np.random.default_rng(42)
    tasks = []
    skipped = []
    entries = os.listdir(data_root)
    for idx, dataset_name in enumerate(entries):
        folder_path = os.path.join(data_root, dataset_name)
        if os.path.isfile(folder_path):
            continue
        required = _required_dataset_csvs(dataset_name, task_type, mask_ratio)
        missing = [
            name for name in required
            if not os.path.isfile(os.path.join(folder_path, name))
        ]
        if missing:
            skipped.append(f"{dataset_name} (missing {', '.join(missing)})")
            continue
        bad_csv = [
            name for name in required
            if not _csv_has_feature_and_target(os.path.join(folder_path, name))
        ]
        if bad_csv:
            skipped.append(
                f"{dataset_name} (CSV must have a header and at least two columns: "
                f"{', '.join(bad_csv)})"
            )
            continue
        sample_index = 0
        repeat_num = _SEARCH_SPACE_REPEAT[task_type]
        while sample_index == 0 or sample_index < search_space_sample_num:
            task = {
                "dataset_idx": idx,
                "dataset_name": dataset_name,
                "benchmark_name": benchmark_name,
                "sample_index": sample_index,
            }
            if search_space_sample_num > 0:
                if sample_index > 0:
                    hyperopt_config, base_config = sample_inferece_params(rng, 2, repeat_num)
                    task["hyperopt_config"] = hyperopt_config
                    task["base_config"] = base_config
                else:
                    task["use_default_search_config"] = True
            tasks.append(task)
            sample_index += 1

    if not tasks:
        detail = "No dataset subdirectories found."
        if skipped:
            shown = skipped[:20]
            extra = f" ... ({len(skipped) - 20} more)" if len(skipped) > 20 else ""
            detail = "No valid dataset found. Skipped: " + "; ".join(shown) + extra
        _exit_bad_data_dir(
            f"--data_dir does not match the required layout: {data_root}\n{detail}",
            task_type,
            mask_ratio,
        )
    if skipped:
        shown = skipped[:10]
        extra = f" ... ({len(skipped) - 10} more)" if len(skipped) > 10 else ""
        print(
            f"Warning: skipped {len(skipped)} folder(s) that do not match "
            f"the dataset layout: {'; '.join(shown)}{extra}"
        )
    return tasks


def compute_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    if y_prob.ndim == 2 and y_prob.shape[1] > 1:
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
    else:
        confidences = y_prob if y_prob.ndim == 1 else y_prob[:, 1]
        predictions = (confidences >= 0.5).astype(int)

    accuracies = (predictions == y_true)
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            avg_conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(acc_in_bin - avg_conf_in_bin) * prop_in_bin
    return ece


def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


_TENSOR_REBUILDERS = {
    "_rebuild_tensor",
    "_rebuild_tensor_v2",
    "_rebuild_parameter",
    "_rebuild_from_type_v2",
    "_rebuild_qtensor",
    "_rebuild_device_tensor_from_numpy",
    "_rebuild_device_tensor_from_cpu_tensor",
    "_load_from_bytes",
}


class _DroppedObject:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


def _drop(*args, **kwargs):
    return None


class _CheckpointMetaUnpickler(pkl.Unpickler):
    """Unpickle a torch checkpoint dict without reconstructing tensor storage."""

    def persistent_load(self, saved_id):
        return None

    def find_class(self, module, name):
        if name in _TENSOR_REBUILDERS or "Storage" in name:
            return _drop
        try:
            return super().find_class(module, name)
        except Exception:
            return _DroppedObject


def _load_ckpt_metadata(model_path: str):
    if zipfile.is_zipfile(model_path):
        with zipfile.ZipFile(model_path) as archive:
            try:
                pickle_name = next(
                    name for name in archive.namelist() if name.endswith("data.pkl")
                )
            except StopIteration as exc:
                raise ValueError(f"checkpoint zip has no data.pkl: {model_path}") from exc
            with archive.open(pickle_name) as fh:
                return _CheckpointMetaUnpickler(fh).load()
    with open(model_path, "rb") as fh:
        return _CheckpointMetaUnpickler(fh).load()


def peek_arch_line(model_path: str):
    ckpt = _load_ckpt_metadata(model_path)
    arch_version = resolve_arch_version(ckpt)
    return resolve_arch_line(arch_version), arch_version


def apply_autobatch_flag(arch_line: str, enabled: bool) -> None:
    if arch_line == "v1_0":
        from ldm_limix.model.v1_0.autobatch import AutobatchConfig
    elif arch_line == "v2_0":
        from ldm_limix.model.v2_0.autobatch import AutobatchConfig
    else:
        raise ValueError(f"Unsupported arch_line={arch_line!r}; expected v1_0 or v2_0")
    AutobatchConfig.ENABLE_AUTOBATCH = bool(enabled)


def _timed_predict(model, x_train, y_train, x_test, task_type, unique_dataset_name, benchmark_time):
    inference_time_ms = None
    max_memory_mb = None
    num_runs = 2 if benchmark_time else 1
    prediction = None
    device = model.device
    use_cuda = isinstance(device, torch.device) and device.type == "cuda"
    for i in range(num_runs):
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(device)
        t1 = time.perf_counter()
        prediction = model.predict(
            x_train, y_train, x_test,
            task_type=task_type,
            unique_dataset_name=unique_dataset_name,
        )
        t2 = time.perf_counter()
        max_allocate_memory = torch.cuda.max_memory_allocated(device) if use_cuda else 0
        if i == num_runs - 1:
            inference_time_ms = (t2 - t1) * 1000
            max_memory_mb = max_allocate_memory / 1024 / 1024
    return prediction, inference_time_ms, max_memory_mb


def inference_classification_v10(
    classifier, le, scaler, X_train, y_train, X_test, y_test,
    unique_dataset_name=None, benchmark_time=False, feature_encoder_mode="current",
):
    X_train, X_test = encode_categorical_features(
        X_train, X_test, feature_encoder_mode=feature_encoder_mode,
    )
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    num_classes = len(le.classes_)

    trainX = np.asarray(X_train, dtype=np.float32)
    trainy = np.asarray(y_train, dtype=np.int64)
    if len(np.unique(trainy)) > 10 or len(np.unique(trainy)) < 2:
        return None, None, None, f"num_classes {len(np.unique(trainy))} is not supported"

    testX = np.asarray(X_test, dtype=np.float32)
    testy = np.asarray(y_test, dtype=np.int64)
    prediction_, inference_time_ms, max_memory_mb = _timed_predict(
        classifier, trainX, trainy, testX, "Classification",
        unique_dataset_name, benchmark_time,
    )

    prediction_label = np.argmax(prediction_, axis=1)
    roc = auc_metric(testy, prediction_)
    acc = accuracy_score(testy, prediction_label)
    f1 = f1_score(testy, prediction_label, average="macro" if num_classes > 2 else "binary")
    ce = log_loss(testy, prediction_)
    ece = compute_ece(testy, prediction_, n_bins=10)
    rst = {
        "num_data_train": len(trainX),
        "num_data_test": len(testX),
        "num_feat": len(trainX[0]),
        "num_class": len(np.unique(trainy)),
        "acc": float(acc),
        "f1": float(f1),
        "logloss": float(ce),
        "ece": float(ece),
        "auc": float(roc),
        "inference_time_ms": inference_time_ms,
        "max_memory_mb": max_memory_mb,
    }
    return rst, prediction_, testy, "success"


def inference_classification_native(
    classifier, X_train, y_train, X_test, y_test,
    unique_dataset_name=None, benchmark_time=False,
):
    try:
        prediction_, inference_time_ms, max_memory_mb = _timed_predict(
            classifier, X_train, y_train, X_test, "Classification",
            unique_dataset_name, benchmark_time,
        )
    except ValueError as e:
        msg = str(e)
        if "num_classes" in msg and "is not supported" in msg:
            return None, None, None, msg
        raise

    testy = classifier.label_encoder.transform(y_test)
    testy = np.asarray(testy, dtype=np.int64)
    num_classes = prediction_.shape[1]
    prediction_label = np.argmax(prediction_, axis=1)
    roc = auc_metric(testy, prediction_)
    acc = accuracy_score(testy, prediction_label)
    f1 = f1_score(testy, prediction_label, average="macro" if num_classes > 2 else "binary")
    ce = log_loss(testy, prediction_)
    ece = compute_ece(testy, prediction_, n_bins=10)
    rst = {
        "num_data_train": len(X_train),
        "num_data_test": len(X_test),
        "num_feat": X_train.shape[1],
        "num_class": num_classes,
        "acc": float(acc),
        "f1": float(f1),
        "logloss": float(ce),
        "ece": float(ece),
        "auc": float(roc),
        "inference_time_ms": inference_time_ms,
        "max_memory_mb": max_memory_mb,
    }
    return rst, prediction_, testy, "success"


def inference_regression_v10(
    X_train, X_test, y_train, y_test, model,
    unique_dataset_name=None, benchmark_time=False,
):
    sample_size, feature_count = X_train.shape
    rmse_results = {"Sample_Size": sample_size, "Feature_Count": feature_count}
    r2_results = {}

    y_mean = y_train.mean()
    y_std = y_train.std()
    if y_std == 0:
        y_std = 1.0
    y_train_normalized = (y_train - y_mean) / y_std
    y_test_normalized = (y_test - y_mean) / y_std

    y_pred, inference_time_ms, max_memory_mb = _timed_predict(
        model, X_train, y_train_normalized, X_test, "Regression",
        unique_dataset_name, benchmark_time,
    )
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.to("cpu")
    rmse = mean_squared_error(y_test_normalized, y_pred)
    rmse_raw = rmse * float(y_std)
    r2 = r2_score(y_test_normalized, y_pred)
    r2_results["R2"] = r2
    rmse_results["rmse"] = rmse
    rmse_results["rmse_raw"] = rmse_raw
    rmse_results["inference_time_ms"] = inference_time_ms
    rmse_results["max_memory_mb"] = max_memory_mb
    pred_result = {"label": y_test, "pred": y_pred * y_std + y_mean}
    return rmse_results, r2_results, pred_result


def inference_regression_native(
    X_train, X_test, y_train, y_test, model,
    unique_dataset_name=None, benchmark_time=False,
):
    sample_size, feature_count = X_train.shape
    rmse_results = {"Sample_Size": sample_size, "Feature_Count": feature_count}
    r2_results = {}
    y_pred, inference_time_ms, max_memory_mb = _timed_predict(
        model, X_train, y_train, X_test, "Regression",
        unique_dataset_name, benchmark_time,
    )
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.to("cpu")
    y_pred = np.asarray(y_pred)
    y_mean = model.y_mean
    y_std = model.y_std
    y_test_normalized = (np.asarray(y_test, dtype=np.float64) - y_mean) / y_std
    y_pred_normalized = (y_pred - y_mean) / y_std
    rmse = mean_squared_error(y_test_normalized, y_pred_normalized)
    rmse_raw = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    r2_results["R2"] = r2
    rmse_results["rmse"] = rmse
    rmse_results["rmse_raw"] = rmse_raw
    rmse_results["inference_time_ms"] = inference_time_ms
    rmse_results["max_memory_mb"] = max_memory_mb
    pred_result = {"label": y_test, "pred": y_pred}
    return rmse_results, r2_results, pred_result


def get_categorical_features_indices(x_):
    min_unique_num_for_numerical_infer = 10
    categories = {}
    x = x_.values if isinstance(x_, pd.DataFrame) else x_
    for idx, col in enumerate(x.T):
        is_integer = np.isclose(col, np.floor(col), atol=1e-6)
        if not is_integer.all():
            continue
        if len(np.unique(col)) < min_unique_num_for_numerical_infer:
            categories[idx] = np.unique(col)
    return categories


def mask_prediction_eval(x_pred_, x_true_, mask, categories):
    x_pred = x_pred_.copy()
    x_true = x_true_.copy()

    total_reg_rmse_list = []
    total_reg_r2_list = []
    for idx in np.arange(x_pred.shape[1]):
        mask_col = mask[:, idx]
        if not np.any(mask_col):
            continue
        cur_col_mask_pred = x_pred[:, idx][mask_col]
        cur_col_mask_true = x_true[:, idx][mask_col]
        total_reg_rmse_list.append(mean_squared_error(cur_col_mask_pred, cur_col_mask_true))
        if len(np.unique(cur_col_mask_true)) > 1:
            total_reg_r2_list.append(r2_score(cur_col_mask_true, cur_col_mask_pred))
        else:
            true_val = cur_col_mask_true[0]
            total_reg_r2_list.append(1.0 if np.allclose(cur_col_mask_pred, true_val) else 0.0)

    total_rmse = 0.0
    total_r2 = 0.0
    if total_reg_rmse_list:
        total_rmse = np.mean(total_reg_rmse_list)
        total_r2 = np.mean(total_reg_r2_list)

    categorical_idx = list(categories.keys())
    for idx in categorical_idx:
        distances = np.abs(x_pred[:, idx][:, np.newaxis] - categories[idx])
        nearest_indices = np.argmin(distances, axis=1)
        x_pred[:, idx] = categories[idx][nearest_indices]

    cls_acc_list = []
    cls_f1_list = []
    for idx in categorical_idx:
        mask_col = mask[:, idx]
        if not np.any(mask_col):
            continue
        cur_col_mask_pred = x_pred[:, idx][mask_col].astype(int)
        cur_col_mask_true = x_true[:, idx][mask_col].astype(int)
        cls_acc_list.append(accuracy_score(cur_col_mask_true, cur_col_mask_pred))
        if len(np.unique(cur_col_mask_true)) > 1:
            cls_f1_list.append(f1_score(cur_col_mask_true, cur_col_mask_pred, average="macro"))
        else:
            cls_f1_list.append(1.0)

    cls_auc = 0.0
    cls_acc = 0.0
    cls_f1 = 0.0
    if cls_acc_list:
        cls_acc = np.mean(cls_acc_list)
        cls_f1 = np.mean(cls_f1_list)

    reg_rmse_list = []
    reg_r2_list = []
    for idx in np.setdiff1d(np.arange(x_pred.shape[1]), categorical_idx):
        mask_col = mask[:, idx]
        if not np.any(mask_col):
            continue
        cur_col_mask_pred = x_pred[:, idx][mask_col]
        cur_col_mask_true = x_true[:, idx][mask_col]
        reg_rmse_list.append(mean_squared_error(cur_col_mask_pred, cur_col_mask_true))
        if len(np.unique(cur_col_mask_true)) > 1:
            reg_r2_list.append(r2_score(cur_col_mask_true, cur_col_mask_pred))
        else:
            true_val = cur_col_mask_true[0]
            reg_r2_list.append(1.0 if np.allclose(cur_col_mask_pred, true_val) else 0.0)

    reg_rmse = 0.0
    reg_r2 = 0.0
    if reg_rmse_list:
        reg_rmse = np.mean(reg_rmse_list)
        reg_r2 = np.mean(reg_r2_list)

    if cls_acc_list and reg_rmse_list:
        data_set_pingce_type = 2
    elif not cls_acc_list and reg_rmse_list:
        data_set_pingce_type = 1
    elif cls_acc_list and not reg_rmse_list:
        data_set_pingce_type = 0
    else:
        data_set_pingce_type = 2

    return {
        "cls_auc": float(cls_auc),
        "cls_acc": float(cls_acc),
        "cls_f1": float(cls_f1),
        "reg_rmse": float(reg_rmse),
        "reg_r2": float(reg_r2),
        "total_rmse": float(total_rmse),
        "total_r2": float(total_r2),
        "data_set_pingce_type": data_set_pingce_type,
    }


def inference_imputation_dataset(X_train, X_test, y_train, y_test, X_test_mask, predictor):
    scaler = MinMaxScaler()
    for col in X_train.columns:
        if X_train[col].dtype == "object":
            try:
                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col])
                X_test[col] = le.transform(X_test[col])
            except Exception:
                X_train = X_train.drop(columns=[col])
                X_test = X_test.drop(columns=[col])
                X_test_mask = X_test_mask.drop(columns=[col])

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int32)

    sample_size, feature_count = X_train.shape
    eva_results = {"Sample_Size": sample_size, "Feature_Count": feature_count}
    feature_categories = get_categorical_features_indices(X_train)
    X_test_mask = X_test_mask.to_numpy()
    testX_original = X_test.copy()
    masked_testX = X_test.copy()
    masked_testX[X_test_mask] = np.nan

    prediction_ = predictor.predict(X_train, y_train, masked_testX, task_type="Feature_imputation")
    mask_prediction_ = prediction_[-X_test.shape[0]:]
    eva_result = mask_prediction_eval(mask_prediction_, testX_original, X_test_mask, feature_categories)
    eva_results["cls_auc"] = eva_result["cls_auc"]
    eva_results["cls_acc"] = eva_result["cls_acc"]
    eva_results["cls_f1"] = eva_result["cls_f1"]
    eva_results["reg_rmse"] = eva_result["reg_rmse"]
    eva_results["reg_r2"] = eva_result["reg_r2"]
    eva_results["total_rmse"] = eva_result["reg_rmse"]
    eva_results["total_r2"] = eva_result["reg_r2"]
    eva_results["data_set_pingce_type"] = eva_result["data_set_pingce_type"]
    pred_result = {"label": testX_original, "pred": mask_prediction_}
    return eva_results, pred_result


def _build_predictor(worker_config, inference_config, gpu_group):
    if is_cpu_gpu_group(gpu_group):
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{gpu_group[0]}")
    kwargs = dict(
        device=device,
        model_path=worker_config["model_path"],
        inference_config=inference_config,
        inference_with_DDP=worker_config["inference_with_DDP"],
        seed=worker_config["seed"],
        deterministic=worker_config["deterministic"],
        regression_decoder_type=worker_config["regression_decoder_type"],
        mix_precision=worker_config.get("mix_precision", True),
        use_data_cache=worker_config.get("use_data_cache", False),
    )
    if worker_config["arch_line"] != "v1_0":
        kwargs["enable_preprocess_parallel"] = worker_config["enable_preprocess_parallel"]
        kwargs["preprocess_num_jobs"] = worker_config["preprocess_num_jobs"]
        if (not is_cpu_gpu_group(gpu_group)) and len(gpu_group) >= 2:
            kwargs["gpu_ids"] = gpu_group
    return LimiXPredictor(**kwargs)


def _maybe_set_search_config(model, inference_config, task, search_space_sample_num):
    if search_space_sample_num <= 0:
        return
    sample_index = task["sample_index"]
    if sample_index > 0:
        model.set_inference_config(
            inference_config=task["hyperopt_config"],
            **task["base_config"],
        )
        print(f"{sample_index}/{search_space_sample_num}", end="\r")
    else:
        model.set_inference_config(inference_config, 0.9, 0)


def _save_cls_pred(save_root, dataset_name, prediction_, testy):
    output_df = {"label": testy}
    for i in range(prediction_.shape[1]):
        output_df[f"pred_{i}"] = prediction_[:, i]
    pd.DataFrame(output_df).to_csv(
        os.path.join(save_root, dataset_name + "_pred_LimiX.csv"),
        index=False,
    )


def _handle_classification_task(task, model, worker_config, inference_config, cls_state, gpu_group):
    dataset_name = task["dataset_name"]
    sample_index = task["sample_index"]
    dataset_idx = task["dataset_idx"]
    data_root = worker_config["data_root"]
    with nvtx.annotate("data-load"):
        X_train, y_train = load_xy(os.path.join(data_root, dataset_name, dataset_name + "_train.csv"))
        X_test, y_test = load_xy(os.path.join(data_root, dataset_name, dataset_name + "_test.csv"))
    unique_dataset_name = f"{task['benchmark_name']}_{dataset_name}"
    _maybe_set_search_config(model, inference_config, task, worker_config["search_space_sample_num"])

    if worker_config["arch_line"] == "v1_0":
        rst, prediction_, testy, info = inference_classification_v10(
            model, cls_state["le"], cls_state["scaler"],
            X_train.copy(), y_train.copy(), X_test.copy(), y_test.copy(),
            unique_dataset_name=unique_dataset_name,
            benchmark_time=worker_config["benchmark_time"],
            feature_encoder_mode=worker_config["feature_encoder_mode"],
        )
    else:
        rst, prediction_, testy, info = inference_classification_native(
            model,
            X_train.copy(), y_train.copy(), X_test.copy(), y_test.copy(),
            unique_dataset_name=unique_dataset_name,
            benchmark_time=worker_config["benchmark_time"],
        )
    assert rst is not None, f"Error processing {dataset_name} with sample_index {sample_index}: {info}"
    rst["dataset name"] = dataset_name
    rst["search_space_sample_index"] = sample_index
    _save_cls_pred(worker_config["save_root"], rst["dataset name"], prediction_, testy)
    del prediction_
    if worker_config["print_result"] or worker_config["debug"]:
        print(f"[{dataset_idx}] GPU{gpu_group} {dataset_name} -> {rst['auc']}")
    return {"status": "success", "rst": rst}


def _handle_regression_task(task, model, worker_config, inference_config, gpu_group):
    dataset_name = task["dataset_name"]
    sample_index = task["sample_index"]
    dataset_idx = task["dataset_idx"]
    data_root = worker_config["data_root"]
    train_data_path = Path(data_root, dataset_name, f"{dataset_name}_train.csv")
    test_data_path = Path(data_root, dataset_name, f"{dataset_name}_test.csv")
    X_train, y_train = load_xy(train_data_path, y_as_float=True)
    X_test, y_test = load_xy(test_data_path, y_as_float=True)
    unique_dataset_name = f"{task['benchmark_name']}_{dataset_name}"
    _maybe_set_search_config(model, inference_config, task, worker_config["search_space_sample_num"])

    infer_fn = inference_regression_v10 if worker_config["arch_line"] == "v1_0" else inference_regression_native
    rmse_result, r2_result, pred_result = infer_fn(
        X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy(),
        model,
        benchmark_time=worker_config["benchmark_time"],
        unique_dataset_name=unique_dataset_name,
    )
    rst = {
        "dataset name": dataset_name,
        "num_data_train": len(X_train),
        "num_data_test": len(X_test),
        "num_feat": X_train.shape[1],
        "num_class": len(np.unique(y_train)),
    }
    rst.update(**rmse_result)
    rst.update(**r2_result)
    rst["search_space_sample_index"] = sample_index
    pd.DataFrame(pred_result).to_csv(
        os.path.join(worker_config["save_root"], rst["dataset name"] + "_pred_LimiX.csv"),
        index=False,
    )
    del pred_result
    if worker_config["print_result"] or worker_config["debug"]:
        print(
            f"[{dataset_idx}] GPU{gpu_group} {dataset_name} -> "
            f"{rst['R2']}, {rst['rmse']}, {rst['rmse_raw']}"
        )
    return {"status": "success", "rst": rst}


def _handle_imputation_task(task, model, worker_config, inference_config, gpu_group):
    dataset_name = task["dataset_name"]
    sample_index = task["sample_index"]
    dataset_idx = task["dataset_idx"]
    data_root = worker_config["data_root"]
    mask_ratio = worker_config["mask_ratio"]
    X_train, y_train = load_xy(Path(data_root, dataset_name, f"{dataset_name}_train.csv"), y_as_float=True)
    X_test, y_test = load_xy(Path(data_root, dataset_name, f"{dataset_name}_test.csv"), y_as_float=True)
    X_test_mask, _ = load_xy(Path(data_root, dataset_name, f"{dataset_name}_{mask_ratio}.csv"), y_as_float=True)
    _maybe_set_search_config(model, inference_config, task, worker_config["search_space_sample_num"])
    eva_results, pred_result = inference_imputation_dataset(
        X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy(),
        X_test_mask.copy(), model,
    )
    rst = {
        "dataset name": dataset_name,
        "num_data_train": len(X_train),
        "num_data_test": len(X_test),
        "num_feat": X_train.shape[1],
        "num_class": len(np.unique(y_train)),
    }
    rst.update(**eva_results)
    rst["search_space_sample_index"] = sample_index
    with open(os.path.join(worker_config["save_root"], rst["dataset name"] + "_pred_LimiX.pkl"), "wb") as f:
        pkl.dump(pred_result, f)
    if worker_config["print_result"] or worker_config["debug"]:
        print(f"[{dataset_idx}] GPU{gpu_group} {dataset_name} -> {eva_results}")
    return {"status": "success", "rst": rst}


def inference_worker(gpu_id, conn, worker_config):
    gpu_group = predictor_gpu_group(gpu_id, worker_config)
    if not is_cpu_gpu_group(gpu_group):
        torch.cuda.set_device(gpu_group[0])
    apply_autobatch_flag(worker_config["arch_line"], worker_config["autobatch"])
    debug = worker_config["debug"]
    debug_main_tasks = worker_config.get("debug_main_tasks")
    task_type = worker_config["task_type"]

    try:
        with open(worker_config["inference_config_path"], "r") as f:
            inference_config = json.load(f)
        model = _build_predictor(worker_config, inference_config, gpu_group)
    except BaseException as e:
        if debug:
            raise
        worker_exit_on_fatal(conn, gpu_id, None, e, phase="model_init")
        return

    cls_state = None
    if task_type == "Classification" and worker_config["arch_line"] == "v1_0":
        cls_state = {"scaler": MinMaxScaler(), "le": LabelEncoder()}

    def handle_task(task):
        dataset_name = task["dataset_name"]
        sample_index = task["sample_index"]
        try:
            if task_type == "Classification":
                return _handle_classification_task(
                    task, model, worker_config, inference_config, cls_state, gpu_group,
                )
            if task_type == "Regression":
                return _handle_regression_task(
                    task, model, worker_config, inference_config, gpu_group,
                )
            return _handle_imputation_task(
                task, model, worker_config, inference_config, gpu_group,
            )
        except Exception as e:
            if debug:
                raise
            msg = str(e)
            if is_fatal_cuda_error(e):
                worker_exit_on_fatal(conn, gpu_id, task, e)
            elif is_cuda_related_error(e) and is_cuda_context_corrupted():
                worker_exit_on_fatal(conn, gpu_id, task, e)
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as cache_err:
                worker_exit_on_fatal(conn, gpu_id, task, cache_err)
            brief = msg.strip().splitlines()[-1][:240] if msg.strip() else msg
            print(
                f"Worker Process Warning: GPU{gpu_group} Error processing {dataset_name} "
                f"with sample_index {sample_index}: {brief}"
            )
            return {"status": "error", "rst": None, "error": msg}
        finally:
            if task_type == "Classification":
                gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                if debug_main_tasks is not None or debug:
                    raise
                worker_exit_on_fatal(conn, gpu_id, task, e)

    if debug:
        results = []
        for task in debug_main_tasks:
            result = handle_task(task)
            if result is not None:
                results.append(enrich_worker_result(gpu_id, task, result))
        return results

    conn.send({"event": "ready"})
    while True:
        command = conn.recv()
        if command.get("command") == "shutdown":
            return
        result = handle_task(command["task"])
        if result is not None:
            conn.send(result)


def _write_results(save_root, worker_results):
    if int(os.environ.get("WORLD_SIZE", -1)) > 0 and get_rank() != 0:
        return
    rsts = []
    for msg in sorted(worker_results, key=lambda x: (x["dataset_idx"], x["sample_index"])):
        if msg["status"] != "success":
            continue
        rsts.append(msg["rst"])
    pd.DataFrame(rsts).to_csv(os.path.join(save_root, "all_rst.csv"), index=False)


def run_unified_inference(args):
    validate_inference_inputs(args)
    device = str(getattr(args, "device", "cuda")).lower()

    task_type = normalize_task_type(args.task_type)
    model_file = args.model_path
    data_root = args.data_dir
    search_space_sample_num = args.search_space_sample_num

    tasks = build_inference_tasks(
        data_root,
        task_type,
        search_space_sample_num=search_space_sample_num,
        mask_ratio=getattr(args, "mask_ratio", "0.05"),
    )

    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Use --device cpu or enable GPU.")

    try:
        arch_line, arch_version = peek_arch_line(model_file)
    except Exception as exc:
        raise SystemExit(
            f"--model_path is not a valid checkpoint: {model_file}\n{exc}"
        ) from exc
    print(f"Detected ckpt arch_version={arch_version} -> {arch_line}")

    inference_config_path = args.inference_config_path
    if not inference_config_path:
        inference_config_path = default_inference_config_path(task_type, arch_line)
        args.inference_config_path = inference_config_path
    inference_config_path, inference_config = _require_json_file(
        inference_config_path, "--inference_config_path",
    )
    args.inference_config_path = inference_config_path

    gpu_ids = parse_gpu_ids(args.gpuid)
    if device == "cuda":
        n_visible = torch.cuda.device_count()
        bad = [gid for gid in gpu_ids if gid < 0 or gid >= n_visible]
        if bad:
            raise SystemExit(
                f"--gpuid {bad} out of range; {n_visible} GPU(s) visible"
            )
    try:
        gpu_groups, gpu_num_per_predictor = resolve_inference_device_layout(
            device, gpu_ids, args.gpu_num_per_predictor, arch_line,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    slot_ids = [group[0] for group in gpu_groups]
    apply_autobatch_flag(arch_line, args.autobatch)

    if args.save_name is None:
        args.save_name = time.strftime("%Y%m%d-%H%M%S")
    save_root = f"./result/{args.save_name}"
    os.makedirs(save_root, exist_ok=True)

    with open(os.path.join(save_root, "config.json"), "w") as f:
        json.dump(inference_config, f)

    mix_precision = not bool(getattr(args, "disable_mix_precision", False))
    worker_config = {
        "model_path": model_file,
        "inference_config_path": inference_config_path,
        "inference_with_DDP": args.inference_with_DDP,
        "regression_decoder_type": args.regression_decoder_type,
        "save_root": save_root,
        "debug": args.debug,
        "deterministic": args.deterministic,
        "seed": args.seed,
        "benchmark_time": args.benchmark_time,
        "search_space_sample_num": search_space_sample_num,
        "data_root": data_root,
        "print_result": args.print_result,
        "show_progress": args.show_progress or args.debug,
        "autobatch": args.autobatch,
        "enable_preprocess_parallel": args.enable_preprocess_parallel,
        "preprocess_num_jobs": args.preprocess_num_jobs,
        "gpu_groups": {group[0]: group for group in gpu_groups},
        "init_timeout_sec": 120 * max(gpu_num_per_predictor, 1),
        "task_timeout_sec": (
            int(os.environ["LDM_INFER_TASK_TIMEOUT_SEC"])
            if os.environ.get("LDM_INFER_TASK_TIMEOUT_SEC")
            else None
        ),
        "arch_line": arch_line,
        "task_type": task_type,
        "feature_encoder_mode": args.feature_encoder_mode,
        "mask_ratio": args.mask_ratio,
        "mix_precision": mix_precision,
        "use_data_cache": bool(getattr(args, "use_data_cache", False)),
    }
    device_desc = "cpu" if device == "cpu" else f"{gpu_num_per_predictor} GPU(s) per predictor, GPU groups {gpu_groups}"
    print(
        f"Running inference: task={task_type} arch={arch_line} device={device} "
        f"{len(gpu_groups)} predictor(s) total, {device_desc}, {len(tasks)} task(s)"
    )
    if args.debug:
        worker_config["debug_main_tasks"] = tasks
        worker_results = inference_worker(slot_ids[0], None, worker_config) or []
        _write_results(save_root, worker_results)
        return save_root

    try:
        worker_results = run_parallel_inference(
            slot_ids, tasks, inference_worker, worker_config,
        )
    except ModelInitError as exc:
        raise SystemExit(f"Model initialization failed: {exc}") from exc
    _write_results(save_root, worker_results)
    return save_root
