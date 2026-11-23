"""Spawn one worker process per GPU to run ensemble pipelines.

Each collect puts pipeline indices on a shared queue (longest-estimated first).
Idle workers pull one index, run that member, then pull again. The parent
broadcasts payload once, gathers per-member outputs, and performs ensemble.
CUDA_VISIBLE_DEVICES is set in the parent immediately before Process.start()
so spawn children inherit it before re-importing __main__.
"""
from __future__ import annotations

import atexit
import multiprocessing as mp
import os
import traceback
from queue import Empty
from typing import Any, Iterable

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def warn_unsupported_predictor_kwargs(kwargs: dict) -> None:
    """Functionality: Print a warning listing constructor kwargs that this predictor version ignores.

    Input:
        kwargs: Extra keyword arguments not consumed as named parameters.

    Output:
        None.
    """
    if kwargs:
        print(
            "WARNING: ignoring unsupported predictor kwargs: "
            f"{sorted(kwargs)}"
        )


def assign_pipelines_to_gpus(n_pipelines: int, gpu_ids: list[int]) -> list[int]:
    """Functionality: Round-robin assign each pipeline index to a GPU id.

    Input:
        n_pipelines: Ensemble member count.
        gpu_ids: GPU ids used for pipeline parallel.

    Output:
        list[int]: gpu id for pipeline 0..n_pipelines-1.
    """
    if n_pipelines < 0:
        raise ValueError("n_pipelines must be non-negative")
    if not gpu_ids:
        raise ValueError("gpu_ids must not be empty")
    n_gpus = len(gpu_ids)
    return [gpu_ids[index % n_gpus] for index in range(n_pipelines)]


def pipelines_by_gpu(n_pipelines: int, gpu_ids: list[int]) -> dict[int, list[int]]:
    """Functionality: Group pipeline indices by the GPU that should run them.

    Input:
        n_pipelines: Ensemble member count.
        gpu_ids: GPU ids used for pipeline parallel.

    Output:
        dict[int, list[int]]: gpu id -> pipeline indices, omitting idle GPUs.
    """
    assignment: dict[int, list[int]] = {}
    for pipeline_index, gpu_id in enumerate(
        assign_pipelines_to_gpus(n_pipelines, gpu_ids)
    ):
        assignment.setdefault(gpu_id, []).append(pipeline_index)
    return assignment


def pipeline_work_score(pipeline_config: dict | None) -> int:
    """Functionality: Cheap static cost for one ensemble member, used only to order the steal queue.

    Input:
        pipeline_config: One pipeline dict from inference_config, or None.

    Output:
        int: higher means likely heavier. Fingerprint, polynomial cap, SVD width, datetime.
    """
    config = pipeline_config or {}
    score = 0
    if config.get("FingerprintFeatureEncoder"):
        score += 100
    poly = config.get("PolynomialInteractionGenerator") or {}
    if isinstance(poly, dict):
        score += int(poly.get("max_interaction_features") or 0)
    rebalance = config.get("RebalanceFeatureDistribution") or {}
    if isinstance(rebalance, dict) and rebalance.get("svd_tag") == "svd":
        score += int(rebalance.get("svd_max_components") or 0)
    datetime_config = config.get("DatetimePreprocessing") or {}
    if isinstance(datetime_config, dict) and datetime_config.get("enabled"):
        score += 10
    return score


def order_pipelines_longest_first(pipeline_configs: Iterable) -> list[int]:
    """Functionality: Pipeline indices sorted by estimated work, ties broken by original index.

    Input:
        pipeline_configs: Ensemble member configs in pipeline-index order.

    Output:
        list[int]: longest-first order for the shared steal queue.
    """
    configs = list(pipeline_configs)
    return sorted(
        range(len(configs)),
        key=lambda index: (-pipeline_work_score(configs[index]), index),
    )


def iter_pipeline_task_queue(task_queue) -> Iterable[int]:
    """Functionality: Yield pipeline indices until a None sentinel.

    Input:
        task_queue: multiprocessing queue of int indices, closed by one None per worker.

    Output:
        iterator of pipeline indices. One get() per yield so a worker cannot drain the queue
        without running the corresponding member.
    """
    while True:
        item = task_queue.get()
        if item is None:
            break
        yield int(item)


def _drain_mp_queue(task_queue) -> None:
    """Functionality: Drop leftover steal-queue items so the next collect starts empty.

    Input:
        task_queue: multiprocessing queue.

    Output:
        None.
    """
    while True:
        try:
            task_queue.get_nowait()
        except Empty:
            break


_RUNTIME_AUDIT_LIST_KEYS = (
    "svd_adaptation_audit",
    "svd_runtime_retry_audit",
    "polynomial_interaction_runtime_retry_audit",
    "cuda_pipeline_fallback_audit",
)


def merge_pipeline_runtime_audits(parts: list[dict] | None) -> dict:
    """Functionality: Merge per-worker SVD / retry / skip audits into one parent-side record.

    Input:
        parts: One audit dict per worker, or None.

    Output:
        dict with concatenated audit lists, any retry_cap, and the max call_index.
        Lists are sorted by pipeline_index so the parent snapshot matches single-GPU order.
    """
    merged = {key: [] for key in _RUNTIME_AUDIT_LIST_KEYS}
    merged["retry_cap"] = None
    merged["call_index"] = 0
    for part in parts or ():
        if not part:
            continue
        for key in _RUNTIME_AUDIT_LIST_KEYS:
            merged[key].extend(part.get(key) or [])
        retry_cap = part.get("retry_cap")
        if retry_cap is not None:
            merged["retry_cap"] = retry_cap
        merged["call_index"] = max(
            int(merged["call_index"] or 0),
            int(part.get("call_index") or 0),
        )
    for key in _RUNTIME_AUDIT_LIST_KEYS:
        merged[key].sort(
            key=lambda record: (
                int((record or {}).get("pipeline_index", -1)),
                int((record or {}).get("pipeline_step_index", -1)),
                int((record or {}).get("inference_call_index", 0)),
            )
        )
    return merged


def warn_uneven_pipeline_gpus(n_pipelines: int, gpu_ids: list[int]) -> None:
    """Functionality: Print a warning when pipeline count is not divisible by GPU count.

    Input:
        n_pipelines: Ensemble member count.
        gpu_ids: GPU ids used for pipeline parallel.

    Output:
        None.
    """
    if not gpu_ids:
        return
    n_gpus = len(gpu_ids)
    if n_pipelines % n_gpus == 0:
        return
    mapping = assign_pipelines_to_gpus(n_pipelines, gpu_ids)
    print(
        f"WARNING: pipeline count {n_pipelines} is not divisible by gpu count "
        f"{n_gpus}; assignment will be uneven: {mapping}"
    )


def normalize_pipeline_gpu_ids(gpu_ids) -> list[int] | None:
    """Functionality: Validate gpu_ids. Empty becomes None. Duplicates and negatives are errors.

    Input:
        gpu_ids: None, or a sequence of GPU indices.

    Output:
        list[int] | None. When CUDA is visible, indices must be in range.
    """
    if gpu_ids is None:
        return None
    gpu_list = list(gpu_ids)
    if len(gpu_list) == 0:
        return None
    normalized: list[int] = []
    for gpu_id in gpu_list:
        if isinstance(gpu_id, bool) or not isinstance(gpu_id, int):
            raise TypeError(f"gpu_ids must contain integers, got {gpu_id!r}")
        if gpu_id < 0:
            raise ValueError(f"gpu_ids must be non-negative, got {gpu_id}")
        normalized.append(int(gpu_id))
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"gpu_ids must be unique, got {normalized}")
    try:
        import torch
    except ImportError:
        return normalized
    if torch.cuda.is_available():
        visible = torch.cuda.device_count()
        for gpu_id in normalized:
            if gpu_id >= visible:
                raise ValueError(
                    f"gpu id {gpu_id} is out of range for {visible} visible CUDA "
                    "device(s)"
                )
    return normalized


def _prepend_pythonpath(root: str) -> None:
    existing = os.environ.get("PYTHONPATH", "")
    parts = [part for part in existing.split(os.pathsep) if part]
    abs_root = os.path.abspath(root)
    if abs_root not in parts:
        parts.insert(0, abs_root)
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)


def cuda_visible_devices_for_worker(gpu_id: int, visible: str | None = None) -> str:
    """Functionality: Map a parent-visible GPU index onto a CUDA_VISIBLE_DEVICES value.

    Input:
        gpu_id: GPU index in the parent's current visible-device list.
        visible: Parent CUDA_VISIBLE_DEVICES, or None to read the environment.

    Output:
        str: Token to expose as the child's only visible CUDA device.
    """
    if visible is None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or visible.strip() == "":
        return str(gpu_id)
    parts = [part.strip() for part in visible.split(",") if part.strip() != ""]
    if gpu_id < 0 or gpu_id >= len(parts):
        raise ValueError(
            f"gpu id {gpu_id} is out of range for CUDA_VISIBLE_DEVICES={visible!r}"
        )
    return parts[gpu_id]


def _start_process_with_visible_gpu(process: mp.Process, gpu_id: int) -> None:
    """Functionality: Spawn a worker with CUDA_VISIBLE_DEVICES set before the child interpreter starts.

    Spawn re-imports the parent's __main__, which usually imports torch and
    initializes CUDA. Setting the env var inside the child target is then too
    late, so it must be in the inherited environment at Process.start().

    Input:
        process: Spawn process that has not been started.
        gpu_id: Parent-visible GPU index this worker should own.

    Output:
        None. Restores the parent's CUDA_VISIBLE_DEVICES after start().
    """
    env_key = "CUDA_VISIBLE_DEVICES"
    previous = os.environ.get(env_key)
    os.environ[env_key] = cuda_visible_devices_for_worker(gpu_id, previous)
    try:
        process.start()
    finally:
        if previous is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = previous


def _worker_torch_device(torch_module, gpu_id: int):
    """Functionality: Pick the worker CUDA device after optional CUDA_VISIBLE_DEVICES remapping.

    Input:
        torch_module: Imported torch module.
        gpu_id: Parent-visible GPU index assigned to this worker.

    Output:
        torch.device: cuda:0 when remapping hid other GPUs, otherwise cuda:gpu_id.
    """
    if torch_module.cuda.is_available() and torch_module.cuda.device_count() > 1:
        device = torch_module.device("cuda", int(gpu_id))
        torch_module.cuda.set_device(device)
        return device
    return torch_module.device("cuda", 0)


def pipeline_gpu_worker_main(
    gpu_id: int,
    conn,
    init_kwargs: dict[str, Any],
    task_queue,
) -> None:
    """Functionality: Child process entry: own one GPU, load a predictor, steal pipeline indices.

    Input:
        gpu_id: GPU index as seen by the parent.
        conn: Spawn Pipe connection to the parent.
        init_kwargs: Constructor kwargs for a gpu_ids=None predictor.
        task_queue: Shared queue of pipeline indices for the current collect.

    Output:
        None. Talks to the parent over conn until close or the process exits.
    """
    _prepend_pythonpath(_PROJECT_ROOT)
    try:
        import torch

        from .predictor import LimiXPredictor
        from ldm_limix.model.v2_0.autobatch import AutobatchConfig

        kwargs = dict(init_kwargs)
        kwargs["device"] = _worker_torch_device(torch, gpu_id)
        kwargs["gpu_ids"] = None
        kwargs.pop("ckpt", None)
        enable_autobatch = kwargs.pop("_enable_autobatch", None)
        if enable_autobatch is not None:
            AutobatchConfig.ENABLE_AUTOBATCH = bool(enable_autobatch)
        predictor = LimiXPredictor(**kwargs)
        predictor._pipeline_indices = []
        device = predictor.device
        device_uuid = None
        if torch.cuda.is_available() and getattr(device, "type", None) == "cuda":
            device_uuid = str(torch.cuda.get_device_properties(device).uuid)
        conn.send(
            {
                "status": "ready",
                "gpu_id": gpu_id,
                "device": str(device),
                "device_uuid": device_uuid,
                "device_count": (
                    torch.cuda.device_count() if torch.cuda.is_available() else 0
                ),
            }
        )
        while True:
            message = conn.recv()
            command = message.get("cmd")
            if command == "close":
                conn.send({"status": "closed"})
                break
            if command == "set_inference_config":
                try:
                    predictor.set_inference_config(
                        inference_config=message["inference_config"],
                        softmax_temperature=message.get("softmax_temperature"),
                        seed=message.get("seed"),
                    )
                    conn.send({"status": "ok"})
                except Exception as error:
                    conn.send(
                        {
                            "status": "error",
                            "error": repr(error),
                            "traceback": traceback.format_exc(),
                        }
                    )
                continue
            if command != "collect":
                conn.send(
                    {
                        "status": "error",
                        "error": f"unknown command {command!r}",
                    }
                )
                continue
            try:
                conn.send({"status": "ready_for_tasks"})
                predictor._pipeline_index_source = (
                    lambda queue=task_queue: iter_pipeline_task_queue(queue)
                )
                try:
                    result = predictor._worker_collect_members(
                        task_type=message["task_type"],
                        x_train=message.get("x_train"),
                        y_train=message.get("y_train"),
                        x_test=message.get("x_test"),
                        unique_dataset_name=message.get("unique_dataset_name"),
                        svd_retry_state=message.get("svd_retry_state"),
                        prepared=message.get("prepared"),
                        member_inputs=message.get("member_inputs"),
                    )
                finally:
                    predictor._pipeline_index_source = None
                conn.send({"status": "ok", "result": result})
            except Exception as error:
                predictor._pipeline_index_source = None
                conn.send(
                    {
                        "status": "error",
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                    }
                )
    except Exception as error:
        try:
            conn.send(
                {
                    "status": "fatal",
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
            )
        except Exception:
            pass
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass


class PipelineGpuWorkerPool:
    """Functionality: Long-lived spawn workers, one GPU each, that steal pipeline indices.

    Input:
        gpu_ids: GPU ids to spawn. Every listed GPU gets a worker.
        worker_init_kwargs: Predictor constructor kwargs (gpu_ids must be omitted or None).
        n_pipelines: Ensemble size; submit() enqueues this many indices unless overridden.

    Output:
        Pool. submit() returns merged per-member results for the parent to ensemble.
    """

    def __init__(
        self,
        gpu_ids: list[int],
        worker_init_kwargs: dict[str, Any],
        n_pipelines: int,
    ):
        if not gpu_ids:
            raise ValueError("gpu_ids must not be empty")
        if n_pipelines <= 0:
            raise ValueError("no pipelines assigned to any GPU")
        self.gpu_ids = list(gpu_ids)
        self.n_pipelines = int(n_pipelines)
        self._ctx = mp.get_context("spawn")
        self._task_queue = self._ctx.Queue()
        self._conns: dict[int, Any] = {}
        self._processes: dict[int, mp.Process] = {}
        self._device_uuids: dict[int, str] = {}
        self._closed = False
        atexit.register(self.close)
        for gpu_id in self.gpu_ids:
            parent_conn, child_conn = self._ctx.Pipe()
            process = self._ctx.Process(
                target=pipeline_gpu_worker_main,
                args=(
                    gpu_id,
                    child_conn,
                    worker_init_kwargs,
                    self._task_queue,
                ),
            )
            _start_process_with_visible_gpu(process, gpu_id)
            child_conn.close()
            self._conns[gpu_id] = parent_conn
            self._processes[gpu_id] = process
            ready = parent_conn.recv()
            if ready.get("status") != "ready":
                self.close()
                raise RuntimeError(
                    f"pipeline GPU worker on gpu {gpu_id} failed to start: {ready}"
                )
            device_uuid = ready.get("device_uuid")
            if device_uuid is not None:
                for other_gpu_id, other_uuid in self._device_uuids.items():
                    if other_uuid == device_uuid:
                        print(
                            "WARNING: pipeline GPU workers for gpu "
                            f"{other_gpu_id} and {gpu_id} bound to the same "
                            f"device {device_uuid}"
                        )
                        break
                self._device_uuids[gpu_id] = device_uuid

    def set_inference_config(
        self,
        inference_config: dict,
        softmax_temperature: float | None = None,
        seed: int | None = None,
        n_pipelines: int | None = None,
    ) -> None:
        """Functionality: Rebuild preprocess pipelines on live workers without respawning them.

        Input:
            inference_config: Current parent v2 config dict.
            softmax_temperature: Temperature to apply on workers.
            seed: Seed used when workers rebuild shufflers.
            n_pipelines: New ensemble size; omitted keeps the current steal-queue length.

        Output:
            None. Raises if the pool is closed or a worker fails the update.
        """
        if self._closed:
            raise RuntimeError("pipeline GPU worker pool is closed")
        message = {
            "cmd": "set_inference_config",
            "inference_config": inference_config,
            "softmax_temperature": softmax_temperature,
            "seed": seed,
        }
        for conn in self._conns.values():
            conn.send(message)
        first_error = None
        for gpu_id, conn in self._conns.items():
            reply = conn.recv()
            if reply.get("status") != "ok" and first_error is None:
                first_error = (gpu_id, reply)
        if first_error is not None:
            gpu_id, reply = first_error
            detail = reply.get("traceback") or reply.get("error") or reply
            raise RuntimeError(
                f"pipeline GPU worker on gpu {gpu_id} failed to set_inference_config: "
                f"{detail}"
            )
        if n_pipelines is not None:
            if int(n_pipelines) <= 0:
                raise ValueError("n_pipelines must be positive")
            self.n_pipelines = int(n_pipelines)

    def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Functionality: Broadcast collect payload, enqueue pipeline indices, merge member outputs.

        Input:
            payload: Must include task_type. Optional pipeline_order overrides longest-first
                enqueue order. Remaining fields are task data for workers.

        Output:
            dict with members (sorted by pipeline index) plus optional extras such as
            target_transforms from the highest pipeline index that produced them.
        """
        if self._closed:
            raise RuntimeError("pipeline GPU worker pool is closed")
        message = dict(payload)
        pipeline_order = list(
            message.pop("pipeline_order", None) or range(self.n_pipelines)
        )
        message["cmd"] = "collect"
        _drain_mp_queue(self._task_queue)
        for conn in self._conns.values():
            conn.send(message)
        for gpu_id, conn in self._conns.items():
            ack = conn.recv()
            if ack.get("status") != "ready_for_tasks":
                detail = ack.get("traceback") or ack.get("error") or ack
                raise RuntimeError(
                    f"pipeline GPU worker on gpu {gpu_id} failed before steal: {detail}"
                )
        for index in pipeline_order:
            self._task_queue.put(int(index))
        for _ in self.gpu_ids:
            self._task_queue.put(None)
        members: list[tuple[int, Any]] = []
        target_transforms = None
        target_transforms_pipeline_index = -1
        last_failure = None
        worker_profiles: dict[int, Any] = {}
        runtime_audits: list[dict] = []
        first_error = None
        for gpu_id, conn in self._conns.items():
            reply = conn.recv()
            if reply.get("status") != "ok":
                if first_error is None:
                    first_error = (gpu_id, reply)
                continue
            result = reply["result"]
            members.extend(result.get("members") or [])
            worker_index = result.get("target_transforms_pipeline_index", -1)
            if worker_index is not None and int(worker_index) > target_transforms_pipeline_index:
                target_transforms_pipeline_index = int(worker_index)
                target_transforms = result.get("target_transforms")
            if result.get("last_failure") is not None:
                last_failure = result["last_failure"]
            worker_profiles[gpu_id] = result.get("profile")
            runtime_audits.append(result.get("runtime_audit") or {})
        _drain_mp_queue(self._task_queue)
        if first_error is not None:
            gpu_id, reply = first_error
            detail = reply.get("traceback") or reply.get("error")
            raise RuntimeError(
                f"pipeline GPU worker on gpu {gpu_id} failed: {detail}"
            )
        members.sort(key=lambda item: item[0])
        return {
            "members": members,
            "target_transforms": target_transforms,
            "target_transforms_pipeline_index": target_transforms_pipeline_index,
            "last_failure": last_failure,
            "worker_profiles": worker_profiles,
            "runtime_audit": merge_pipeline_runtime_audits(runtime_audits),
        }

    def close(self) -> None:
        """Functionality: Ask workers to exit and join the processes.

        Input:
            self.

        Output:
            None.
        """
        if self._closed:
            return
        self._closed = True
        for _ in self.gpu_ids:
            try:
                self._task_queue.put_nowait(None)
            except Exception:
                pass
        for gpu_id, conn in list(self._conns.items()):
            process = self._processes.get(gpu_id)
            try:
                if process is not None and process.is_alive():
                    conn.send({"cmd": "close"})
                    try:
                        conn.recv()
                    except EOFError:
                        pass
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
            if process is not None:
                process.join(timeout=30)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=10)
        _drain_mp_queue(self._task_queue)
        self._conns.clear()
        self._processes.clear()
