"""Pipe-based GPU worker orchestration for inference.

Each GPU runs one long-lived worker process. The parent dispatches exactly one
task at a time per GPU and tracks the in-flight task locally (no Manager/inflight).
Task results use status: success | error | fatal.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Literal, Optional, Set, Tuple

FATAL_CUDA_MARKERS = (
    'cuda error: an illegal memory access',
    'cuda error: device-side assert',
    'cuda error: misaligned address',
    'cuda error: invalid device function',
    'cuda error: invalid configuration argument',
    'cannot be accessed from triton',
    'pointer argument',
)

DEFAULT_TASK_TIMEOUT_SEC = None
DEFAULT_INIT_TIMEOUT_SEC = 240
POLL_INTERVAL_SEC = 0.5

# Project root (parent of utils/) — workers need this on PYTHONPATH for spawn imports.
_DEFAULT_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

WaitPhase = Literal['init', 'restart']


def _prepend_pythonpath(*paths: str) -> None:
    """Ensure paths are on PYTHONPATH so spawn children can import project modules."""
    existing = os.environ.get('PYTHONPATH', '')
    parts = [p for p in existing.split(os.pathsep) if p]
    for path in reversed(paths):
        abs_path = os.path.abspath(path)
        if abs_path not in parts:
            parts.insert(0, abs_path)
    os.environ['PYTHONPATH'] = os.pathsep.join(parts)


def _resolve_project_root(worker_config: Dict[str, Any]) -> str:
    root = worker_config.get('project_root')
    if root:
        return os.path.abspath(root)
    return _DEFAULT_PROJECT_ROOT


class ModelInitError(RuntimeError):
    """Raised when a worker fails to load the model."""


def parse_gpu_ids(gpu_id_args: Optional[List[int]]) -> List[int]:
    if not gpu_id_args:
        return [0]
    return list(gpu_id_args)


def task_id(task: Dict[str, Any]) -> Tuple[int, int]:
    return (task['dataset_idx'], task.get('sample_index', 0))


def is_fatal_cuda_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in FATAL_CUDA_MARKERS)


def is_cuda_related_error(exc: BaseException) -> bool:
    if is_fatal_cuda_error(exc):
        return True
    msg = str(exc).lower()
    if 'out of memory' in msg and 'cuda' in msg:
        return True
    cuda_hints = ('cuda', 'triton', 'cudnn', 'cublas', 'device-side assert')
    return any(hint in msg for hint in cuda_hints)


def is_cuda_context_corrupted() -> bool:
    import torch
    if not torch.cuda.is_available():
        return False
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return False
    except Exception:
        return True


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h{minutes:02d}m"


def _print_inference_progress(
    completed: int,
    total: int,
    elapsed: float,
) -> None:
    pct = (completed / total) * 100 if total > 0 else 0.0
    eta = (elapsed / completed) * (total - completed) if completed > 0 else 0.0
    bar_width = 30
    filled = int(bar_width * completed / total) if total > 0 else 0
    bar = '█' * filled + '-' * (bar_width - filled)
    print(
        f"\rProgress [{bar}] {completed}/{total} ({pct:.1f}%) | "
        f"elapsed: {_format_duration(elapsed)} | "
        f"ETA: {_format_duration(eta)}",
        end='',
        flush=True,
    )
    if completed >= total:
        print()


def worker_exit_on_fatal(
    conn: Any,
    gpu_id: int,
    task: Optional[Dict[str, Any]],
    exc: BaseException,
    phase: str = 'task',
) -> None:
    print(
        f"Worker Process Fatal Error: GPU[{gpu_id}] phase={phase} "
        f"fatal error: {str(exc)[:200]}"
    )
    message = {
        'status': 'fatal',
        'phase': phase,
        'error': repr(exc),
        'traceback': traceback.format_exc(),
    }
    try:
        conn.send(message)
    except (BrokenPipeError, EOFError, OSError):
        pass
    try:
        conn.close()
    except Exception:
        pass
    # Parent enriches task metadata and settles on the fatal message above.
    os._exit(70)


def enrich_worker_result(
    gpu_id: int,
    task: Dict[str, Any],
    worker_msg: Dict[str, Any],
) -> Dict[str, Any]:
    """Attach scheduler-owned task metadata to a minimal worker payload."""
    dataset_name = task['dataset_name']
    result = dict(worker_msg)
    result.update({
        'gpu_id': gpu_id,
        'task_id': task_id(task),
        'task': task,
        'dataset_idx': task['dataset_idx'],
        'sample_index': task.get('sample_index', 0),
        'dataset_name': dataset_name,
    })
    return result


@dataclass
class _WorkerSlot:
    process: Optional[mp.Process] = None
    conn: Any = None
    task: Optional[Dict[str, Any]] = None
    task_deadline: Optional[float] = None
    bootstrapping: bool = True


class _InferenceScheduler:
    def __init__(
        self,
        ctx: mp.context.BaseContext,
        gpu_ids: List[int],
        tasks: List[Dict[str, Any]],
        worker_fn: Callable[..., None],
        worker_config: Dict[str, Any],
    ) -> None:
        self.ctx = ctx
        self.gpu_ids = gpu_ids
        self.worker_fn = worker_fn
        self.worker_config = worker_config
        self.pending: Deque[Dict[str, Any]] = deque(tasks)
        self.results: List[Dict[str, Any]] = []
        self.settled_ids: Set[Tuple[int, int]] = set()
        self.workers: Dict[int, _WorkerSlot] = {
            gpu_id: _WorkerSlot() for gpu_id in gpu_ids
        }
        self.total_tasks = len(tasks)
        self.task_timeout_sec = worker_config.get('task_timeout_sec', DEFAULT_TASK_TIMEOUT_SEC)
        self.init_timeout_sec = worker_config.get('init_timeout_sec', DEFAULT_INIT_TIMEOUT_SEC)
        self.show_progress = worker_config.get('show_progress', True)
        self.start_time = time.perf_counter()

    def run(self) -> List[Dict[str, Any]]:
        if not self.pending:
            return []

        try:
            for gpu_id in self.gpu_ids:
                self._start_worker(gpu_id)
            self._wait_workers_ready(self.gpu_ids, phase='init')
            self._dispatch_idle_workers()

            while len(self.settled_ids) < self.total_tasks:
                self._check_timeouts()
                self._poll_events()
                if not self._has_outstanding_work():
                    break
                self._dispatch_idle_workers()

            return self.results
        except (KeyboardInterrupt, SystemExit):
            raise
        finally:
            self._shutdown_all_workers()

    def _has_outstanding_work(self) -> bool:
        if self.pending:
            return True
        return any(slot.task is not None for slot in self.workers.values())

    def _slot_ready(self, slot: _WorkerSlot) -> bool:
        return not slot.bootstrapping

    def _clear_task(self, gpu_id: int) -> None:
        slot = self.workers[gpu_id]
        slot.task = None
        slot.task_deadline = None

    def _start_worker(self, gpu_id: int) -> None:
        parent_conn, child_conn = self.ctx.Pipe(duplex=True)
        proc = self.ctx.Process(
            target=self.worker_fn,
            args=(gpu_id, child_conn, self.worker_config),
            daemon=False,
        )
        proc.start()
        child_conn.close()
        slot = self.workers[gpu_id]
        slot.process = proc
        slot.conn = parent_conn
        slot.task = None
        slot.task_deadline = None
        slot.bootstrapping = True

    def _raise_model_init_failure(self, error: str) -> None:
        raise ModelInitError(error)

    def _wait_workers_ready(
        self,
        gpu_ids: List[int],
        *,
        phase: WaitPhase = 'init',
    ) -> None:
        deadline = time.monotonic() + self.init_timeout_sec
        waiting = set(gpu_ids)

        while waiting:
            if time.monotonic() > deadline:
                if phase == 'restart' and len(waiting) == 1:
                    gpu_id = next(iter(waiting))
                    self._raise_model_init_failure(
                        f"GPU {gpu_id} worker restart timed out after "
                        f"{self.init_timeout_sec}s"
                    )
                self._raise_model_init_failure(
                    f"Worker init timed out after {self.init_timeout_sec}s "
                    f"for GPU(s): {sorted(waiting)}"
                )

            for gpu_id in list(waiting):
                slot = self.workers[gpu_id]
                if self._poll_init_message(gpu_id, waiting):
                    continue
                if slot.process is not None and not slot.process.is_alive():
                    exitcode = slot.process.exitcode
                    if phase == 'restart' and len(waiting) == 1:
                        self._raise_model_init_failure(
                            f"GPU {gpu_id} worker exited during restart init "
                            f"(exitcode={exitcode})"
                        )
                    self._raise_model_init_failure(
                        f"GPU {gpu_id} worker exited during model init "
                        f"(exitcode={exitcode})"
                    )

            if waiting:
                time.sleep(POLL_INTERVAL_SEC)

    def _poll_init_message(
        self,
        gpu_id: int,
        waiting: Set[int],
    ) -> bool:
        slot = self.workers[gpu_id]
        if slot.conn is None or not slot.conn.poll(0):
            return False
        try:
            msg = slot.conn.recv()
        except EOFError:
            return False
        if msg.get('event') == 'ready':
            slot.bootstrapping = False
            waiting.discard(gpu_id)
            return True
        if msg.get('status') == 'fatal' and msg.get('phase') == 'model_init':
            self._raise_model_init_failure(msg.get('error', 'model init failed'))
        self._raise_model_init_failure(
            f"GPU {gpu_id} sent unexpected message during init: {msg}"
        )

    def _dispatch_idle_workers(self) -> None:
        for gpu_id in self.gpu_ids:
            slot = self.workers[gpu_id]
            if not self._slot_ready(slot) or slot.task is not None or not self.pending:
                continue
            task = self.pending.popleft()
            slot.task = task
            slot.task_deadline = (
                None if self.task_timeout_sec is None
                else time.monotonic() + self.task_timeout_sec
            )
            slot.conn.send({
                'command': 'run',
                'task': task,
            })

    def _poll_events(self) -> None:
        for gpu_id, slot in list(self.workers.items()):
            if slot.conn is not None and slot.conn.poll(POLL_INTERVAL_SEC):
                self._handle_conn_message(gpu_id)
            if slot.process is not None and not slot.process.is_alive():
                self._handle_process_exit(gpu_id)

    def _handle_conn_message(self, gpu_id: int) -> None:
        slot = self.workers[gpu_id]
        if slot.conn is None:
            return

        while slot.conn.poll():
            try:
                msg = slot.conn.recv()
            except EOFError:
                self._handle_process_exit(gpu_id)
                return
            if msg.get('event') == 'ready':
                slot.bootstrapping = False
                continue

            status = msg.get('status')
            if status == 'fatal' and msg.get('phase') == 'model_init':
                self._raise_model_init_failure(msg.get('error', 'model init failed'))

            current = slot.task
            if current is not None:
                expected = task_id(current)
                msg_tid = msg.get('task_id')
                if msg_tid is not None and msg_tid != expected:
                    continue

            if status in ('success', 'error'):
                self._settle(gpu_id, worker_msg=msg)
            elif status == 'fatal':
                self._settle(gpu_id, worker_msg=msg)
                self._restart_worker_after_fatal(gpu_id)
            else:
                raise ValueError(f"Unknown worker message status: {status}")

    def _handle_process_exit(self, gpu_id: int) -> None:
        slot = self.workers[gpu_id]
        if slot.process is None or slot.process.is_alive():
            return

        exitcode = slot.process.exitcode
        if slot.task is None and slot.bootstrapping:
            self._raise_model_init_failure(
                f"GPU {gpu_id} worker exited during model init (exitcode={exitcode})"
            )

        if slot.task is None:
            return

        if task_id(slot.task) in self.settled_ids:
            self._clear_task(gpu_id)
            return

        self._settle(
            gpu_id,
            status='fatal',
            error=f"Worker exited unexpectedly (exitcode={exitcode})",
        )
        self._restart_worker_after_fatal(gpu_id)

    def _check_timeouts(self) -> None:
        now = time.monotonic()
        for gpu_id, slot in self.workers.items():
            if slot.task is None or slot.task_deadline is None:
                continue
            if now <= slot.task_deadline:
                continue
            self._terminate_worker(gpu_id)
            self._settle(
                gpu_id,
                status='fatal',
                error=f"Task timed out after {self.task_timeout_sec}s",
            )
            self._restart_worker_after_fatal(gpu_id)

    def _enrich_result(
        self,
        gpu_id: int,
        task: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        result = enrich_worker_result(gpu_id, task, payload)
        if 'pid' not in result and payload.get('status') == 'fatal':
            slot = self.workers[gpu_id]
            result.setdefault('pid', slot.process.pid if slot.process else None)
        return result

    def _settle(
        self,
        gpu_id: int,
        *,
        status: Optional[str] = None,
        worker_msg: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        phase: str = 'task',
    ) -> bool:
        slot = self.workers[gpu_id]
        task = slot.task
        if task is None:
            return False

        tid = task_id(task)
        if tid in self.settled_ids:
            self._clear_task(gpu_id)
            return False

        if worker_msg is not None:
            payload = dict(worker_msg)
        else:
            payload = {
                'status': status or 'fatal',
                'phase': phase,
                'error': error or '',
                'traceback': '',
            }

        self.settled_ids.add(tid)
        self.results.append(self._enrich_result(gpu_id, task, payload))
        status_now = payload.get('status')
        if status_now in ('error', 'fatal'):
            err = str(payload.get('error') or '')
            brief = err.strip().splitlines()[-1][:400] if err.strip() else err
            print(
                f"Task failed GPU{gpu_id} {task.get('dataset_name')}: "
                f"status={status_now} {brief}"
            )
        self._clear_task(gpu_id)
        self._report_progress()
        return True

    def _restart_worker_after_fatal(self, gpu_id: int) -> None:
        self._terminate_worker(gpu_id)
        self._start_worker(gpu_id)
        self._wait_workers_ready([gpu_id], phase='restart')

    def _terminate_worker(self, gpu_id: int) -> None:
        slot = self.workers[gpu_id]
        if slot.conn is not None:
            try:
                slot.conn.send({'command': 'shutdown'})
            except (BrokenPipeError, EOFError, OSError):
                pass
            try:
                slot.conn.close()
            except Exception:
                pass
            slot.conn = None

        if slot.process is not None:
            if slot.process.is_alive():
                slot.process.terminate()
                slot.process.join(timeout=10)
                if slot.process.is_alive():
                    slot.process.kill()
                    slot.process.join(timeout=5)
            slot.process = None

        slot.bootstrapping = True

    def _shutdown_all_workers(self) -> None:
        for gpu_id in list(self.workers.keys()):
            self._terminate_worker(gpu_id)

    def _report_progress(self) -> None:
        if not self.show_progress:
            return
        elapsed = time.perf_counter() - self.start_time
        _print_inference_progress(len(self.settled_ids), self.total_tasks, elapsed)


def run_parallel_inference(
    gpu_ids: List[int],
    tasks: List[Dict[str, Any]],
    worker_fn: Callable[..., None],
    worker_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Dispatch tasks to one long-lived worker per GPU via Pipe."""
    if not tasks:
        return []

    project_root = _resolve_project_root(worker_config)
    _prepend_pythonpath(project_root)

    ctx = mp.get_context('spawn')
    scheduler = _InferenceScheduler(ctx, gpu_ids, tasks, worker_fn, worker_config)
    return scheduler.run()
