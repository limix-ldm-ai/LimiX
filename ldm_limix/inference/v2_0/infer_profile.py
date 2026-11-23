"""Optional stage timers for one predict() call. Enable with LDM_INFER_PROFILE=1."""
from __future__ import annotations

import os
import time
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar

_ENABLED = os.environ.get("LDM_INFER_PROFILE") == "1"
_round: ContextVar[str] = ContextVar("infer_profile_round", default="")
_acc: dict[str, float] = defaultdict(float)
_count: dict[str, int] = defaultdict(int)


def enabled() -> bool:
    return _ENABLED


def reset() -> None:
    _acc.clear()
    _count.clear()


def snapshot() -> dict[str, float]:
    return dict(_acc)


def counts() -> dict[str, int]:
    return dict(_count)


def _key(name: str) -> str:
    round_name = _round.get()
    return f"{round_name}.{name}" if round_name else name


def add(name: str, ms: float) -> None:
    if not _ENABLED:
        return
    key = _key(name)
    _acc[key] += float(ms)
    _count[key] += 1


@contextmanager
def round_name(name: str):
    token = _round.set(name)
    try:
        yield
    finally:
        _round.reset(token)


@contextmanager
def span(name: str, cuda: bool = False):
    if not _ENABLED:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        add(name, (time.perf_counter() - start) * 1000.0)


def merge_worker(gpu_id: int, profile: dict | None) -> None:
    if not _ENABLED or not profile:
        return
    prefix = f"gpu{gpu_id}."
    for name, ms in profile.items():
        add(prefix + name, ms)


def attach(result: dict) -> dict:
    if not _ENABLED:
        return result
    attached = dict(result)
    attached["profile"] = snapshot()
    return attached
