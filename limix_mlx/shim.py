"""Stubs for CUDA-only deps (nvtx, triton) so the reference LimiX code runs on Mac CPU.

Import this module first, before importing anything from limix_src.
"""
import sys
import types


def stub_cuda_only():
    if "nvtx" not in sys.modules:
        class _Annotate:
            def __init__(self, *a, **k): pass
            def __call__(self, f): return f
            def __enter__(self): return self
            def __exit__(self, *a): return False

        _nvtx = types.ModuleType("nvtx")
        _nvtx.annotate = _Annotate
        sys.modules["nvtx"] = _nvtx

    if "triton" not in sys.modules:
        _triton = types.ModuleType("triton")
        _triton.jit = lambda f=None, **k: (f if f is not None else (lambda g: g))
        _triton.next_power_of_2 = lambda n: 1 << (n - 1).bit_length()
        _tl = types.ModuleType("triton.language")
        _tl.constexpr = None
        _tl.float32 = "fp32"
        _tl.bfloat16 = "bf16"
        _tl.float16 = "fp16"
        sys.modules["triton"] = _triton
        sys.modules["triton.language"] = _tl


stub_cuda_only()
