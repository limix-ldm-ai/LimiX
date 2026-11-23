import torch
import torch.nn as nn
from .torch_rmsnorm import RMSNormMixedPrecision
from .triton_rmsnorm import TritonRMSNorm
from .triton_qknorm import TritonQKNorm


def build_rmsnorm(
        hidden_size,
        eps=1e-5,
        elementwise_affine=False,
        device=None,
        dtype=None,
        **kwargs,
    ):

    norm_impl = kwargs.get('norm_impl', 'torch')
    # norm_type and recompute are ONLY used for triton impl.
    norm_type = kwargs.get('norm_type', 'rmsnorm')
    recompute = kwargs.get('recompute', False)

    assert norm_impl in ['torch', 'triton'], f'invalid norm_impl {norm_impl}'
    assert norm_type in ['rmsnorm', 'qknorm'], f'invalid norm_type {norm_type}'

    if norm_impl == 'torch':
        return RMSNormMixedPrecision(
            hidden_size,
            eps=eps,
            elementwise_affine=elementwise_affine,
            device=device,
            dtype=dtype,
        )
    elif norm_impl == 'triton':
        if norm_type == 'rmsnorm':
            return TritonRMSNorm(
                hs=hidden_size,
                eps=eps,
                elementwise_affine=elementwise_affine,
                device=device,
                dtype=torch.float32 if dtype is None else dtype,
                recompute=recompute,
            )
        elif norm_type == 'qknorm':
            return TritonQKNorm(
                hs=hidden_size,
                eps=eps,
                elementwise_affine=elementwise_affine,
                device=device,
                dtype=torch.float32 if dtype is None else dtype,
                recompute=recompute,
            )


__all__ = ['build_rmsnorm', 'RMSNormMixedPrecision', 'TritonRMSNorm', 'TritonQKNorm']
