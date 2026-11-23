import logging

import torch
import torch.nn as nn
from functools import wraps
from typing import Callable, List, Any, Optional
import random
from contextlib import ContextDecorator
import numpy as np
from torch.nn.attention import SDPBackend, sdpa_kernel
from .operators.rmsnorm import build_rmsnorm


def get_logger(root: str = 'root', module: str = None) -> logging.Logger:
    full_module = f'{root}.{module}' if module is not None else root
    return logging.getLogger(full_module)


def sdpa_context(deterministic: bool):
    """SDPA backend context; disables FlashAttention when deterministic=True."""
    backends = [
        SDPBackend.MATH,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.CUDNN_ATTENTION,
    ]
    if not deterministic:
        backends = [SDPBackend.FLASH_ATTENTION, *backends]
    return sdpa_kernel(backends)


activation_map = {
    'ReLU': nn.ReLU,
    'GELU': nn.GELU,
    'SiLU': nn.SiLU,
}

def create_mlp_layer(mlp_type, input_dim_size, hidden_size, output_dim_size, dropout=0.0, bias=True, mlp_pattern=None,
                     y_token=0, norm_impl='triton', recompute=False):
    if mlp_type == 'Free_MLP':
        return create_mlp_4_free_type(
            mlp_pattern, input_dim_size, output_dim_size, bias=True, y_token=y_token,
            norm_impl=norm_impl, recompute=recompute,
        )

    active_func = 'ReLU'
    if 'ReLU' in mlp_type:
        active_func = 'ReLU'
    elif 'GELU' in mlp_type:
        active_func = 'GELU'
    elif 'SiLU' in mlp_type:
        active_func = 'SiLU'

    hidden_linear = nn.Linear(input_dim_size, hidden_size, bias=bias)
    activation = activation_map[active_func]()
    output_linear = nn.Linear(hidden_size, output_dim_size, bias=bias)

    layer_elements = []

    if '_pernorm' in mlp_type:
        layer_elements.append(build_rmsnorm(
            input_dim_size, eps=1e-5, elementwise_affine=True,
            norm_impl=norm_impl, recompute=recompute,
        ))

    layer_elements.append(hidden_linear)

    if '_innernorm' in mlp_type:
        layer_elements.append(build_rmsnorm(
            hidden_size, eps=1e-5, elementwise_affine=True,
            norm_impl=norm_impl, recompute=recompute,
        ))

    layer_elements.append(activation)

    if dropout > 0:
        layer_elements.append(nn.Dropout(dropout))

    if '_activationnorm' in mlp_type:
        layer_elements.append(build_rmsnorm(
            hidden_size, eps=1e-5, elementwise_affine=True,
            norm_impl=norm_impl, recompute=recompute,
        ))

    layer_elements.append(output_linear)

    if '_postnorm' in mlp_type:
        layer_elements.append(build_rmsnorm(
            output_dim_size, eps=1e-5, elementwise_affine=True,
            norm_impl=norm_impl, recompute=recompute,
        ))

    mlp_layer = nn.Sequential(*layer_elements)

    return mlp_layer


def create_mlp_4_free_type(mlp_pattern, input_dim_size, output_dim_size, bias=True, y_token=0,
                           norm_impl='triton', recompute=False):
    """Build a generic MLP from mlp_pattern, e.g. 'linear-768_GELU_dropout-0.1_linear-768_linear-10'."""
    mlp_pattern_elements = mlp_pattern.split('_')

    last_layer_output_dim = input_dim_size
    mlp_elements = []
    for layer_pattern in mlp_pattern_elements:
        layer_elements = layer_pattern.split('-')
        assert layer_elements[0] in ['linear', 'GELU', 'RMSNorm', 'Norm', 'ReLU', 'SiLU',
                                     'dropout'], f"Unknown layer pattern {layer_pattern} in mlp pattern {mlp_pattern}"
        if 'linear' == layer_elements[0]:
            assert len(layer_elements) == 2 and layer_elements[
                1].isdigit(), f"Invalid linear layer pattern {layer_pattern} in mlp pattern {mlp_pattern}"
            cur_layer_output_dim = int(layer_elements[1]) if (y_token == 0 or int(layer_elements[1]) == 10 or int(layer_elements[1])==5000) else int(
                layer_elements[1]) * y_token
            mlp_elements.append(nn.Linear(last_layer_output_dim, cur_layer_output_dim, bias=bias))
            last_layer_output_dim = cur_layer_output_dim
        elif layer_elements[0] in activation_map:
            active_func_type = layer_elements[0]
            mlp_elements.append(activation_map[active_func_type]())
        elif 'dropout' == layer_elements[0]:
            assert len(layer_elements) == 2 and layer_elements[1].replace('.', '',
                                                                          1).isdigit(), f"Invalid dropout layer pattern {layer_pattern} in mlp pattern {mlp_pattern}"
            dropout_prob = float(layer_elements[1])
            mlp_elements.append(nn.Dropout(dropout_prob))
        elif layer_elements[0] in ('Norm', 'RMSNorm'):
            assert len(layer_elements) == 1, f"Invalid norm layer pattern {layer_pattern} in mlp pattern {mlp_pattern}"
            mlp_elements.append(build_rmsnorm(
                last_layer_output_dim, eps=1e-5, elementwise_affine=True,
                norm_impl=norm_impl, recompute=recompute,
            ))
        else:
            raise ValueError(f"Unknown layer pattern {layer_pattern} in mlp pattern {mlp_pattern}")

    assert last_layer_output_dim == output_dim_size, f"Output dim size {last_layer_output_dim} not equal to expected {output_dim_size} in mlp pattern {mlp_pattern}"

    mlp_layer = nn.Sequential(*mlp_elements)
    return mlp_layer


class AdapterWithResidual(nn.Module):
    def __init__(self, adapter, dim, use_residual=True, norm_impl='triton', recompute=False):
        super().__init__()
        self.adapter = adapter
        self.layer_norm = build_rmsnorm(
            dim, eps=1e-5, elementwise_affine=True,
            norm_impl=norm_impl, recompute=recompute,
        )
        self.use_residual = use_residual

    def forward(self, x, task_type = 'feat'):
        if self.use_residual:
            return self.layer_norm(x + self.adapter(x))
        else:
            return self.layer_norm(self.adapter(x))


def slice_args(args: Any, bs: int, batch_dim: int, start: int, end: int, num_args: Optional[int]=None):
    chunked_args = []
    counter = 0
    for arg in args:
        # ignore args whose batch_dim is not bs
        if isinstance(arg, torch.Tensor) and arg.size(batch_dim) == bs:
            counter += 1
            if num_args is None or counter <= num_args:
                slice_obj = [slice(None)] * arg.dim()
                slice_obj[batch_dim] = slice(start, end)
                chunked_args.append(arg[tuple(slice_obj)])
            else:
                chunked_args.append(arg)
        else:
            chunked_args.append(arg)
    return chunked_args

def simple_autobatch(batch_size: int, batch_dim: int=0, num_args: Optional[int]=None):
    def decorator(func: Callable) -> Callable:
        @wraps(func) # keep the original function metadata
        def wrapper(*args, **kwargs) -> torch.Tensor:
            # return if no tensor args
            tensor_args = [ arg for arg in args if isinstance(arg, torch.Tensor) ]
            if not tensor_args:
                return func(*args, **kwargs)

            # return if bs <= batch_size
            bs = tensor_args[0].size(batch_dim)
            if bs <= batch_size:
                return func(*args, **kwargs)

            # NOTE: using torch.narrow can reduce peak memory compared with torch.concat
            output = None
            offset = 0
            for start in range(0, bs, batch_size):
                end = start + batch_size
                chunked_args = slice_args(args, bs, batch_dim, start, end, num_args)
                chunk_output = func(*chunked_args, **kwargs)
                if output is None:
                    full_shape = list(chunk_output.shape)
                    full_shape[batch_dim] = bs
                    output = torch.empty(full_shape, dtype=chunk_output.dtype, device=chunk_output.device)
                output.narrow(batch_dim, offset, chunk_output.size(batch_dim)).copy_(chunk_output)
                del chunk_output
                offset += (end - start)

            return output
        return wrapper
    return decorator


class SetRandomSeed(ContextDecorator):
    '''Set RNG seeds as a decorator or context manager. If seed is None, RNG state is left unchanged.'''
    def __init__(self, seed: int = None):
        self.seed = seed
        self.state_python = None
        self.state_numpy = None
        self.state_torch = None

    def __enter__(self):
        if self.seed is None:
            return self

        self.state_python = random.getstate()
        self.state_numpy = np.random.get_state()
        self.state_torch = torch.get_rng_state()
        if torch.cuda.is_available():
            self.state_torch_cuda = torch.cuda.get_rng_state()

        random.seed(self.seed)
        np.random.seed(self.seed + 1)
        torch.manual_seed(self.seed + 2)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed + 3)

        return self

    def __exit__(self, exc_type, exc, exc_tb):
        if self.seed is None:
            return False

        random.setstate(self.state_python)
        np.random.set_state(self.state_numpy)
        torch.set_rng_state(self.state_torch)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(self.state_torch_cuda)

        return False
