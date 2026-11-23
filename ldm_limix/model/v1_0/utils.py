import torch
import torch.nn as nn
from functools import wraps
from typing import Callable, List, Any, Optional
import random
from contextlib import ContextDecorator
import numpy as np
import logging
from .operators.triton_rmsnorm import build_norm


def get_logger(root: str = 'root', module: str = None) -> logging.Logger:
    full_module = f'{root}.{module}' if module is not None else root
    return logging.getLogger(full_module)


def create_mlp_layer(mlp_type, input_dim_size, hidden_size, output_dim_size, dropout=0.0, bias=True, mlp_pattern=None, use_rmsnorm=False, layer_recompute=False):
    if mlp_type == 'Free_MLP':
        return create_mlp_4_free_type(mlp_pattern, input_dim_size, output_dim_size, bias=True)

    if 'GELU' not in mlp_type:
        raise ValueError(f"Unknown mlp_type: {mlp_type}")

    layer_elements = [
        nn.Linear(input_dim_size, hidden_size, bias=bias),
    ]
    if '_innernorm' in mlp_type:
        layer_elements.append(build_norm(hidden_size, use_rmsnorm=use_rmsnorm, recompute=layer_recompute))
    layer_elements.append(nn.GELU())
    if dropout > 0:
        layer_elements.append(nn.Dropout(dropout))
    layer_elements.append(nn.Linear(hidden_size, output_dim_size, bias=bias))
    if '_postnorm' in mlp_type:
        layer_elements.append(build_norm(output_dim_size, use_rmsnorm=use_rmsnorm, recompute=layer_recompute))
    return nn.Sequential(*layer_elements)


def create_mlp_4_free_type(mlp_pattern, input_dim_size, output_dim_size, bias=True):
    """Build an MLP from mlp_pattern, e.g. 'linear-768_GELU_linear-10'."""
    last_layer_output_dim = input_dim_size
    mlp_elements = []
    for layer_pattern in mlp_pattern.split('_'):
        layer_elements = layer_pattern.split('-')
        if 'linear' == layer_elements[0]:
            assert len(layer_elements) == 2 and layer_elements[1].isdigit(), f"Invalid linear layer pattern {layer_pattern} in mlp pattern {mlp_pattern}"
            cur_layer_output_dim = int(layer_elements[1])
            mlp_elements.append(nn.Linear(last_layer_output_dim, cur_layer_output_dim, bias=bias))
            last_layer_output_dim = cur_layer_output_dim
        elif layer_elements[0] == 'GELU':
            mlp_elements.append(nn.GELU())
        else:
            raise ValueError(f"Unknown layer pattern {layer_pattern} in mlp pattern {mlp_pattern}")

    assert last_layer_output_dim == output_dim_size, f"Output dim size {last_layer_output_dim} not equal to expected {output_dim_size} in mlp pattern {mlp_pattern}"
    return nn.Sequential(*mlp_elements)


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

            outputs = []
            for start in range(0, bs, batch_size):
                end = start + batch_size
                chunked_args = slice_args(args, bs, batch_dim, start, end, num_args)
                chunk_output = func(*chunked_args, **kwargs)
                outputs.append(chunk_output)
            output = torch.concat(outputs, dim=batch_dim)
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
