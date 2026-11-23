from functools import wraps
import torch
import numpy as np
import gc
from torch.cuda import OutOfMemoryError
from typing import Literal


class AutobatchConfig:
    ENABLE_AUTOBATCH = True


def _is_retryable_cuda_error(exc: BaseException) -> bool:
    if isinstance(exc, OutOfMemoryError):
        return True
    if isinstance(exc, MemoryError):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return 'cuda error: an illegal memory access' in msg or 'cuda error: invalid configuration argument' in msg or 'out of memory' in msg
    return False


def _dtype_nbytes(dtype) -> int:
    if isinstance(dtype, torch.dtype):
        return torch.tensor([], dtype=dtype).element_size()
    return int(np.dtype(dtype).itemsize)


def _shape_nbytes(shape, dtype) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n * _dtype_nbytes(dtype)


def _cuda_free_bytes(device) -> int | None:
    if not isinstance(device, torch.device) or device.type != 'cuda':
        return None
    try:
        free, _total = torch.cuda.mem_get_info(device)
        return int(free)
    except Exception:
        props = torch.cuda.get_device_properties(device)
        return int(props.total_memory - torch.cuda.memory_allocated(device))


def _try_empty_tensor(shape, dtype, device):
    """Allocate a full output buffer. Return None if allocation would OOM."""
    nbytes = _shape_nbytes(shape, dtype)
    free = _cuda_free_bytes(device)
    if free is not None and nbytes > int(0.8 * free):
        print(
            f"auto batch skip inplace prealloc: need {nbytes / 1024**3:.2f}GiB, "
            f"free {free / 1024**3:.2f}GiB"
        )
        return None
    try:
        return torch.empty(shape, dtype=dtype, device=device)
    except Exception as e:
        if not _is_retryable_cuda_error(e):
            raise
        print(f"auto batch inplace prealloc OOM, fallback to chunk concat: {e}")
        if isinstance(device, torch.device) and device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        return None


def _try_empty_ndarray(shape, dtype):
    try:
        return np.empty(shape, dtype=dtype)
    except Exception as e:
        if not _is_retryable_cuda_error(e):
            raise
        print(f"auto batch inplace ndarray prealloc failed, fallback to chunk concat: {e}")
        gc.collect()
        return None


def _preallocate_from_probe(test_output, num_samples, batch_dim, device):
    """
    Try to allocate a full-size output from a 1-sample probe.
    Returns (output, output_type, do_inplace).
    """
    if isinstance(test_output, (torch.Tensor, np.ndarray)):
        assert test_output.shape[batch_dim] == 1, "batch_dim mismatch!"
        output_shape = test_output.shape[:batch_dim] + (num_samples,) + test_output.shape[batch_dim + 1:]
        if isinstance(test_output, torch.Tensor):
            output_tensor = _try_empty_tensor(output_shape, test_output.dtype, device)
        else:
            output_tensor = _try_empty_ndarray(output_shape, test_output.dtype)
        if output_tensor is None:
            return None, None, False
        slices = [slice(None)] * get_dim(output_tensor)
        slices[batch_dim] = slice(0, 1)
        output_tensor[tuple(slices)] = test_output
        return output_tensor, type(test_output), True

    if isinstance(test_output, (list, tuple)):
        output_tensor = []
        output_type = type(test_output)
        for item in test_output:
            if item is not None and not isinstance(item, (torch.Tensor, np.ndarray)):
                return None, None, False
            if isinstance(item, (torch.Tensor, np.ndarray)) and item.shape[batch_dim] == 1:
                item_shape = item.shape[:batch_dim] + (num_samples,) + item.shape[batch_dim + 1:]
                if isinstance(item, torch.Tensor):
                    tmp_tensor = _try_empty_tensor(item_shape, item.dtype, device)
                else:
                    tmp_tensor = _try_empty_ndarray(item_shape, item.dtype)
                if tmp_tensor is None:
                    return None, None, False
                slices = [slice(None)] * get_dim(tmp_tensor)
                slices[batch_dim] = slice(0, 1)
                tmp_tensor[tuple(slices)] = item
                output_tensor.append(tmp_tensor)
            else:
                output_tensor.append(item)
        return output_tensor, output_type, True

    return None, None, False


def get_dim(x):
    return x.dim() if isinstance(x, torch.Tensor) else x.ndim


def slice_args(args, kwargs, start_idx, end_idx, batch_dim, num_samples, element_type, num_batched_tensor):
    ret_args, ret_kwargs = [], {}
    counter = 0
    for arg in args:
        if isinstance(arg, element_type) and arg.shape[batch_dim] == num_samples:
            counter += 1
            if num_batched_tensor is None or counter <= num_batched_tensor:
                slices = [slice(None)] * get_dim(arg)
                slices[batch_dim] = slice(start_idx, end_idx)
                ret_args.append(arg[tuple(slices)])
            else:
                ret_args.append(arg)
        else:
            ret_args.append(arg)
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor) and value.shape[batch_dim] == num_samples:
            counter += 1
            if num_batched_tensor is None or counter <= num_batched_tensor:
                slices = [slice(None)] * get_dim(value)
                slices[batch_dim] = slice(start_idx, end_idx)
                ret_kwargs[key] = value[tuple(slices)]
        else:
            ret_kwargs[key] = value
    return ret_args, ret_kwargs


def autobatch(
        batch_size: int|None=None,
        batch_dim: int = 0,
        auto_adjust: bool = True,
        inplace: bool = True,
        element_type: Literal['tensor', 'ndarray'] = 'tensor',
        num_batched_tensor: int|None = None
            ):
    '''
    Autobatch decorator factory.
    :param batch_size: Manual batch size; None enables adaptive batching
    :type batch_size: int | None
    :param batch_dim: Dimension to split on, default 0
    :type batch_dim: int
    :param auto_adjust: Halve batch_size and retry on OOM during batched compute
    :type auto_adjust: bool
    :param inplace: Preallocate output buffers to reduce peak memory (tensors, ndarrays, or lists/tuples of those or None)
    :type inplace: bool
    :param element_type: 'tensor' or 'ndarray'
    :type element_type: Literal['tensor', 'ndarray']
    :param num_batched_tensor: Number of tensors to batch, used when shapes would otherwise collide
    :type num_batched_tensor: int | None
    '''
    if element_type == 'tensor':
        element_type = torch.Tensor
        is_tensor = True
    elif element_type == 'ndarray':
        element_type = np.ndarray
        is_tensor = False
    else:
        raise ValueError(f"invalid element_type: {element_type}")
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not AutobatchConfig.ENABLE_AUTOBATCH:
                return func(*args, **kwargs)
            if batch_size is None:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if not _is_retryable_cuda_error(e):
                        raise
                    torch.cuda.empty_cache()
                    gc.collect()
            
            print("auto batch start")
            num_samples = None
            device = 'cpu'
            for arg in args:
                if isinstance(arg, element_type):
                    num_samples = arg.shape[batch_dim]
                    if is_tensor:
                        device = arg.device
                    break
            if num_samples is None:
                for key, value in kwargs.items():
                    if isinstance(value, element_type):
                        num_samples = value.shape[batch_dim]
                        if is_tensor:
                            device = value.device
                        break
            
            if num_samples is None:
                raise ValueError("no tensor or array found in args or kwargs")
            
            if num_samples == 1:
                return func(*args, **kwargs)
            
            if batch_size is not None and batch_size >= num_samples:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if not _is_retryable_cuda_error(e):
                        raise
                    torch.cuda.empty_cache()
                    gc.collect()
            
            do_inplace = False
            output_tensor = None
            output_type = None
            max_used_memory = None
            effective_batch_size = num_samples // 2
            is_cuda = isinstance(device, torch.device) and device.type == 'cuda'

            if inplace:
                test_args, test_kwargs = slice_args(args, kwargs, 0, 1, batch_dim, num_samples, element_type, num_batched_tensor)
                if batch_size is None and is_cuda:
                    current_memory = torch.cuda.memory_allocated(device)
                    torch.cuda.reset_peak_memory_stats(device)
                test_output = func(*test_args, **test_kwargs)
                if batch_size is None and is_cuda:
                    max_used_memory = torch.cuda.max_memory_allocated(device) - current_memory
                output_tensor, output_type, do_inplace = _preallocate_from_probe(
                    test_output, num_samples, batch_dim, device
                )
                if batch_size is None and is_cuda and max_used_memory and max_used_memory > 0:
                    free_memory = _cuda_free_bytes(device)
                    if free_memory is None:
                        free_memory = torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)
                    effective_batch_size = max(1, int(0.8 * free_memory / max_used_memory))
                else:
                    effective_batch_size = num_samples // 2
            

            effective_batch_size = batch_size if batch_size is not None else effective_batch_size
            effective_batch_size = max(min(effective_batch_size, num_samples), 1)
            
            while (effective_batch_size>=1):
                iterator = range(1, num_samples, effective_batch_size) if do_inplace else range(0, num_samples, effective_batch_size)
                
                passed = True
                for idx, start_idx in enumerate(iterator):
                    end_idx = min(start_idx + effective_batch_size, num_samples)
                    batch_size_ = end_idx - start_idx
                    batch_args, batch_kwargs = slice_args(args, kwargs, start_idx, end_idx, batch_dim, num_samples, element_type, num_batched_tensor)
                    
                    try:
                        if do_inplace:
                            if isinstance(output_tensor, (torch.Tensor, np.ndarray)):
                                slices = [slice(None)] * get_dim(output_tensor)
                                slices[batch_dim] = slice(start_idx, end_idx)
                                output_tensor[tuple(slices)] = func(*batch_args, **batch_kwargs)
                            else:
                                batch_out = func(*batch_args, **batch_kwargs)
                                for i, out in enumerate(batch_out):
                                    if isinstance(out, (torch.Tensor, np.ndarray)) and out.shape[batch_dim] == batch_size_:
                                        slices = [slice(None)] * get_dim(out)
                                        slices[batch_dim] = slice(start_idx, end_idx)
                                        output_tensor[i][tuple(slices)] = out
                                    else:
                                        output_tensor[i] = out

                        else:
                            batch_result = func(*batch_args, **batch_kwargs)
                            
                            if idx == 0:
                                if not isinstance(batch_result, (torch.Tensor, np.ndarray, tuple, list, dict)):
                                    raise RuntimeError('unsupported output dtype!')
                                else:
                                    result = batch_result
                                    res_type = type(result)
                            else:
                                if isinstance(batch_result, torch.Tensor):
                                    result = torch.cat([result, batch_result], dim=batch_dim)
                                elif isinstance(batch_result, np.ndarray):
                                    result = np.concatenate([result, batch_result], axis=batch_dim)
                                elif isinstance(batch_result, (list, tuple)):
                                    result = list(result) if not isinstance(result, list) else result
                                    for i in range(len(result)):
                                        if isinstance(batch_result[i], torch.Tensor):
                                            if batch_result[i].shape[batch_dim] == batch_size_:
                                                result[i] = torch.cat((result[i], batch_result[i]), dim=batch_dim)
                                            else:
                                                result[i] = batch_result[i]
                                        elif isinstance(batch_result[i], np.ndarray):
                                            if batch_result[i].shape[batch_dim] == batch_size_:
                                                result[i] = np.concatenate((result[i], batch_result[i]), axis=batch_dim)
                                            else:
                                                result[i] = batch_result[i]
                                        elif isinstance(batch_result[i], (list, tuple)):
                                            result[i] = type(result[i])(result[i] + batch_result[i])
                                elif isinstance(batch_result, dict):
                                    for key in result.keys():
                                        if isinstance(batch_result[key], torch.Tensor):
                                            if batch_result[key].shape[batch_dim] == batch_size_:
                                                result[key] = torch.cat((result[key], batch_result[key]), dim=batch_dim)
                                            else:
                                                result[key] = batch_result[key]
                                        elif isinstance(batch_result[key], np.ndarray):
                                            if batch_result[key].shape[batch_dim] == batch_size_:
                                                result[key] = np.concatenate((result[key], batch_result[key]), axis=batch_dim)
                                            else:
                                                result[key] = batch_result[key]
                                        elif isinstance(batch_result[key], (list, tuple)):
                                            result[key] = type(result[key])(result[key] + batch_result[key])   
                            
                    except Exception as e:
                        if not _is_retryable_cuda_error(e):
                            raise
                        if auto_adjust:
                            passed = False
                            torch.cuda.empty_cache()
                            gc.collect()
                            if effective_batch_size == 1:
                                raise e
                            effective_batch_size = max(1, effective_batch_size // 2)
                            break
                        else:
                            torch.cuda.empty_cache()
                            gc.collect()
                            raise e
                
                if passed:
                    if do_inplace:
                        return output_tensor if not isinstance(output_tensor, list) else output_type(output_tensor)
                    else:
                        if isinstance(result, np.ndarray):
                            return result
                        return res_type(result)
        return wrapper
    return decorator