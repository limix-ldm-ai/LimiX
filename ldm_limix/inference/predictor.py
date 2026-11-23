"""Architecture router. Predictor implementations live in inference.v1_0 and inference.v2_0."""

from __future__ import annotations

import torch
import time
import nvtx

from ldm_limix.model.version import resolve_arch_line, resolve_arch_version


def get_predictor_version(arch_version):
    """Return the LimiXPredictor class for a checkpoint architecture version.

    Input:
        arch_version: Version string or packaging.version.Version from the
            checkpoint (for example '1.0' or '2.0'). Values < 2.0 route to
            inference.v1_0, 2.0 routes to inference.v2_0.

    Output:
        type: the version-specific LimiXPredictor class, not an instance.
    """
    line = resolve_arch_line(arch_version)
    if line == "v1_0":
        from ldm_limix.inference.v1_0.predictor import LimiXPredictor as predictor
    else:
        from ldm_limix.inference.v2_0.predictor import LimiXPredictor as predictor
    return predictor


def _resolve_model_path(args, kwargs) -> str:
    """Read model_path from LimiXPredictor constructor arguments.

    Input:
        args: Positional args forwarded to the version-specific constructor.
            When model_path is not in kwargs, args[1] must be the checkpoint path.
        kwargs: Keyword args. model_path, if present, wins over args[1].

    Output:
        str: checkpoint path. Raises TypeError if neither source provides it.
    """
    if "model_path" in kwargs:
        return kwargs["model_path"]
    if len(args) >= 2:
        return args[1]
    raise TypeError("LimiXPredictor requires model_path")


def LimiXPredictor(*args, **kwargs):
    """Construct the version-specific predictor and load the checkpoint once.

    This is the public factory. It peeks at the checkpoint, then instantiates
    inference.v1_0 or inference.v2_0 LimiXPredictor.

    Input:
        *args: Positional args for the version-specific constructor. The usual
            call is (device, model_path, inference_config, ...). args[1] is
            treated as model_path when that name is not in kwargs.
        **kwargs: Keyword args forwarded to the routed constructor.
            model_path (str): checkpoint path; required unless given as args[1].
            ckpt (dict, optional): already-loaded checkpoint. When set, skips
            torch.load. Remaining keys must match the routed class (unknown
            names are warned and ignored).

    Output:
        LimiXPredictor: a version-specific instance with weights loaded on CPU.
            Call predict() for inference. Classification returns probabilities
            of shape (n_query, n_classes); regression returns shape (n_query,).
    """
    model_path = _resolve_model_path(args, kwargs)
    ckpt = kwargs.pop("ckpt", None)
    if ckpt is None:
        load_tic = time.time()
        with nvtx.annotate('ckpt-load'):
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        load_toc = time.time()
        load_time = (load_toc - load_tic) * 1000
        # print(f'loading ckpt ({model_path}) costs {load_time:.2f}ms')
    arch_version = resolve_arch_version(ckpt)
    line = resolve_arch_line(arch_version)
    predictor = get_predictor_version(arch_version)
    print(f"arch_version={arch_version} -> inference.{line}.predictor ({predictor.__module__})")
    kwargs["ckpt"] = ckpt
    return predictor(*args, **kwargs)


__all__ = ["LimiXPredictor", "get_predictor_version"]
