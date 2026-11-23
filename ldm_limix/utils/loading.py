"""Architecture-version dispatcher for model load.

Implementation lives in model.v1_0.loading and model.v2_0.loading.
"""
from ldm_limix.model.version import resolve_arch_line, resolve_arch_version


def _loading_module(arch_version):
    line = resolve_arch_line(arch_version)
    if line == "v1_0":
        from ldm_limix.model.v1_0 import loading as loading_mod
    else:
        from ldm_limix.model.v2_0 import loading as loading_mod
    return loading_mod


def load_model(model_path, mask_prediction: bool = False, deterministic: bool = False):
    import torch
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    arch_version = resolve_arch_version(ckpt)
    return _loading_module(arch_version).load_from_checkpoint(
        ckpt, mask_prediction=mask_prediction, deterministic=deterministic
    )


def load_from_checkpoint(ckpt, mask_prediction: bool = False, deterministic: bool = False):
    arch_version = resolve_arch_version(ckpt)
    return _loading_module(arch_version).load_from_checkpoint(
        ckpt, mask_prediction=mask_prediction, deterministic=deterministic
    )
