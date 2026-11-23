"""Precision helpers shared by the explicit FlashAttention backends."""

from __future__ import annotations

from typing import Literal, cast

import torch


FlashAttentionPrecision = Literal["inherit", "fp16", "bf16"]

_FLASH_ATTENTION_DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def normalize_flash_attention_precision(
    precision: str | None,
) -> FlashAttentionPrecision:
    """Validate and normalize the configured FlashAttention precision."""
    normalized = "inherit" if precision is None else str(precision).lower()
    normalized = {
        "float16": "fp16",
        "bfloat16": "bf16",
    }.get(normalized, normalized)
    if normalized not in {"inherit", *_FLASH_ATTENTION_DTYPES}:
        raise ValueError(
            "flash_attention_precision must be one of "
            "{'inherit', 'fp16', 'bf16'}, "
            f"got {precision!r}"
        )
    return cast(FlashAttentionPrecision, normalized)


def resolve_flash_attention_dtype(
    precision: FlashAttentionPrecision,
    input_dtype: torch.dtype,
) -> torch.dtype:
    """Return the dtype to use inside FlashAttention itself."""
    if precision == "inherit":
        return input_dtype
    return _FLASH_ATTENTION_DTYPES[precision]


def cast_flash_attention_inputs(
    precision: FlashAttentionPrecision,
    *tensors: torch.Tensor | None,
) -> tuple[torch.Tensor | None, ...]:
    """Cast only FlashAttention inputs, leaving projections/model state alone."""
    reference = next((tensor for tensor in tensors if tensor is not None), None)
    if reference is None:
        return tensors
    target_dtype = resolve_flash_attention_dtype(precision, reference.dtype)
    return tuple(
        tensor.to(dtype=target_dtype) if tensor is not None else None
        for tensor in tensors
    )
