"""Pure helpers for selecting a CUDA-memory-aware SVD feature budget."""

from __future__ import annotations

import math
from typing import Any


MIB = 1024**2


def estimate_inference_activation_bytes(
    *,
    row_count: int,
    feature_count: int,
    embedding_dim: int,
    num_heads: int,
    mixed_precision: bool,
    sequence_attention_limit: int | None = None,
    safety_factor: float = 1.25,
) -> int:
    """Estimate peak feature-sensitive inference activations.

    This intentionally models token activations plus both feature- and
    sequence-attention workspaces.  It is a conservative capacity estimator,
    not an exact PyTorch allocator simulation.
    """
    for name, value in (
        ("row_count", row_count),
        ("feature_count", feature_count),
        ("embedding_dim", embedding_dim),
        ("num_heads", num_heads),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if safety_factor <= 0:
        raise ValueError("safety_factor must be positive")
    if sequence_attention_limit is not None and sequence_attention_limit < 1:
        raise ValueError("sequence_attention_limit must be positive when provided")

    scalar_bytes = 2 if mixed_precision else 4
    sequence_span = min(
        row_count,
        sequence_attention_limit or row_count,
    )
    token_elements = 6.0 * row_count * feature_count * embedding_dim
    feature_attention_elements = (
        0.5 * num_heads * row_count * feature_count * feature_count
    )
    sequence_attention_elements = (
        0.5 * num_heads * row_count * sequence_span * feature_count
    )
    return int(
        math.ceil(
            scalar_bytes
            * safety_factor
            * (
                token_elements
                + feature_attention_elements
                + sequence_attention_elements
            )
        )
    )


def choose_svd_components(
    *,
    requested_components: int,
    base_output_features: int,
    train_rows: int,
    query_rows: int,
    free_cuda_bytes: int,
    model_bytes_to_load: int,
    embedding_dim: int,
    num_heads: int,
    mixed_precision: bool,
    memory_fraction: float = 0.65,
    reserve_mb: int = 512,
    safety_factor: float = 1.25,
    minimum_components: int = 0,
    sequence_attention_limit: int | None = None,
) -> dict[str, Any]:
    """Return the largest requested SVD count fitting the activation budget."""
    for name, value, minimum in (
        ("requested_components", requested_components, 0),
        ("base_output_features", base_output_features, 1),
        ("train_rows", train_rows, 1),
        ("query_rows", query_rows, 1),
        ("free_cuda_bytes", free_cuda_bytes, 0),
        ("model_bytes_to_load", model_bytes_to_load, 0),
        ("minimum_components", minimum_components, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    if not 0.0 < memory_fraction <= 1.0:
        raise ValueError("memory_fraction must be in (0, 1]")
    if reserve_mb < 0:
        raise ValueError("reserve_mb must be non-negative")

    minimum_components = min(minimum_components, requested_components)
    reserve_bytes = reserve_mb * MIB
    unreserved_bytes = max(
        0,
        free_cuda_bytes - model_bytes_to_load - reserve_bytes,
    )
    activation_budget_bytes = int(unreserved_bytes * memory_fraction)
    row_count = train_rows + query_rows

    def estimate(component_count: int) -> int:
        return estimate_inference_activation_bytes(
            row_count=row_count,
            feature_count=base_output_features + component_count,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            mixed_precision=mixed_precision,
            sequence_attention_limit=sequence_attention_limit,
            safety_factor=safety_factor,
        )

    requested_estimate = estimate(requested_components)
    base_estimate = estimate(0)
    low = minimum_components
    high = requested_components
    if estimate(low) > activation_budget_bytes:
        selected_components = 0
    else:
        while low < high:
            middle = (low + high + 1) // 2
            if estimate(middle) <= activation_budget_bytes:
                low = middle
            else:
                high = middle - 1
        selected_components = low

    selected_estimate = estimate(selected_components)
    adapted = selected_components < requested_components
    if not adapted:
        reason = "requested_components_fit_memory_budget"
    elif base_estimate > activation_budget_bytes:
        reason = "base_features_already_exceed_memory_budget"
    else:
        reason = "svd_components_reduced_for_memory_budget"
    return {
        "requested_components": int(requested_components),
        "selected_components": int(selected_components),
        "adapted": bool(adapted),
        "reason": reason,
        "base_output_features": int(base_output_features),
        "selected_output_features": int(
            base_output_features + selected_components
        ),
        "train_rows": int(train_rows),
        "query_rows_for_budget": int(query_rows),
        "free_cuda_bytes": int(free_cuda_bytes),
        "model_bytes_to_load": int(model_bytes_to_load),
        "activation_budget_bytes": int(activation_budget_bytes),
        "estimated_base_activation_bytes": int(base_estimate),
        "estimated_requested_activation_bytes": int(requested_estimate),
        "estimated_selected_activation_bytes": int(selected_estimate),
    }
