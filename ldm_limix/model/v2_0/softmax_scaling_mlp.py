import torch
import torch.nn as nn
from typing import override


def _soft_clamp_range(
    x: torch.Tensor, lower_bound: float, upper_bound: float
) -> torch.Tensor:
    """Soft-clamp x elementwise into (lower_bound, upper_bound)."""
    width = upper_bound - lower_bound
    midpoint = (upper_bound + lower_bound) / 2
    return (width / 2) * torch.tanh((2 / width) * x) + midpoint

class SoftmaxScalingMLP(nn.Module):
    """Simplified per-head scalar scaling with log-n temperature.

    Applies scaling to queries:

        q_scaled = q * scales,

    where scales = logn_scale * bias_scale, and the learned temperature
    `soft_pos_weight` is soft-clamped into
    (temp_lower_bound, temp_upper_bound).

    - scale_linear.weight stores per-head log-n sensitivity.
    - scale_linear.bias stores per-head base scalar.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        temp_upper_bound: float = 0.5,
        temp_lower_bound: float = 0.0,
        scale_base_bound: float = 5.0,
    ):
        super().__init__()
        if temp_upper_bound <= temp_lower_bound:
            raise ValueError("temp_upper_bound must be greater than temp_lower_bound")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.temp_upper_bound = temp_upper_bound
        self.temp_lower_bound = temp_lower_bound
        self.scale_base_bound = scale_base_bound

        # Merged layer: log(n/400) -> H
        # weight (0.5): per-head log-n sensitivity (formerly head_temperature)
        # bias  (1.0):  per-head base scalar       (formerly base_direction)
        self.scale_linear = nn.Linear(1, num_heads, bias=True)

    def _clamped_direction(self, x: torch.Tensor) -> torch.Tensor:
        """Soft-clamp x elementwise via tanh(x/10)*10 to (-10, 10).

        Shape: same as input.
        """
        return (self.scale_base_bound / 2) * torch.tanh((x * 2)/self.scale_base_bound) + (self.scale_base_bound / 2)

    @override
    def forward(self, q_BSHD: torch.Tensor, n: int) -> torch.Tensor:
        """Applies per-head scalar scaling to queries.

        Args:
            q_BSHD: Query tensor, shape [B, S, H, D].
            n:      Sequence length used for log(n/400) scaling.

        Returns:
            Scaled query tensor, same shape as q_BSHD.
        """
        # 1. log(n): scalar input to linear, shape [1, 1]
        logn_ratio_11 = torch.log(
            torch.tensor(n / 1.0, device=q_BSHD.device)
        ).reshape(1, 1).to(dtype=q_BSHD.dtype)

        # 2. log-n scaling factor: (soft_pos_weight * logn) + 1 -> [H]
        soft_pos_weight = _soft_clamp_range(
            self.scale_linear.weight,
            self.temp_lower_bound,
            self.temp_upper_bound,
        )  # [H, 1]
        logn_scale_H = (soft_pos_weight @ logn_ratio_11.T).squeeze() + 1.0  # [H]

        # 3. Base scaling factor: soft-clamp bias -> [H]
        bias_scale_H = self._clamped_direction(self.scale_linear.bias).to(dtype=q_BSHD.dtype)  # [H]

        # 4. Final scale: elementwise product -> [H] -> [1, 1, H, 1]
        scales_H = logn_scale_H * bias_scale_H
        base_scales = scales_H.reshape(1, 1, self.num_heads, 1)

        return q_BSHD * base_scales


