import nvtx
import torch
import torch.nn as nn
from torch.amp import autocast


class RMSNormMixedPrecision(nn.RMSNorm):
    """
    When the embedding dimension is below 512, use half precision for computation to improve performance.
    If the embedding dimension exceeds 512, it may cause training instability.
    """

    # NOTE: remove nvtx.annotate to avoid re-entry under recompute since out-layer has nvtx.annotate
    def forward(self, input: torch.Tensor):
        if input.dtype == torch.float16 and sum(self.normalized_shape) < 512:
            with autocast(device_type="cuda" if input.is_cuda else "cpu", enabled=False):
                return self._forward(input)
        else:
            return self._forward(input)

    def _forward(self, input: torch.Tensor):
        if self.elementwise_affine and input.dtype != self.weight.dtype:
            input = input.to(self.weight.dtype)
        return super().forward(input)
