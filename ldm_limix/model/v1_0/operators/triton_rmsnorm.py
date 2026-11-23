import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    input_ptr, weight_ptr, output_ptr, size,
    NO_WEIGHT: tl.constexpr, OUT_DTYPE: tl.constexpr, EPS: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    bid = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < size

    input = tl.load(input_ptr + bid * size + offsets, mask=mask, other=0.0)
    input_fp32 = input.to(tl.float32)

    # NOTE: using rsqrt instead of /sqrt can be stable
    var = tl.sum(input_fp32 * input_fp32) / size
    inv_rms = tl.rsqrt(var + EPS)
    norm = input_fp32 * inv_rms
    if NO_WEIGHT:
        output = norm
    else:
        weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
        output = norm * weight.to(tl.float32)

    tl.store(output_ptr + bid * size + offsets, output.to(OUT_DTYPE), mask=mask)


def _backward_impl(ctx, grad_output):
    input, weight = ctx.saved_tensors
    eps = ctx.eps

    input_fp32 = input.float()
    weight_fp32 = None if weight is None else weight.float()
    grad_output_fp32 = grad_output.float()

    """
    RMS => y = norm * weight
             = x / rms * weight
    -> dy/dnorm = dL/dy * weight
    -> dy/drms = dL/dy * weight * (-x / rms ** 2) = -dy/dnorm * norm * / rms (**the same shape with rms**)
    -> dL/dx = dL/dy * dy/dx = dL/dy (weight / rms - weight * x / rms^2 * drmx/dx)
    ->       = dy/dnorm / rms - dy/dnorm * norm / rms * (x / rms / n)
    ->       = dy/dnorm / rms + dy/drms * (x / rms / n)
    ->       = dy/dnorm / rms + dy/drms * norm / n
    """

    grad_norm = grad_output_fp32 if weight is None else grad_output_fp32 * weight_fp32
    inv_rms = torch.rsqrt(torch.mean(input_fp32 * input_fp32, dim=-1, keepdim=True) + eps)
    norm = input_fp32 * inv_rms
    grad_rms = -torch.sum(grad_norm * norm * inv_rms, dim=-1, keepdim=True)
    grad_input = grad_norm * inv_rms + grad_rms * norm / norm.size(-1)
    if weight is None:
        grad_weight = None
    else:
        dims = tuple(range(input.ndim - 1)) # only keep the last dim
        grad_weight = torch.sum(grad_output_fp32 * norm, dim=dims).to(weight.dtype)

    return grad_input.to(input.dtype), grad_weight, None, None


class TritonRMSNormOP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, eps, output):
        ctx.save_for_backward(input, weight)
        ctx.eps = eps

        if output is None:
            output = torch.empty_like(input)

        H = input.size(-1)
        B = input.numel() // H
        BLOCK_SIZE = triton.next_power_of_2(H)

        if input.dtype == torch.float32:
            out_dtype = tl.float32
        elif input.dtype == torch.bfloat16:
            out_dtype = tl.bfloat16
        elif input.dtype == torch.float16:
            out_dtype = tl.float16
        rmsnorm_kernel[(B,)](input, weight, output, H, weight is None, out_dtype, eps, BLOCK_SIZE)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        return _backward_impl(ctx, grad_output)


class TritonRMSNormOPCompiled(TritonRMSNormOP):
    @staticmethod
    def forward(ctx, input, weight, eps, output):
        return TritonRMSNormOP.forward(ctx, input, weight, eps, output)

    # NOTE: dgrad rel_error increases with torch.compile
    # NOTE: calling superclass staticmethod in torch.compile is forbidden under autograd.Function.backward
    @staticmethod
    # @torch.compile(backend='inductor')
    def backward(ctx, grad_output):
        return _backward_impl(ctx, grad_output)


class TritonRMSNorm(nn.Module):
    def __init__(self, hs, eps, elementwise_affine=True, device=None, dtype=torch.float32, recompute=False):
        super(TritonRMSNorm, self).__init__()
        self.hs = hs
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.device = device # ONLY keep the same arglist with nn.RMSNorm
        self.dtype = dtype
        self.recompute = recompute
        weight = nn.Parameter(torch.ones(self.hs, dtype=self.dtype)) if self.elementwise_affine else None
        self.register_parameter('weight', weight)

    def forward(self, input, output=None):
        # TODO: improve tl.load to get non-contiguous data
        if not input.is_contiguous():
            input = input.contiguous()
        if not input.is_cuda:
            weight = self.weight
            if weight is not None:
                weight = weight.to(device=input.device, dtype=input.dtype)
            shape = (int(self.hs),) if isinstance(self.hs, int) else tuple(int(x) for x in self.hs)
            return F.rms_norm(input, shape, weight, self.eps)

        # NOTE: torch.compile + autograd.Function.backward is forbidden under checkpoint
        if self.recompute:
            return TritonRMSNormOP.apply(input, self.weight, self.eps, output)
        else:
            return TritonRMSNormOPCompiled.apply(input, self.weight, self.eps, output)

    def extra_repr(self):
        return 'hs=%s, eps=%s, elementwise_affine=%s, dtype=%s, recompute=%s' % (
            self.hs, self.eps, self.elementwise_affine, self.dtype, self.recompute
        )


def build_norm(
        normalized_shape,
        eps=1e-5,
        elementwise_affine=True,
        *,
        use_rmsnorm=False,
        recompute=False,
        device=None,
        dtype=None,
):
    """Build LayerNorm, or TritonRMSNorm when use_rmsnorm=True.

    TritonRMSNorm args match the former post-init replacement: hs is
    LayerNorm.normalized_shape (a 1-tuple when given an int), and
    device/dtype are omitted so weights default to float32.
    """
    if use_rmsnorm:
        hs = (normalized_shape,) if isinstance(normalized_shape, int) else normalized_shape
        return TritonRMSNorm(
            hs=hs,
            eps=eps,
            elementwise_affine=elementwise_affine,
            recompute=recompute,
        )
    return nn.LayerNorm(
        normalized_shape,
        eps=eps,
        elementwise_affine=elementwise_affine,
        device=device,
        dtype=dtype,
    )
