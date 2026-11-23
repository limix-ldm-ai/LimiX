import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import triton
import triton.language as tl
from ..utils import calc_relative_error, dtype_mapping


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


class TritonQKNormOP(torch.autograd.Function):
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


class TritonQKNormOPCompiled(TritonQKNormOP):
    @staticmethod
    def forward(ctx, input, weight, eps, output):
        return TritonQKNormOP.forward(ctx, input, weight, eps, output)

    # NOTE: dgrad rel_error increases with torch.compile
    # NOTE: calling superclass staticmethod in torch.compile is forbidden under autograd.Function.backward
    @staticmethod
    # @torch.compile(backend='inductor')
    def backward(ctx, grad_output):
        return _backward_impl(ctx, grad_output)


class TritonQKNorm(nn.Module):
    def __init__(self, hs, eps, elementwise_affine=True, device=None, dtype=torch.float32, recompute=False):
        super(TritonQKNorm, self).__init__()
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
            return TritonQKNormOP.apply(input, self.weight, self.eps, output)
        else:
            return TritonQKNormOPCompiled.apply(input, self.weight, self.eps, output)

    def extra_repr(self):
        return 'hs=%s, eps=%s, elementwise_affine=%s, dtype=%s, recompute=%s' % (
            self.hs, self.eps, self.elementwise_affine, self.dtype, self.recompute
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='customized triton RMSNorm operator')
    parser.add_argument('--bs', type=int, default=2, help='batch size')
    parser.add_argument('--seq-len', type=int, default=4096, help='sequence length')
    parser.add_argument('--hs', type=int, default=192, help='hidden size')
    parser.add_argument('--eps', type=float, default=1e-5, help='epsilon')
    parser.add_argument('--dtype', type=str, default='float16', choices=dtype_mapping.keys())
    args = parser.parse_args()
    dtype = dtype_mapping[args.dtype]

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # define inputs
    # NOTE: add unit_test for large seq_len (bs * seq_len * hs > max(int32))
    input_ref = torch.randn((args.bs, args.seq_len, args.hs), dtype=dtype, device='cuda')
    input_ref.requires_grad_(True)
    input_cmp = input_ref.detach().clone().requires_grad_(True)
    print(f'input | size = {list(input_ref.size())}, dtype = {input_ref.dtype}, device = {input_ref.device}')

    # define modules
    rn_ref = nn.RMSNorm(normalized_shape=args.hs, eps=args.eps, elementwise_affine=True, dtype=torch.float32)
    rn_ref = rn_ref.cuda()
    rn_cmp = TritonQKNorm(hs=args.hs, eps=args.eps, elementwise_affine=True, dtype=torch.float32)
    rn_cmp = rn_cmp.cuda()

    # verify forward and backward correctness
    output_ref = rn_ref(input_ref).to(input_ref.dtype) # return FP32 by default
    output_cmp = rn_cmp(input_cmp).to(input_cmp.dtype)
    fprop_rel_error = calc_relative_error(output_ref, output_cmp)
    print(f'fprop rel_error = {fprop_rel_error:.5e}')
    assert fprop_rel_error < 1e-3, f'invalid fprop rel_error {fprop_rel_error:.5e} (shoule be < 1e-3)'

    output_ref.sum().backward()
    output_cmp.sum().backward()
    dgrad_rel_error = calc_relative_error(input_ref.grad, input_cmp.grad)
    wgrad_rel_error = calc_relative_error(rn_ref.weight.grad, rn_cmp.weight.grad)
    print(f'dgrad rel_error = {dgrad_rel_error:.5e}')
    print(f'wgrad rel_error = {wgrad_rel_error:.5e}')
    assert dgrad_rel_error < 1e-3, f'invalid dgrad rel_error {dgrad_rel_error:.5e} (shoule be < 1e-3)'
    assert wgrad_rel_error < 1e-3, f'invalid wgrad rel_error {wgrad_rel_error:.5e} (shoule be < 1e-3)'
