import torch


dtype_mapping = {
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
    'float16': torch.float16,
}


def calc_relative_error(tensor_ref, tensor_cmp, eps=1e-12):
    tensor_ref = tensor_ref.float()
    tensor_cmp = tensor_cmp.float()
    norm_diff = torch.norm((tensor_cmp - tensor_ref), p='fro')
    norm = torch.norm(tensor_ref, p='fro')
    return norm_diff / (norm + eps)


def assert_relative_error(
    name: str,
    tensor_ref: torch.Tensor | None,
    tensor_cmp: torch.Tensor | None,
    threshold: float,
):
    if tensor_ref is None and tensor_cmp is None:
        return
    assert tensor_ref is not None and tensor_cmp is not None, f'{name} grad mismatch: one side is None'
    rel_error = calc_relative_error(tensor_ref, tensor_cmp)
    print(f'{name} rel_error = {rel_error:.5e}')
    assert rel_error < threshold, f'invalid {name} rel_error {rel_error:.5e} (should be < {threshold:.5e})'


def print_tensor_diff(tensor_cmp, tensor_ref, name='tensor', topk=0):
    diff = (tensor_cmp - tensor_ref).abs()
    diff_flat = diff.view(-1)
    max_idx_flat = torch.argmax(diff_flat)
    max_idx_tensor = torch.unravel_index(max_idx_flat, diff.shape)
    max_idx = tuple(t.item() for t in max_idx_tensor)

    tensor_cmp_val = tensor_cmp[max_idx].item()
    tensor_ref_val = tensor_ref[max_idx].item()
    max_diff_val = diff[max_idx].item()
    print(f'{name} max diff: {max_diff_val:.6f} ({tensor_cmp_val:.6f} vs. {tensor_ref_val:.6f})')

    if topk > 1:
        print(f'{name} top-{topk} diffs')
        values, indices = torch.topk(diff_flat, topk)

        for i in range(topk):
            idx_tensor = torch.unravel_index(indices[i], diff.shape)
            idx = tuple(t.item() for t in idx_tensor)

            tensor_cmp_val = tensor_cmp[idx].item()
            tensor_ref_val = tensor_ref[idx].item()
            diff_val = diff[idx].item()

            print(f'top-{i+1} | diff: {diff_val:.6f} ({tensor_cmp_val:>.6f} vs. {tensor_ref_val:.6f})')


def benchmark(test_func, num_warmup=3, num_tests=20):
    # flush L2 with 256 MB data
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda')
    cache.zero_()

    # warmup
    for _ in range(num_warmup):
        test_func()

    # test
    tic_event = torch.cuda.Event(enable_timing=True) # ms
    toc_event = torch.cuda.Event(enable_timing=True)
    tic_event.record()
    for _ in range(num_tests):
        test_func()
    toc_event.record()
    torch.cuda.synchronize()

    time_s = (tic_event.elapsed_time(toc_event) / num_tests) * 1e-3
    return time_s
