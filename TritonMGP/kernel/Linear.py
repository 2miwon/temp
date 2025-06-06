import triton
import triton.language as tl
from mgp import empty


@triton.jit
def _linear_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    M, N, K,
    stride_im,
    stride_ik,
    stride_wk,
    stride_wn,
    stride_om,
    stride_on,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_im = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_wn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    input_ptrs = input_ptr + (offs_im[:, None] * stride_im + offs_k[None, :] * stride_ik)
    weight_ptrs = weight_ptr + (offs_k[:, None] * stride_wk + offs_wn[None, :] * stride_wn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        input_val = tl.load(input_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        weight_val = tl.load(weight_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(input_val, weight_val)
        input_ptrs += BLOCK_SIZE_K * stride_ik
        weight_ptrs += BLOCK_SIZE_K * stride_wk
    output_val = accumulator.to(tl.float16)

    offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_ptrs = output_ptr + stride_om * offs_om[:, None] + stride_on * offs_on[None, :]
    output_mask = (offs_om[:, None] < M) & (offs_on[None, :] < N)
    tl.store(output_ptrs, output_val, mask=output_mask)


@triton.jit
def _add_bias_kernel(
    output_ptr,
    bias_ptr,
    M,
    N,
    stride_om,
    stride_on,
    stride_bias,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_elements = M * N
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    row = offsets // N
    col = offsets % N

    output_ptrs = output_ptr + row * stride_om + col * stride_on
    bias_ptrs = bias_ptr + col * stride_bias

    output_val = tl.load(output_ptrs, mask=mask, other=0.0)
    bias_val = tl.load(bias_ptrs, mask=mask & (col < N), other=0.0)

    output_val = output_val + bias_val
    tl.store(output_ptrs, output_val, mask=mask)


def triton_linear(x, weight, bias=None):
    original_shape = x.shape
    if x.dim() > 2:
        x = x.view(-1, x.shape[-1])

    M, K = x.shape  # M = batch_size * other_dims, K = in_features
    _, N = weight.shape

    output = empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']), )

    _linear_kernel[grid](
        x, weight, output,  
        M, N, K,  
        x.stride(0), x.stride(1),  
        weight.stride(0), weight.stride(1),  
        output.stride(0), output.stride(1),  
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
    )

    if bias is not None:
        grid_bias = lambda meta: (triton.cdiv(M * N, meta['BLOCK_SIZE']), )
        _add_bias_kernel[grid_bias](
            output,
            bias,
            M,
            N,
            output.stride(0),
            output.stride(1),
            bias.stride(0),
            BLOCK_SIZE=1024
        )

    if len(original_shape) > 2:
        output_shape = original_shape[:-1] + (N,)
        output = output.view(output_shape)

    return output