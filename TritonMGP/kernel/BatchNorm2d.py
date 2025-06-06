import triton
import triton.language as tl
from mgp import empty


@triton.jit
def _batchnorm2d_kernel(
    input_ptr, 
    output_ptr, 
    weight_ptr, 
    bias_ptr,
    running_mean_ptr, 
    running_var_ptr,
    N, C, H, W,
    stride_n, stride_c, stride_h, stride_w,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid_c = tl.program_id(0)  # Channel dimension
    pid_batch = tl.program_id(1)  # Batch dimension
    pid_spatial = tl.program_id(2)  # Spatial dimension

    if pid_c >= C or pid_batch >= N:
        return

    running_mean = tl.load(running_mean_ptr + pid_c)
    running_var = tl.load(running_var_ptr + pid_c)
    inv_std = 1.0 / tl.sqrt(running_var + eps)

    weight = tl.load(weight_ptr + pid_c) if weight_ptr else 1.0
    bias = tl.load(bias_ptr + pid_c) if bias_ptr else 0.0

    spatial_size = H * W
    elements_per_block = BLOCK_SIZE
    start_idx = pid_spatial * elements_per_block
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < spatial_size

    h = offsets // W
    w = offsets % W

    input_offsets = (pid_batch * stride_n + pid_c * stride_c + 
                    h * stride_h + w * stride_w)

    x = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)

    normalized = (x - running_mean) * inv_std
    result = normalized * weight + bias

    tl.store(output_ptr + input_offsets, result, mask=mask)


def triton_bn2d(
    input_tensor, 
    weight, 
    bias, 
    running_mean, 
    running_var, 
    momentum, 
    eps
):
    N, C, H, W = input_tensor.shape

    output = empty((N, C, H, W), device=input_tensor.device, dtype=input_tensor.dtype)

    spatial_blocks = triton.cdiv(H * W, 256)
    grid = (C, N, spatial_blocks)

    _batchnorm2d_kernel[grid](
        input_tensor, output,
        weight if weight is not None else None,
        bias if bias is not None else None, 
        running_mean, running_var,
        N, C, H, W,
        input_tensor.stride(0), input_tensor.stride(1),
        input_tensor.stride(2), input_tensor.stride(3),
        eps,
        BLOCK_SIZE=256
    )

    return output