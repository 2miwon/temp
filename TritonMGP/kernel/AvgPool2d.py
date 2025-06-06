import triton
import triton.language as tl
from mgp import empty


@triton.jit
def _avgpool2d_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    input_height,
    input_width,
    output_height,
    output_width,
    kernel_height,
    kernel_width,
    stride_h,
    stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    total_elements = batch_size * channels * output_height * output_width
    element_id = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = element_id < total_elements

    batch_id = element_id // (channels * output_height * output_width)
    remainder = element_id % (channels * output_height * output_width)
    channel_id = remainder // (output_height * output_width)
    spatial_id = remainder % (output_height * output_width)
    out_h = spatial_id // output_width
    out_w = spatial_id % output_width

    in_h_start = out_h * stride_h
    in_w_start = out_w * stride_w

    sum_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    count = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for kh in range(kernel_height):
        for kw in range(kernel_width):
            in_h = in_h_start + kh
            in_w = in_w_start + kw

            valid = (in_h < input_height) & (in_w < input_width)

            input_idx = (batch_id * channels * input_height * input_width +
                        channel_id * input_height * input_width +
                        in_h * input_width + in_w)

            input_val = tl.load(input_ptr + input_idx, mask=mask & valid, other=0.0)
            sum_val = tl.where(valid, sum_val + input_val, sum_val)
            count = tl.where(valid, count + 1.0, count)

    avg_val = sum_val / count
    tl.store(output_ptr + element_id, avg_val, mask=mask)


def triton_avgpool2d(input, pool_size, stride):
    batch_size, channels, input_height, input_width = input.shape
    kernel_height, kernel_width = pool_size
    stride_h, stride_w = stride

    output_height = (input_height - kernel_height) // stride_h + 1
    output_width = (input_width - kernel_width) // stride_w + 1

    output = empty((batch_size, channels, output_height, output_width), 
                  device=input.device, dtype=input.dtype)

    BLOCK_SIZE = 256
    total_elements = batch_size * channels * output_height * output_width
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)

    _avgpool2d_kernel[grid](
        input, output,
        batch_size, channels, input_height, input_width,
        output_height, output_width, kernel_height, kernel_width,
        stride_h, stride_w, BLOCK_SIZE
    )

    return output