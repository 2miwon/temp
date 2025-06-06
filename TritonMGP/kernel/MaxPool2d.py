import triton
import triton.language as tl
from mgp import empty


@triton.jit
def _maxpool2d_kernel(
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
    padding_h,
    padding_w,
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

    in_h_start = out_h * stride_h - padding_h
    in_w_start = out_w * stride_w - padding_w

    max_val = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)

    for kh in range(kernel_height):
        for kw in range(kernel_width):
            in_h = in_h_start + kh
            in_w = in_w_start + kw

            valid = (in_h >= 0) & (in_h < input_height) & (in_w >= 0) & (in_w < input_width)

            input_idx = (batch_id * channels * input_height * input_width +
                        channel_id * input_height * input_width +
                        in_h * input_width + in_w)

            input_val = tl.load(input_ptr + input_idx, mask=mask & valid, other=float('-inf'))
            max_val = tl.maximum(max_val, input_val)

    tl.store(output_ptr + element_id, max_val, mask=mask)


def triton_maxpool2d(input, kernel_size, stride, padding):
    batch_size, channels, input_height, input_width = input.shape
    kernel_height, kernel_width = kernel_size
    stride_h, stride_w = stride
    padding_h, padding_w = padding

    output_height = (input_height + 2 * padding_h - kernel_height) // stride_h + 1
    output_width = (input_width + 2 * padding_w - kernel_width) // stride_w + 1

    output = empty((batch_size, channels, output_height, output_width), 
                  device=input.device, dtype=input.dtype)

    BLOCK_SIZE = 256
    total_elements = batch_size * channels * output_height * output_width
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)

    _maxpool2d_kernel[grid](
        input, output,
        batch_size, channels, input_height, input_width,
        output_height, output_width, kernel_height, kernel_width,
        stride_h, stride_w, padding_h, padding_w, BLOCK_SIZE
    )

    return output