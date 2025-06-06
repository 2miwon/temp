import triton
import triton.language as tl
from mgp import empty


@triton.jit
def _conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
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
    input_batch_stride,
    input_channel_stride,
    input_h_stride,
    input_w_stride,
    weight_out_stride,
    weight_in_stride,
    weight_h_stride,
    weight_w_stride,
    output_batch_stride,
    output_channel_stride,
    output_h_stride,
    output_w_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    block_start = pid * BLOCK_SIZE
    output_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    total_outputs = batch_size * out_channels * output_height * output_width
    mask = output_offsets < total_outputs

    temp = output_offsets
    out_w = temp % output_width
    temp = temp // output_width
    out_h = temp % output_height
    temp = temp // output_height
    out_ch = temp % out_channels
    batch_id = temp // out_channels

    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for in_ch in range(in_channels):
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                in_h = out_h * stride_h - padding_h + kh
                in_w = out_w * stride_w - padding_w + kw

                # Bounds checking
                valid = (in_h >= 0) & (in_h < input_height) & (in_w >= 0) & (in_w < input_width) & mask

                input_offsets = (batch_id * input_batch_stride +
                               in_ch * input_channel_stride +
                               in_h * input_h_stride +
                               in_w * input_w_stride)

                weight_offsets = (out_ch * weight_out_stride + 
                                in_ch * weight_in_stride + 
                                kh * weight_h_stride + 
                                kw * weight_w_stride)

                input_val = tl.load(input_ptr + input_offsets, mask=valid, other=0.0)
                weight_val = tl.load(weight_ptr + weight_offsets, mask=mask, other=0.0)

                accumulator = tl.where(valid, accumulator + input_val * weight_val, accumulator)

    output_offsets_final = (batch_id * output_batch_stride + 
                           out_ch * output_channel_stride + 
                           out_h * output_h_stride + 
                           out_w * output_w_stride)

    tl.store(output_ptr + output_offsets_final, accumulator, mask=mask)


def triton_conv2d(input, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    batch_size, in_channels, input_height, input_width = input.shape
    out_channels, _, kernel_height, kernel_width = weight.shape

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(padding, int):
        padding_h = padding_w = padding
    else:
        padding_h, padding_w = padding

    output_height = (input_height + 2 * padding_h - kernel_height) // stride_h + 1
    output_width = (input_width + 2 * padding_w - kernel_width) // stride_w + 1

    output = empty((batch_size, out_channels, output_height, output_width), 
                  device=input.device, dtype=input.dtype)

    total_outputs = batch_size * out_channels * output_height * output_width

    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_outputs, BLOCK_SIZE),)

    _conv2d_kernel[grid](
        input, weight, output,
        batch_size, in_channels, out_channels,
        input_height, input_width, output_height, output_width,
        kernel_height, kernel_width, stride_h, stride_w, padding_h, padding_w,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_SIZE
    )

    return output