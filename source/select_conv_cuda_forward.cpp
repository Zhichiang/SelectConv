#include "select_conv_cuda.hpp"


void select_conv_forward(
    Tensor input, Tensor weight,
    Tensor output, Tensor columns, Tensor ones, int kW,
    int kH, int dW, int dH, int padW, int padH,
    int dilationW, int dilationH, int group, int im2col_step) {
    if (input.device().is_cuda()) {
    #ifdef WITH_CUDA
        CHECK_CUDA_INPUT(input);
        // CHECK_CUDA_INPUT(offset);
        CHECK_CUDA_INPUT(weight);
        CHECK_CUDA_INPUT(output);
        CHECK_CUDA_INPUT(columns);
        CHECK_CUDA_INPUT(ones);
    #else
        AT_ERROR("DeformConv is not compiled with GPU support");
    #endif
    } else {
        CHECK_CPU_INPUT(input);
        // CHECK_CPU_INPUT(offset);
        CHECK_CPU_INPUT(weight);
        CHECK_CPU_INPUT(output);
        CHECK_CPU_INPUT(columns);
        CHECK_CPU_INPUT(ones);
    }

    select_conv_shape_check(
        input, NULL, weight, kH, kW, dH, dW, padH,
        padW, dilationH, dilationW, group);
    at::DeviceGuard guard(input.device());

    int batch = 1;
    if (input.ndimension() == 3) {
        // Force batch
        batch = 0;
        input.unsqueeze_(0);
        // offset.unsqueeze_(0);
    }

    // todo: assert batchsize dividable by im2col_step

    long batchSize = input.size(0);
    long nInputPlane = input.size(1);
    long inputHeight = input.size(2);
    long inputWidth = input.size(3);

    long nOutputPlane = weight.size(0);

    long outputWidth =
        (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
    long outputHeight =
        (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

    // TORCH_CHECK((offset.size(0) == batchSize), "invalid batch size of offset");

    output = output.view({
        batchSize / im2col_step, im2col_step, nOutputPlane, outputHeight, outputWidth});
    columns = at::zeros(
        {nInputPlane * kW * kH, im2col_step * outputHeight * outputWidth},
        input.options());

    if (ones.ndimension() != 2 ||
        ones.size(0) * ones.size(1) < outputHeight * outputWidth) {
        ones = at::ones({outputHeight, outputWidth}, input.options());
    }

    input = input.view({
        batchSize / im2col_step, im2col_step, nInputPlane, inputHeight, inputWidth});
    // offset = offset.view({
    //     batchSize / im2col_step, im2col_step,
    //     deformable_group * 2 * kH * kW, outputHeight, outputWidth});

    Tensor output_buffer = at::zeros({
        batchSize / im2col_step, nOutputPlane,
        im2col_step * outputHeight, outputWidth}, output.options());

    output_buffer = output_buffer.view(
        {output_buffer.size(0), group, output_buffer.size(1) / group,
        output_buffer.size(2), output_buffer.size(3)});

    for (int elt = 0; elt < batchSize / im2col_step; elt++) {
        select_im2col_cuda(
            input[elt], nInputPlane, inputHeight,
            inputWidth, kH, kW, padH, padW, dH, dW, dilationH,
            dilationW, im2col_step, columns);

        columns = columns.view({group, columns.size(0) / group, columns.size(1)});
        weight = weight.view({group, weight.size(0) / group, weight.size(1),
                              weight.size(2), weight.size(3)});

        for (int g = 0; g < group; g++) {
          output_buffer[elt][g] = output_buffer[elt][g]
                                      .flatten(1)
                                      .addmm_(weight[g].flatten(1), columns[g])
                                      .view_as(output_buffer[elt][g]);
        }
        columns =
            columns.view({columns.size(0) * columns.size(1), columns.size(2)});
        weight = weight.view({weight.size(0) * weight.size(1), weight.size(2),
                              weight.size(3), weight.size(4)});
    }

    output_buffer = output_buffer.view(
        {output_buffer.size(0), output_buffer.size(1) * output_buffer.size(2),
        output_buffer.size(3), output_buffer.size(4)});

    output_buffer = output_buffer.view({batchSize / im2col_step, nOutputPlane,
                                        im2col_step, outputHeight, outputWidth});
    output_buffer.transpose_(1, 2);
    output.copy_(output_buffer);
    output = output.view({batchSize, nOutputPlane, outputHeight, outputWidth});

    input = input.view({batchSize, nInputPlane, inputHeight, inputWidth});
    // offset = offset.view(
    //     {batchSize, deformable_group * 2 * kH * kW, outputHeight, outputWidth});

    if (batch == 0) {
        output = output.view({nOutputPlane, outputHeight, outputWidth});
        input = input.view({nInputPlane, inputHeight, inputWidth});
        // offset = offset.view({offset.size(1), offset.size(2), offset.size(3)});
    }
}
