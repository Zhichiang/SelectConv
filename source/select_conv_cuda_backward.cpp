#include "select_conv_cuda.hpp"


void select_conv_backward_input(
    Tensor input, Tensor gradOutput, Tensor gradInput,
    Tensor weight, Tensor columns, int kW, int kH,
    int dW, int dH, int padW, int padH,
    int dilationW, int dilationH, int group, int im2col_step) {
    if (input.device().is_cuda()) {
    #ifdef MMCV_WITH_CUDA
        CHECK_CUDA_INPUT(input);
        // CHECK_CUDA_INPUT(offset);
        CHECK_CUDA_INPUT(gradOutput);
        CHECK_CUDA_INPUT(gradInput);
        // CHECK_CUDA_INPUT(gradOffset);
        CHECK_CUDA_INPUT(weight);
        CHECK_CUDA_INPUT(columns);
    #else
        AT_ERROR("SelectConv is not compiled with GPU support");
    #endif
    } else {
        CHECK_CPU_INPUT(input);
        // CHECK_CPU_INPUT(offset);
        CHECK_CPU_INPUT(gradOutput);
        CHECK_CPU_INPUT(gradInput);
        // CHECK_CPU_INPUT(gradOffset);
        CHECK_CPU_INPUT(weight);
        CHECK_CPU_INPUT(columns);
    }
    select_conv_shape_check(
        input, &gradOutput, weight, kH, kW, dH, dW,
        padH, padW, dilationH, dilationW, group);

    at::DeviceGuard guard(input.device());

    int batch = 1;
    if (input.ndimension() == 3) {
        // Force batch
        batch = 0;
        input = input.view({1, input.size(0), input.size(1), input.size(2)});
        // offset = offset.view({1, offset.size(0), offset.size(1), offset.size(2)});
        gradOutput = gradOutput.view(
            {1, gradOutput.size(0), gradOutput.size(1), gradOutput.size(2)});
    }

    long batchSize = input.size(0);
    long nInputPlane = input.size(1);
    long inputHeight = input.size(2);
    long inputWidth = input.size(3);

    long nOutputPlane = weight.size(0);

    long outputWidth =
        (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
    long outputHeight =
        (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

//   TORCH_CHECK((offset.size(0) == batchSize), 3, "invalid batch size of offset");
    gradInput = gradInput.view({batchSize, nInputPlane, inputHeight, inputWidth});
    columns = at::zeros(
        {nInputPlane * kW * kH, im2col_step * outputHeight * outputWidth},
        input.options());

    // change order of grad output
    gradOutput = gradOutput.view({batchSize / im2col_step, im2col_step,
                                    nOutputPlane, outputHeight, outputWidth});
    gradOutput.transpose_(1, 2);

    gradInput = gradInput.view({batchSize / im2col_step, im2col_step, nInputPlane,
                                inputHeight, inputWidth});
    input = input.view({batchSize / im2col_step, im2col_step, nInputPlane,
                        inputHeight, inputWidth});
    // gradOffset = gradOffset.view({batchSize / im2col_step, im2col_step,
    //                                 deformable_group * 2 * kH * kW, outputHeight,
    //                                 outputWidth});
//   offset =
//       offset.view({batchSize / im2col_step, im2col_step,
//                    deformable_group * 2 * kH * kW, outputHeight, outputWidth});

    for (int elt = 0; elt < batchSize / im2col_step; elt++) {
        // divide into groups
        columns = columns.view({group, columns.size(0) / group, columns.size(1)});
        weight = weight.view({group, weight.size(0) / group, weight.size(1),
                            weight.size(2), weight.size(3)});
        gradOutput = gradOutput.view(
            {gradOutput.size(0), group, gradOutput.size(1) / group,
            gradOutput.size(2), gradOutput.size(3), gradOutput.size(4)});

        for (int g = 0; g < group; g++) {
        columns[g] = columns[g].addmm_(weight[g].flatten(1).transpose(0, 1),
                                        gradOutput[elt][g].flatten(1), 0.0f, 1.0f);
        }

        columns =
            columns.view({columns.size(0) * columns.size(1), columns.size(2)});
        gradOutput = gradOutput.view(
            {gradOutput.size(0), gradOutput.size(1) * gradOutput.size(2),
            gradOutput.size(3), gradOutput.size(4), gradOutput.size(5)});

    // deformable_col2im_coord_impl(columns, input[elt], offset[elt], nInputPlane,
    //                              inputHeight, inputWidth, kH, kW, padH, padW,
    //                              dH, dW, dilationH, dilationW, im2col_step,
    //                              deformable_group, gradOffset[elt]);

        select_col2im_impl(columns, nInputPlane, inputHeight,
                            inputWidth, kH, kW, padH, padW, dH, dW, dilationH,
                            dilationW, im2col_step, gradInput[elt]);

        weight = weight.view({weight.size(0) * weight.size(1), weight.size(2),
                            weight.size(3), weight.size(4)});
    }

    gradOutput.transpose_(1, 2);
    gradOutput =
        gradOutput.view({batchSize, nOutputPlane, outputHeight, outputWidth});

    gradInput = gradInput.view({batchSize, nInputPlane, inputHeight, inputWidth});
    input = input.view({batchSize, nInputPlane, inputHeight, inputWidth});
    // gradOffset = gradOffset.view(
    //     {batchSize, deformable_group * 2 * kH * kW, outputHeight, outputWidth});
    // offset = offset.view(
    //     {batchSize, deformable_group * 2 * kH * kW, outputHeight, outputWidth});

    if (batch == 0) {
        gradOutput = gradOutput.view({nOutputPlane, outputHeight, outputWidth});
        input = input.view({nInputPlane, inputHeight, inputWidth});
        gradInput = gradInput.view({nInputPlane, inputHeight, inputWidth});
        // offset = offset.view({offset.size(1), offset.size(2), offset.size(3)});
        // gradOffset =
        //     gradOffset.view({offset.size(1), offset.size(2), offset.size(3)});
    }
}


void select_conv_backward_parameters(
    Tensor input, Tensor gradOutput, Tensor gradWeight,
    Tensor columns, Tensor ones, int kW,
    int kH, int dW, int dH, int padW, int padH,
    int dilationW, int dilationH, int group,
    float scale, int im2col_step) {
  if (input.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(input);
    // CHECK_CUDA_INPUT(offset);
    CHECK_CUDA_INPUT(gradOutput);
    CHECK_CUDA_INPUT(gradWeight);
    CHECK_CUDA_INPUT(columns);
    CHECK_CUDA_INPUT(ones);
#else
    AT_ERROR("SelectConv is not compiled with GPU support");
#endif
  } else {
    CHECK_CPU_INPUT(input);
    // CHECK_CPU_INPUT(offset);
    CHECK_CPU_INPUT(gradOutput);
    CHECK_CPU_INPUT(gradWeight);
    CHECK_CPU_INPUT(columns);
    CHECK_CPU_INPUT(ones);
  }

    select_conv_shape_check(
        input, &gradOutput, gradWeight, kH, kW, dH,
        dW, padH, padW, dilationH, dilationW, group);
  at::DeviceGuard guard(input.device());

  int batch = 1;

  if (input.ndimension() == 3) {
    // Force batch
    batch = 0;
    input = input.view(
        at::IntList({1, input.size(0), input.size(1), input.size(2)}));
    gradOutput = gradOutput.view(
        {1, gradOutput.size(0), gradOutput.size(1), gradOutput.size(2)});
  }

  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputHeight = input.size(2);
  long inputWidth = input.size(3);

  long nOutputPlane = gradWeight.size(0);

  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

//   TORCH_CHECK((offset.size(0) == batchSize), "invalid batch size of offset");

  columns = at::zeros(
      {nInputPlane * kW * kH, im2col_step * outputHeight * outputWidth},
      input.options());

  gradOutput = gradOutput.view({batchSize / im2col_step, im2col_step,
                                nOutputPlane, outputHeight, outputWidth});
  gradOutput.transpose_(1, 2);

  Tensor gradOutputBuffer = at::zeros_like(gradOutput);
  gradOutputBuffer =
      gradOutputBuffer.view({batchSize / im2col_step, nOutputPlane, im2col_step,
                             outputHeight, outputWidth});
  gradOutputBuffer = gradOutputBuffer.contiguous();
  gradOutputBuffer.copy_(gradOutput);
  gradOutputBuffer =
      gradOutputBuffer.view({batchSize / im2col_step, nOutputPlane,
                             im2col_step * outputHeight, outputWidth});

  gradOutput.transpose_(1, 2);
  gradOutput =
      gradOutput.view({batchSize, nOutputPlane, outputHeight, outputWidth});

  input = input.view({batchSize / im2col_step, im2col_step, nInputPlane,
                      inputHeight, inputWidth});
//   offset =
//       offset.view({batchSize / im2col_step, im2col_step,
//                    deformable_group * 2 * kH * kW, outputHeight, outputWidth});

  for (int elt = 0; elt < batchSize / im2col_step; elt++) {
    select_im2col_impl(
        input[elt], nInputPlane, inputHeight,
        inputWidth, kH, kW, padH, padW, dH, dW, dilationH,
        dilationW, im2col_step, columns);

    // divide into group
    gradOutputBuffer = gradOutputBuffer.view(
        {gradOutputBuffer.size(0), group, gradOutputBuffer.size(1) / group,
         gradOutputBuffer.size(2), gradOutputBuffer.size(3)});
    columns = columns.view({group, columns.size(0) / group, columns.size(1)});
    gradWeight =
        gradWeight.view({group, gradWeight.size(0) / group, gradWeight.size(1),
                         gradWeight.size(2), gradWeight.size(3)});

    for (int g = 0; g < group; g++) {
      gradWeight[g] = gradWeight[g]
                          .flatten(1)
                          .addmm_(gradOutputBuffer[elt][g].flatten(1),
                                  columns[g].transpose(1, 0), 1.0, scale)
                          .view_as(gradWeight[g]);
    }
    gradOutputBuffer = gradOutputBuffer.view(
        {gradOutputBuffer.size(0),
         gradOutputBuffer.size(1) * gradOutputBuffer.size(2),
         gradOutputBuffer.size(3), gradOutputBuffer.size(4)});
    columns =
        columns.view({columns.size(0) * columns.size(1), columns.size(2)});
    gradWeight = gradWeight.view({gradWeight.size(0) * gradWeight.size(1),
                                  gradWeight.size(2), gradWeight.size(3),
                                  gradWeight.size(4)});
  }

  input = input.view({batchSize, nInputPlane, inputHeight, inputWidth});
//   offset = offset.view(
//       {batchSize, deformable_group * 2 * kH * kW, outputHeight, outputWidth});

  if (batch == 0) {
    gradOutput = gradOutput.view({nOutputPlane, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputHeight, inputWidth});
  }
}

