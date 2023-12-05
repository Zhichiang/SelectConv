#include "pytorch_cpp_helper.hpp"
#include "select_conv_uitls.hpp"
#include "select_conv_cuda.cuh"


void select_conv_forward(
    Tensor input, Tensor weight,
    Tensor output, Tensor columns, Tensor ones, int kW,
    int kH, int dW, int dH, int padW, int padH,
    int dilationW, int dilationH, int group, int im2col_step);

void select_conv_backward_input(
    Tensor input, Tensor gradOutput, Tensor gradInput,
    Tensor weight, Tensor columns, int kW, int kH,
    int dW, int dH, int padW, int padH,
    int dilationW, int dilationH, int group, int im2col_step);

void select_conv_backward_parameters(
    Tensor input, Tensor gradOutput, Tensor gradWeight,
    Tensor columns, Tensor ones, int kW,
    int kH, int dW, int dH, int padW, int padH,
    int dilationW, int dilationH, int group,
    float scale, int im2col_step);

