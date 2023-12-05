#ifndef SELECT_CONV_UTILS
#define SELECT_CONV_UTILS

#include "pytorch_cpp_helper.hpp"

void select_conv_shape_check(
    at::Tensor input, at::Tensor *gradOutput,
    at::Tensor weight, int kH, int kW, int dH, int dW,
    int padH, int padW, int dilationH, int dilationW, int group);

#endif  // SELECT_CONV_UTILS
