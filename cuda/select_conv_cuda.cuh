#include "pytorch_cuda_helper.hpp"
#include "select_conv_cuda_im2col_kernel.cuh"
#include "select_conv_cuda_col2im_kernel.cuh"


void select_im2col_cuda(Tensor data_im,
                            const int channels, const int height,
                            const int width, const int ksize_h,
                            const int ksize_w, const int pad_h, const int pad_w,
                            const int stride_h, const int stride_w,
                            const int dilation_h, const int dilation_w,
                            const int parallel_imgs,
                            Tensor data_col);

void select_col2im_cuda(
    Tensor data_col, const int channels, const int height,
    const int width, const int ksize_h, const int ksize_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int parallel_imgs, Tensor grad_im);


