#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "pytorch_cuda_helper.hpp"


template <typename T>
__global__ void select_col2im_gpu_kernel(
    const int n, const T *data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int batch_size,
    const int height_col, const int width_col, T *grad_im) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int j = (index / width_col / height_col / batch_size) % kernel_w;
    const int i =
        (index / width_col / height_col / batch_size / kernel_w) % kernel_h;
    const int c =
        index / width_col / height_col / batch_size / kernel_w / kernel_h;
    // compute the start and end of the output

    // const int deformable_group_index = c / channel_per_deformable_group;

    int w_out = index % width_col;
    int h_out = (index / width_col) % height_col;
    int b = (index / width_col / height_col) % batch_size;
    int w_in = w_out * stride_w - pad_w;
    int h_in = h_out * stride_h - pad_h;

    // const T *data_offset_ptr =
    //     data_offset + (b * deformable_group + deformable_group_index) * 2 *
    //                       kernel_h * kernel_w * height_col * width_col;
    // const int data_offset_h_ptr =
    //     ((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out;
    // const int data_offset_w_ptr =
    //     ((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out;
    // const T offset_h = data_offset_ptr[data_offset_h_ptr];
    // const T offset_w = data_offset_ptr[data_offset_w_ptr];
    const T cur_inv_h_data = h_in + i * dilation_h; //  + offset_h;
    const T cur_inv_w_data = w_in + j * dilation_w; //  + offset_w;

    const T cur_top_grad = data_col[index];
    const int cur_h = (int)cur_inv_h_data;
    const int cur_w = (int)cur_inv_w_data;
    for (int dy = -2; dy <= 2; dy++) {
      for (int dx = -2; dx <= 2; dx++) {
        if (cur_h + dy >= 0 && cur_h + dy < height && cur_w + dx >= 0 &&
            cur_w + dx < width && abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
            abs(cur_inv_w_data - (cur_w + dx)) < 1) {
          int cur_bottom_grad_pos =
              ((b * channels + c) * height + cur_h + dy) * width + cur_w + dx;
        //   T weight = get_gradient_weight(cur_inv_h_data, cur_inv_w_data,
        //                                  cur_h + dy, cur_w + dx, height, width);
          atomicAdd(grad_im + cur_bottom_grad_pos, cur_top_grad);
        }
      }
    }
  }
}



