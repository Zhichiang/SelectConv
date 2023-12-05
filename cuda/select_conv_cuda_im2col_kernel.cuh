#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "pytorch_cuda_helper.hpp"

template <typename T>
__device__ T select_im2col_selection(const T *input, const int data_width,
                                        const int height, const int width, T h,
                                        T w) {
  if (h <= -1 || height <= h || w <= -1 || width <= w) {
    return 0;
  }

  int h_low = floorf(h);
  int w_low = floorf(w);

  T val = 0;
  if (h_low >= 0 && w_low >= 0) val = input[h_low * data_width + w_low];

  return val;
}


template <typename T>
__global__ void select_im2col_gpu_kernel(
    const int n, const T *data_im, const int height,
    const int width, const int kernel_h, const int kernel_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int batch_size,
    const int num_channels, const int height_col,
    const int width_col, T *data_col) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    // index index of output matrix
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int b_col = (index / width_col / height_col) % batch_size;
    const int c_im = (index / width_col / height_col) / batch_size;
    const int c_col = c_im * kernel_h * kernel_w;

    // compute deformable group index
    // const int deformable_group_index = c_im / channel_per_deformable_group;

    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;
    T *data_col_ptr =
        data_col +
        ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
    const T *data_im_ptr =
        data_im + (b_col * num_channels + c_im) * height * width;
    // const T *data_offset_ptr =
    //     data_offset + (b_col * deformable_group + deformable_group_index) * 2 *
    //                       kernel_h * kernel_w * height_col * width_col;

    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        // const int data_offset_h_ptr =
        //     ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        // const int data_offset_w_ptr =
        //     ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col +
        //     w_col;
        // const T offset_h = data_offset_ptr[data_offset_h_ptr];
        // const T offset_w = data_offset_ptr[data_offset_w_ptr];
        T val = static_cast<T>(0);
        const T h_im = h_in + i * dilation_h; // + offset_h;
        const T w_im = w_in + j * dilation_w; // + offset_w;
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width)
          // val = deformable_im2col_bilinear(data_im_ptr, width, height, width,
          //                                  h_im, w_im);
          val = select_im2col_selection(
            data_im_ptr, width, height, width, h_im, w_im);
        *data_col_ptr = val;
        data_col_ptr += batch_size * height_col * width_col;
      }
    }
  }
}
