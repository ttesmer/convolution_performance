#include "im2col.h"

void im2col_cpu(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    double* data_col) {
	
  const int channel_size = height * width;
  for (int fitting_height = 0; fitting_height <= (height - kernel_h); fitting_height += stride_h) {
    for (int fitting_width = 0; fitting_width <= (width - kernel_w); fitting_width += stride_w) {
      for (int channel = 0; channel < channels; channel++) {
        for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
          for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
            int input_row = fitting_height + kernel_row;
            int input_col = fitting_width + kernel_col;
            *(data_col++) = data_im[input_row * width + input_col + channel_size * channel];
          }
        }
      }
    }
  }
}

