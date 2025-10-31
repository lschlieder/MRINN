#ifndef cuda_conv
#define cuda_conv

#include <cuda.h>
#include "VectorNd.h"
#include <cudnn.h>
namespace MRINN{
void output_forward_convolution_bias(int in_n, int in_c, int in_h, int in_w, float * input, float *  output, int kernel_num, int kernel_size, float *  kernel,int bias_size, float *bias, cudnnHandle_t cudnn_handle);
void output_forward_convolution(int in_n, int in_c, int in_h, int in_w, float * input, float *  output, int kernel_num, int kernel_size, float *  kernel, cudnnHandle_t cudnn_handle);
void output_backward_convolution_data(int in_n, int in_c, int in_h, int in_w, float * grad_input, int out_n, int out_c, int out_h, int out_w, float * grad_output, int kernel_num, int kernel_size, float* kernel, cudnnHandle_t cudnn_handle);
void output_backward_convolution(float * input, float * output, VectorNd<float> kernel, cudnnHandle_t cudnn_handle);
void output_backward_convolution_parameters_bias(int in_n, int in_c, int in_h, int in_w, float * input, int grad_n, int grad_c, int grad_h, int grad_w, float * grad_input, int kernel_num,int kernel_size, int params_size, float * params,  cudnnHandle_t cudnn_handle);
void output_backward_convolution_parameters(int in_n, int in_c, int in_h, int in_w, float * input, int grad_n, int grad_c, int grad_h, int grad_w, float * grad_input, int kernel_num,int kernel_size, int params_size, float * params,  cudnnHandle_t cudnn_handle);
}

#endif
