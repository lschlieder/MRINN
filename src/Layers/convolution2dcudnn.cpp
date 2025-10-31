/*
 * Copyright 2019 <copyright holder> <email>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <boost/serialization/export.hpp>
#include "convolution2dcudnn.h"
#include "../CudaConvolution.cuh"
#include <cudnn.h>
#include "../VectorNd.h"
#include <cmath>

#define CUDA_CALL(f) { \
  cudaError_t err = (f); \
  if (err != cudaSuccess) { \
    std::cout \
        << "    Error occurred: " << err << std::endl; \
    std::exit(1); \
  } \
}

#define CUDNN_CALL(f) { \
  cudnnStatus_t err = (f); \
  if (err != CUDNN_STATUS_SUCCESS) { \
    std::cout \
        << "    CUDNN Error occurred: " << err << std::endl; \
    std::exit(1); \
  } \
}

namespace MRINN{
    Convolution2DCudnn::Convolution2DCudnn( std::vector<int> inp_size, std::vector<int> out_size, int kernel_size, int kernel_num, bool use_bias ):Layer(inp_size, out_size){
             
        this->kernel_size = kernel_size;
        this->kernel_number = kernel_num;
        this->use_bias = use_bias;
        if( out_size[0] != kernel_number){
            throw std::invalid_argument("output dimension does not check with kernel number. out_dim[0] must be == kernel_num");
        }

        if( inp_size.size() != out_size.size()){
            throw std::invalid_argument("input and output dimensions do not match. There should be output dimension == input dimension");
        }

        /*for( int i = inp_size.size() -1; i > 0; i--){
            if( inp_size[i] != out_size[i]){
                throw std::invalid_argument("one of the output dimension has the wrong size. Please check that newside = oldside ");
            }
        }*/


        batch_size = 0;
        dimension = inp_size.size()+1;
        int kernel_arr_size = pow(kernel_size,dimension-2)*(inp_size[0] * out_size[0]);
        kernels = std::vector<float>( kernel_arr_size,1.0);
        if( use_bias){
            bias = std::vector<float>(kernel_number,0.0);
        }
        params = std::vector<float>(kernel_arr_size + use_bias* kernel_number,0.0);
        for( unsigned int i_params = 0; i_params < params.size();i_params++){
            if( i_params < kernels.size()){
                //kernels[i_params] =(((float)rand()/RAND_MAX) -0.5) *2.0* sqrt(6.0/(inp_num+out_num)) ;
                //kernels[i_params] = (((float)rand()/RAND_MAX))*2.0* sqrt(6.0/(inp_num+out_num)) ;
                kernels[i_params] =(((float)rand()/RAND_MAX) -0.5)/6.0 ;//*2*sqrt(6/9) ;
                //kernels[i_params] = 0.01;
               // kernels[i_params] =(((float)rand()/RAND_MAX) -0.5) /2 ;
                /*if( i_params == 0)
                    kernels[i_params] = 1.0;
                else
                    kernels[i_params] = 0.0;*/
                params[i_params] = kernels[i_params];
            }else{
                //bias[i_params-kernels.size()] = ((float)rand()/RAND_MAX -0.5)/10.0;
                //bias[i_params-kernels.size()] = 1.0;
                //std::cout << bias[i_params] << std::endl;
                bias[i_params - kernels.size()] = 0.0;
                //bias[i_params-kernels.size()] = 1.0;
                params[i_params] = bias[i_params-kernels.size()];
            }

        }
        /*for( int i= 0; i < params.size();i++){
            std::cout << params[i] ;
        }
        std::cin.ignore();*/
        gradient = VectorNd<float>(2,1,params.size());
    // int s_kernel = pow(kernel_size,dimension-2)* inp_size[0] * batch_number;
    // kernels = std::vector<float>(s_kernel);
        
    cudnnHandle_t cudnn_context;
    CUDNN_CALL(cudnnCreate(&cudnn_context));
    this->cudnn_handle = cudnn_context;
    //std::cout << "created handle" << std::endl;
    this->is_gpu_layer = true;
    //setup_convolution( input.get_dimensionSize(0), inp_size[0], inp_size[1], inp_size[2], kernel_number, kernel_size, &kernels[0], kernel_num, &bias[0]);
            
    }
	
    /*Convolution2DCudnn::setup_convolution(int in_n, int in_c, int in_h, int in_w,int kernel_num, int kernel_size,float * kernels,  int bias_size, float * bias){
        ///// input
        cudnnTensorDescriptor_t in_desc;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
        CUDNN_CALL(cudnnSetTensor4dDescriptor(
        in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        in_n, in_c, in_h, in_w));

        float *in_data;
        CUDA_CALL(cudaMalloc(
                &in_data, in_n * in_c * in_h * in_w * sizeof(float) ) );
        
        //// grad_input
        cudnnTensorDescriptor_t grad_in_desc;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
        CUDNN_CALL(cudnnSetTensor4dDescriptor(
        grad_in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        in_n, in_c, in_h, in_w));
        
        float * in_data;
        CUDA_CALL(cudaMalloc(
                &grad_in_data, in_n * in_c * in_h * in_w * sizeof(float) ) );
        //CUDA_CALL(cudaMemcpy( in_data, input, sizeof(float)*in_n*in_c*in_h*in_w, cudaMemcpyDefault) );
        ///// filter
        cudnnFilterDescriptor_t filt_desc;
        CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
        CUDNN_CALL(cudnnSetFilter4dDescriptor(
                filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                filt_k, filt_c, filt_h, filt_w));

        float *filt_data;
        CUDA_CALL(cudaMalloc(
            &filt_data, filt_k * filt_c * filt_h * filt_w * sizeof(float)));
        CUDA_CALL(cudaMemcpy( filt_data, kernel, sizeof(float)*filt_k * filt_c * filt_h * filt_w , cudaMemcpyDefault) );
        
        //////grad filter
        cudnnFilterDescriptor_t grad_filt_desc;
        CUDNN_CALL(cudnnCreateFilterDescriptor(&grad_filt_desc));
        CUDNN_CALL(cudnnSetFilter4dDescriptor(
                grad_filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                filt_k, filt_c, filt_h, filt_w));

        float *grad_filt_data;
        CUDA_CALL(cudaMalloc(
            &grad_filt_data, filt_k * filt_c * filt_h * filt_w * sizeof(float)));
        //CUDA_CALL(cudaMemcpy( grad_filt_data, kernel, sizeof(float)*filt_k * filt_c * filt_h * filt_w , cudaMemcpyDefault) );
        
        
        const int pad_h = kernel_size/2;
        const int pad_w = kernel_size/2;
        const int str_h = 1;
        const int str_w = 1;
        const int dil_h = 1;
        const int dil_w = 1;
        
        ////////////// forward convolution
        cudnnConvolutionDescriptor_t conv_desc;
        CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));

        CUDNN_CALL(cudnnSetConvolution2dDescriptor(
            conv_desc,
            pad_h, pad_w, str_h, str_w, dil_h, dil_w,
            CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));

        // output
        int out_n;
        int out_c;
        int out_h;
        int out_w;
        
        CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
            conv_desc, in_desc, filt_desc,
            &out_n, &out_c, &out_h, &out_w));
        ////// out description
        cudnnTensorDescriptor_t out_desc;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
        CUDNN_CALL(cudnnSetTensor4dDescriptor(
                out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                out_n, out_c, out_h, out_w));

        float *out_data;
        CUDA_CALL(cudaMalloc(
                &out_data, out_n * out_c * out_h * out_w * sizeof(float)));
        ////// grad out description
        cudnnTensorDescriptor_t grad_out_desc;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_out_desc));
        CUDNN_CALL(cudnnSetTensor4dDescriptor(
                grad_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                out_n, out_c, out_h, out_w));

        float *grad_out_data;
        CUDA_CALL(cudaMalloc(
                &grad_out_data, out_n * out_c * out_h * out_w * sizeof(float)))
        
        
        // algorithm
        cudnnConvolutionFwdAlgo_t forward_algo;
        CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(
                cudnn_handle,
                in_desc, filt_desc, conv_desc, out_desc,
                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));
        
          // workspace
        size_t forward_ws_size;
        CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
                cudnn_handle, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size));

        float *forward_ws_data;
        CUDA_CALL(cudaMalloc(&ws_data, ws_size));

          //create bias
        float *bias_data;
        CUDA_CALL(cudaMalloc(&bias_data, bias_size*sizeof(float)));
        cudnnTensorDescriptor_t bias_desc;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
        CUDNN_CALL(cudnnSetTensor4dDescriptor( bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, bias_size, 1, 1));
        CUDA_CALL(cudaMemcpy( bias_data, bias, sizeof(float)*bias_size , cudaMemcpyDefault) );
        
        //// grad bias
        float *grad_bias_data;
        CUDA_CALL(cudaMalloc(&grad_bias_data, bias_size*sizeof(float)));
        cudnnTensorDescriptor_t grad_bias_desc;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_bias_desc));
        CUDNN_CALL(cudnnSetTensor4dDescriptor( grad_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, bias_size, 1, 1));
        //CUDA_CALL(cudaMemcpy( bias_data, bias, sizeof(float)*bias_size , cudaMemcpyDefault) );
        
    }*/
    
    Convolution2DCudnn::~Convolution2DCudnn() {
	// TODO Auto-generated destructor stub
        //std::cout << "destroy Convo " << std::endl;
        cudnnDestroy(cudnn_handle);
    }
    
    void Convolution2DCudnn::set_var_parameters(std::vector<float> & pars){
        /*std::cout << std::endl;
        for ( int i =0; i < pars.size();i++){
            std::cout << pars[i] << std::endl;
        }
        std::cin.ignore();*/
        
        std::copy(pars.begin(), pars.begin()+kernels.size(), kernels.begin());
        if( use_bias){
            std::copy(pars.begin()+kernels.size(), pars.end(), bias.begin());
        }
    }
    
    int Convolution2DCudnn::get_kernel_number(){
        return this->kernel_number;
    }

    std::vector<float> & Convolution2DCudnn::get_var_parameters(){
        std::copy(kernels.begin(), kernels.end(), params.begin());
        if( use_bias){
        std::copy(bias.begin(), bias.end(), params.begin()+kernels.size());
        }
        return params;
        
    }
    
    void Convolution2DCudnn::set_cudnn_handle(cudnnHandle_t handle){
        this->cudnn_handle = handle;
    }
    
    
    VectorNd<float> & Convolution2DCudnn::get_parameter_gradient(VectorNd<float> & backprop_vec){
   // void output_backward_convolution_parameters_bias(int in_n, int in_c, int in_h, int in_w, float * input, int grad_n, int grad_c, int grad_h, int grad_w, float * grad_input, int kernel_num,int kernel_size, int params_size, float * params,  cudnnHandle_t cudnn_handle);
        
        if( use_bias){
            output_backward_convolution_parameters_bias(input.getDimensionSize(0), inp_size[0], inp_size[1], inp_size[2], &input.get_data()[0], output.getDimensionSize(0), out_size[0], out_size[1], out_size[2], &backprop_vec.get_data()[0], kernel_number, kernel_size, params.size(), &gradient.get_data()[0], cudnn_handle);
            //output_backward_convolution_parameters_bias(output.getDimensionSize(0), out_size[0], out_size[1], out_size[2], &backprop_vec.get_data()[0],input.getDimensionSize(0), inp_size[0], inp_size[1], inp_size[2], &input.get_data()[0] , kernel_number, kernel_size, params.size(), &gradient.get_data()[0], cudnn_handle);
        }else{
            output_backward_convolution_parameters(input.getDimensionSize(0), inp_size[0], inp_size[1], inp_size[2], &input.get_data()[0], output.getDimensionSize(0), out_size[0], out_size[1], out_size[2], &backprop_vec.get_data()[0], kernel_number, kernel_size, params.size(), &gradient.get_data()[0], cudnn_handle);
        }
        /*
        for( int i =0; i < gradient.get_data().size();i++){
            std::cout << gradient.get_data()[i] << " " ;
        }
        std::cout << "end_gradient" << std::endl;
        std::cin.ignore();*/
        return gradient;
    }

    void Convolution2DCudnn::process(){
        //std::cout << "process" << std::endl;
        if( use_bias){
            output_forward_convolution_bias(input.getDimensionSize(0), inp_size[0], inp_size[1], inp_size[2], &input.get_data()[0], &output.get_data()[0],kernel_number, kernel_size, &kernels[0], bias.size(), &bias[0] ,cudnn_handle);
        }else{
            output_forward_convolution(input.getDimensionSize(0), inp_size[0], inp_size[1], inp_size[2], &input.get_data()[0], &output.get_data()[0],kernel_number, kernel_size, &kernels[0] ,cudnn_handle);
        }
    }
    
    void Convolution2DCudnn::backpropagate(){
            output_backward_convolution_data(error_inp.getDimensionSize(0), out_size[0], out_size[1], out_size[2], &error_inp.get_data()[0], error_out.getDimensionSize(0), inp_size[0], inp_size[1], inp_size[2], &error_out.get_data()[0], kernel_number, kernel_size, &kernels[0], cudnn_handle); 
    }
    

}
