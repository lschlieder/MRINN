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

#ifndef CONVOLUTION2DCUDNN_H
#define CONVOLUTION2DCUDNN_H
#include "Layer.h"
#include <cuda.h>
#include <cudnn.h>

namespace MRINN{
    class Convolution2DCudnn;
}

namespace boost { namespace serialization {
template<class Archive>
inline void load_construct_data( Archive & ar, MRINN::Convolution2DCudnn * t, const unsigned int file_version);
template<class Archive>
inline void save_construct_data(Archive & ar, const MRINN::Convolution2DCudnn* t, const unsigned int file_version);

}} /*end namespace boost::serialization*/
/**
 * @todo write docs
 */

namespace MRINN{
    
class Convolution2DCudnn :  public Layer
{
public:
	// befriend serialization
	friend class boost::serialization::access;

	template<class Archive> friend void boost::serialization::save_construct_data(Archive & ar, const MRINN::Convolution2DCudnn* t, const unsigned int file_version);
	template<class Archive> friend void boost::serialization::load_construct_data(Archive & ar, MRINN::Convolution2DCudnn * t, const unsigned int file_version);

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
    	//ar & boost::serialization::base_object<Layer>(*this);
    	ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Layer);
    	ar & kernel_size;
    	ar & kernel_number;
    	ar & batch_size;
        //ar & cudnn_handle;
        ar & dimension;
        ar & use_bias;
    	ar & kernels;
    	ar & bias;
    	ar & params;

    }
	void set_var_parameters(std::vector<float> &pars);
	std::vector<float> & get_var_parameters();
	//std::vector<float> * get_var_parameters();
	//std::vector<double> get_var_parameters();
	//std::vector<float> get_parameter_gradient( std::vector<float> backprop_vec);
	VectorNd<float> & get_parameter_gradient(VectorNd<float> & backprop_vec);
    void set_cudnn_handle(cudnnHandle_t handle);
    int get_kernel_number();

	Convolution2DCudnn( std::vector<int> inp_size, std::vector<int> out_size, int kernel_size, int kernel_num, bool use_bias );
	virtual ~Convolution2DCudnn();

protected:
	int kernel_size;
	int kernel_number;
	int batch_size;
    size_t dimension;
    cudnnHandle_t cudnn_handle; 

	std::vector<float> kernels;
	std::vector<float> bias;
	std::vector<float> params;

    //bool setup_convolution(int in_n, int in_c, int in_h, int in_w,int kernel_num, int kernel_size, float* kernels, int bias_size, float *bias );
    
    ////////////////////////// CUDA STUFF
    /*cudnnTensorDescriptor_t in_desc, out_desc, grad_in_desc, grad_out_desc, filt_desc, bias_desc, grad_filt_desc, grad_bias_desc;
    
    float * in_data, out_data, grad_in_data, grad_out_data, filt_data, bias_data, grad_filt_data, grad_bias_data; 
    size_t ws_size;
    float * ws_data;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnConvolutionFwdAlgo_t algo;
    */
    
    
    ///////////////////////// CUDA STUFF_END
    

	bool use_bias;

	void backpropagate();
	void process();
};

}


/*
 * Overload the load operator from boost::serialization to execute the non emtpy constructor of this Layer
 */
namespace boost { namespace serialization {

template<class Archive>
inline void save_construct_data(
    Archive & ar, const MRINN::Convolution2DCudnn * t, const unsigned int file_version){
    // save data required to construct instance
    ar << t->inp_size;
    ar << t->out_size;
    ar << t->kernel_size;
    ar << t->kernel_number;
    //ar << t->cudnn_handle;
    ar << t->use_bias;

}
template<class Archive>
inline void load_construct_data(
    Archive & ar, MRINN::Convolution2DCudnn * t, const unsigned int file_version){
    // retrieve data from archive required to construct new instance
	std::vector<int> inp_size, out_size;
    int kernel_size, kernel_number;
    bool use_bias;
    //cudnnHandle_t handle;
    ar >> inp_size;
    ar >> out_size;
    ar >> kernel_size;
    ar >> kernel_number;
    //ar >> handle;
    ar >> use_bias;

    // invoke inplace constructor to initialize instance of my_class
    ::new(t)MRINN::Convolution2DCudnn( inp_size, out_size,  kernel_size,  kernel_number, use_bias);
}
}}

#endif // CONVOLUTION2DCUDNN_H
