/*
 * ConvolutionalLayer.h
 *
 *  Created on: Dec 4, 2017
 *      Author: lennart
 */

#ifndef CONVOLUTIONALLAYER_H_
#define CONVOLUTIONALLAYER_H_


#include "Layer.h"
#include "mkl_dnn.h"


namespace MRINN{
class ConvolutionalLayer;
}/* end namespace MRINN*/


namespace boost { namespace serialization {
template<class Archive>
inline void load_construct_data( Archive & ar, MRINN::ConvolutionalLayer * t, const unsigned int file_version);
template<class Archive>
inline void save_construct_data(Archive & ar, const MRINN::ConvolutionalLayer* t, const unsigned int file_version);

}} /*end namespace boost::serialization*/
/*
 * Implements the mkl_dnn version of convolution
 *
 */


namespace MRINN {



class ConvolutionalLayer: public Layer {
public:
	// befriend serialization
	friend class boost::serialization::access;

	template<class Archive> friend void boost::serialization::save_construct_data(Archive & ar, const MRINN::ConvolutionalLayer* t, const unsigned int file_version);
	template<class Archive> friend void boost::serialization::load_construct_data(Archive & ar, MRINN::ConvolutionalLayer * t, const unsigned int file_version);

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
    	//ar & boost::serialization::base_object<Layer>(*this);
    	ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Layer);
    	ar & kernel_size;
    	ar & kernel_number;
    	ar & batch_size;
    	ar & dimension;
    	ar & border_method;
    	ar & kernels;
    	ar & bias;
    	ar & params;
    	ar & use_bias;
    }
	void set_var_parameters(std::vector<float> &pars);
	std::vector<float> & get_var_parameters();
	//std::vector<float> * get_var_parameters();
	//std::vector<double> get_var_parameters();
	//std::vector<float> get_parameter_gradient( std::vector<float> backprop_vec);
	VectorNd<float> & get_parameter_gradient(VectorNd<float> & backprop_vec);

	ConvolutionalLayer( std::vector<int> inp_size, std::vector<int> out_size, int kernel_size, int kernel_num, dnnBorder_t border_type, bool use_bias );
	virtual ~ConvolutionalLayer();

protected:
	int kernel_size;
	int kernel_number;
	int batch_size;
	size_t dimension;
	dnnBorder_t border_method;
	dnnPrimitiveAttributes_t attributes;
	dnnPrimitive_t conv_prim = NULL;
	dnnPrimitive_t conv_backwards_data = NULL;
	dnnPrimitive_t conv_backwards_kernel = NULL;
	dnnPrimitive_t conv_backwards_bias = NULL;

	dnnPrimitive_t conversion_user_to_input = NULL;
	dnnPrimitive_t conversion_output_to_user = NULL;
	dnnPrimitive_t conversion_user_to_kernels = NULL;
	dnnPrimitive_t conversion_user_to_bias = NULL;

	dnnPrimitive_t conversion_errorsrc_to_user = NULL;
	dnnPrimitive_t conversion_user_to_errordst = NULL;
	dnnPrimitive_t conversion_errorkernels_to_user= NULL;
	dnnPrimitive_t conversion_errorbias_to_user = NULL;




	bool use_bias;

	std::vector<float> kernels;
	std::vector<float> bias;
	std::vector<float> params;

	float* resConv1[dnnResourceNumber] = {0};
	void setup_convolution(int batch_number);

	void backpropagate();
	void process();
};

} /* namespace MRINN */

/*
 * Overload the load operator from boost::serialization to execute the non emtpy constructor of this Layer
 */
namespace boost { namespace serialization {

template<class Archive>
inline void save_construct_data(
    Archive & ar, const MRINN::ConvolutionalLayer * t, const unsigned int file_version){
    // save data required to construct instance
    ar << t->inp_size;
    ar << t->out_size;
    ar << t->kernel_size;
    ar << t->kernel_number;
    ar << t->border_method;
    ar << t->use_bias;
}
template<class Archive>
inline void load_construct_data(
    Archive & ar, MRINN::ConvolutionalLayer * t, const unsigned int file_version
){
    // retrieve data from archive required to construct new instance
	std::vector<int> inp_size, out_size;
    int kernel_size, kernel_number;
    dnnBorder_t border_method;
    bool use_bias;
    ar >> inp_size;
    ar >> out_size;
    ar >> kernel_size;
    ar >> kernel_number;
    ar >> border_method;
    ar >> use_bias;

    // invoke inplace constructor to initialize instance of my_class
    ::new(t)MRINN::ConvolutionalLayer( inp_size, out_size,  kernel_size,  kernel_number, border_method, use_bias);
}
}}


#endif /* CONVOLUTIONALLAYER_H_ */
