/*
 * Convolution2DLayer.h
 *
 *  Created on: Dec 5, 2017
 *      Author: lennart
 */

#ifndef CONVOLUTION2DLAYER_H_
#define CONVOLUTION2DLAYER_H_

#include "Layer.h"
//#include <boost/serialization/access.hpp>
//#include <boost/serialization/export.hpp>
//#include <boost/archive/binary_iarchive.hpp>
//#include <boost/archive/binary_oarchive.hpp>

namespace MRINN{
class Convolution2DLayer;
}/* end namespace MRINN*/


namespace boost { namespace serialization {
template<class Archive>
inline void load_construct_data( Archive & ar, MRINN::Convolution2DLayer * t, const unsigned int file_version);
template<class Archive>
inline void save_construct_data(Archive & ar, const MRINN::Convolution2DLayer* t, const unsigned int file_version);

}} /*end namespace boost::serialization*/

namespace MRINN {



class Convolution2DLayer: public Layer {
public:
	enum crop_method{cutoff, wrap_around, zero_padded, valid};

	friend class boost::serialization::access;

	template<class Archive> friend void boost::serialization::save_construct_data(Archive & ar, const MRINN::Convolution2DLayer* t, const unsigned int file_version);
	template<class Archive> friend void boost::serialization::load_construct_data(Archive & ar, MRINN::Convolution2DLayer * t, const unsigned int file_version);


    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
    	//ar & boost::serialization::base_object<Layer>(*this);
    	ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Layer);
    	ar & kernel_size_x;
    	ar & kernel_size_y;
    	ar & kernel_number;
    	ar & c_method;
    	ar & backwards_c_method;
    	ar & kernel_padding_x;
    	ar & kernel_padding_y;
    	ar & kernels;
    	ar & rotated_kernels;

    	ar & input_matrix;
    	ar & error_matrix;
    	ar & gradient;
    	ar & temp_grad_vec;
    }

	void calculate_jacobian();
	void set_var_parameters(std::vector<float> & pars);
	std::vector<float> & get_var_parameters();
	void Convolution2D(float * inp, int inp_size_x, int inp_size_y, float * kernel, int kernel_size_x, int kernel_size_y, crop_method cm, int zero_padding_x, int zero_padding_y, float* result);


	VectorNd<float> & get_parameter_gradient(VectorNd<float> & backprop_vec);

	void print_weights();

	Convolution2DLayer(std::vector<int> inp_size, std::vector<int> out_size, int kernel_size, int kernel_number, crop_method cm);
	Convolution2DLayer(std::vector<int> inp_size, std::vector<int> out_size, int kernel_size_x, int kernel_size_y, int kernel_number, crop_method cm);
	virtual ~Convolution2DLayer();
protected:
	int kernel_size_x;
	int kernel_size_y;
	int kernel_number;

	crop_method c_method;
	crop_method backwards_c_method;
	int kernel_padding_x;
	int kernel_padding_y;

	VectorNd<float> kernels;
	VectorNd<float> rotated_kernels;
	VectorNd<float> input_matrix;
	VectorNd<float> error_matrix;

	VectorNd<float> gradient;

	std::vector<float> temp_grad_vec;
	void fill_input_matrix();
	void fill_error_matrix();
	void resize_error_matrix();
	void resize_input_matrix();
//std::vector<float> convolution(std::vector<float> inp);

	float partial_derivative( int out_x, int out_y, int kern_x, int kern_y, int i_pic, int i_batch);


	void process();
	void backpropagate();


};

} /* namespace MRINN */

/*
 * Overload the load operator from boost::serialization to execute the non emtpy constructor of this Layer
 */
namespace boost { namespace serialization {

template<class Archive>
inline void save_construct_data(
    Archive & ar, const MRINN::Convolution2DLayer * t, const unsigned int file_version){
    // save data required to construct instance
    ar << t->inp_size;
    ar << t->out_size;
    ar << t->kernel_size_x;
    ar << t->kernel_size_y;
    ar << t->kernel_number;
    ar << t->c_method;
}
template<class Archive>
inline void load_construct_data(
    Archive & ar, MRINN::Convolution2DLayer * t, const unsigned int file_version
){
    // retrieve data from archive required to construct new instance
	std::vector<int> inp_size, out_size;
    int kernel_size_x, kernel_size_y, kernel_number;
    ar >> inp_size;
    ar >> out_size;
    ar >> kernel_size_x ;
    ar >> kernel_size_y;
    ar >> kernel_number;
    MRINN::Convolution2DLayer::crop_method cm;
    ar >> cm;
    // invoke inplace constructor to initialize instance of my_class
    ::new(t)MRINN::Convolution2DLayer( inp_size, out_size,  kernel_size_x,  kernel_size_y,  kernel_number, cm);
}
}}

#endif /* CONVOLUTION2DLAYER_H_ */
