/*
 * Layer.h
 *
 *  Created on: Nov 14, 2017
 *      Author: lennart
 */

#ifndef LAYER_H_
#define LAYER_H_

#include <vector>
#include "../VectorNd.h"
#include "string"
//#include <boost/archive/test_oarchive.hpp>
//#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
//#include <boost/serialization/export.hpp>

namespace MRINN{
class Layer;
}/* end namespace MRINN*/


namespace boost { namespace serialization {
template<class Archive>
inline void load_construct_data(
    Archive & ar, MRINN::Layer * t, const unsigned int file_version);
template<class Archive>
inline void save_construct_data(Archive & ar, const MRINN::Layer* t, const unsigned int file_version);
}} /*end namespace boost::serialization*/

namespace MRINN{
class Layer{
public:
	friend class boost::serialization::access;

	template<class Archive> friend void boost::serialization::save_construct_data(Archive & ar, const MRINN::Layer* t, const unsigned int file_version);
	template<class Archive> friend void boost::serialization::load_construct_data(Archive & ar, MRINN::Layer * t, const unsigned int file_version);

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & inp_num;
        ar & out_num;
        ar & inp_dim;
        ar & out_dim;
        ar & inp_size;
        ar & out_size;
        ar & input;
        ar & output;
        ar & error_inp;
        ar & error_out;
    	ar & gradient;
    }

	std::vector<int>  get_inp_size();
	std::vector<int> get_out_size();
	int get_output_num();
	int get_input_num();
	int get_input_dimension();
	int get_output_dimension();


	virtual void set_var_parameters( std::vector<float> &pars); // implemts the setter for the parameters that are to be learned
	virtual std::vector<float> & get_var_parameters(); 		//implements the getter for the parameters that are to be learned
	virtual int get_var_parameter_number(); // returns the number of variables that can be learned;
	virtual VectorNd<float> & get_parameter_gradient(VectorNd<float> & backprop_vec); // returns the gradient of the parameters for backpropagation

	virtual void calculate_jacobian();
	std::vector<float> get_jacobian();
	virtual VectorNd<float> & get_output();
	virtual VectorNd<float> & calc_output(VectorNd<float> & input);

	virtual VectorNd<float> & get_error();
	virtual VectorNd<float> & calc_error(VectorNd<float> & error_inp);

	virtual void print_weights();
	virtual void print_layer();
    
    virtual bool get_is_gpu_layer();

	Layer( std::vector<int> inp_size, std::vector<int> out_size);
	virtual ~Layer();
protected:
	int inp_num; // the number of inputs
	int out_num; // the number of outputs
	int inp_dim; // the number of input dimensions
	int out_dim; // the number of output dimensions
	std::vector<int> inp_size; // the size of the input dimensions
	std::vector<int> out_size; // the size of the output dimensions
	VectorNd<float> input;
	VectorNd<float> output;

	//std::vector<float> error_inp;
	//std::vector<float> error_out;
	VectorNd<float> error_inp;
	VectorNd<float> error_out;

    bool is_gpu_layer = false;
	std::vector<float> Jacobian;

	VectorNd<float> gradient; // This is allocated at the start once and again if batch_size_back changes for a backpropagation;
	virtual void backpropagate();
	virtual void process(); // takes this->input and does the needed calculations to get this->output / Override this for children classes
};


}/*namespace MRINN*/

/*
 * Overload the load operator from boost::serialization to execute the non emtpy constructor of this Layer
 */
namespace boost { namespace serialization {

template<class Archive>
inline void save_construct_data(
    Archive & ar, const MRINN::Layer * t, const unsigned int file_version
){
    // save data required to construct instance
    ar << t->inp_size;
    ar << t->out_size;
}
template<class Archive>
inline void load_construct_data(
    Archive & ar, MRINN::Layer * t, const unsigned int file_version
){
    // retrieve data from archive required to construct new instance
	std::vector<int> inp_size, out_size;
	ar >> inp_size;
	ar >> out_size;
    // invoke inplace constructor to initialize instance of my_class
    ::new(t)MRINN::Layer(inp_size,out_size);
}
}}

#endif /* LAYER_H_ */
