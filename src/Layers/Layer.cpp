/*
 * Layer.cpp
 *
 *  Created on: Nov 14, 2017
 *      Author: lennart
 */

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/export.hpp>
#include "Layer.h"
#include <vector>
#include <iostream>
#include <string>

namespace MRINN{

Layer::Layer( std::vector<int> inp_size, std::vector<int> out_size) {
	int inp_num = 1;
	int out_num = 1;
	for( unsigned int i =0; i < inp_size.size();i++){
		inp_num = inp_num*inp_size[i];
	}
	for ( unsigned int i = 0; i < out_size.size(); i++){
		out_num = out_num*out_size[i];
	}
	this->inp_num = inp_num;
	this->out_num = out_num;
	this->inp_dim = inp_size.size();
	this->out_dim = out_size.size();
	this->inp_size = inp_size;
	this->out_size = out_size;

	//this->output = std::vector<float>(out_num,0.0);
	this->output = VectorNd<float>(2,1,out_num);
	//this->input =  std::vector<float>(inp_num,0.0);
	this->input =  VectorNd<float>(2,1,inp_num);

	//this->error_inp = std::vector<float>(out_num,0.0);
	//this->error_out = std::vector<float>(inp_num,0.0);
	this->error_inp = VectorNd<float>(2,1,out_num);
	this->error_out = VectorNd<float>(2,1,inp_num);



	//Jacobian = std::vector<float>(inp_num*out_num,0.0);
	gradient = VectorNd<float>(2,0,0);
	//gradient = VectorNd<float>(2,1,1); gradient(2,0,0)= 0.0;
}

Layer::~Layer() {

}

/**
 * returns the number of variable parameters that can be learned
 */
int Layer::get_var_parameter_number(){
	std::vector<float> ptr = get_var_parameters();

	//if( ptr != NULL){
    //std::cout << ptr.size() << std::endl;
    
    return ptr.size();
	//}else{
	//	return 0;
	//}
	//return 0;
	//float* params  = this->get_var_parameters();
	//return sizeof(params)/sizeof(*params); // returns the size of the whole array divided by the size of one element
}

/**
* get the number of inputs this layer processes
*/
int Layer::get_input_num(){
	return this->inp_num;
}

/**
 * get the number of output this layer calculates
 */
int Layer::get_output_num(){
	return this->out_num;
}
/**
 * get the dimensions the output of this layer has
 */
int Layer::get_output_dimension(){
	return this->out_dim;
}

/**
 * get the dimensions the input of this layer has
 */
int Layer::get_input_dimension(){
	return this->inp_dim;
}

/**
 * get the size of the dimensions for the input
 */
std::vector<int> Layer::get_inp_size(){
	return this->inp_size;
}

/**
 * get the size of the dimensions of the output
 */
std::vector<int> Layer::get_out_size(){
	return this->out_size;
}

VectorNd<float> & Layer::get_output(){
	//std::vector<float> out;
	//return out;
	return output;
}

/**
 * Calculate the output of the layer. Dummy method. Overload this!!
 */
void Layer::process(){
	this->output = VectorNd<float>(1,0);
}

/**
 * Calculate the backpropagation error. Dummy method. Overload this!!
 */
void Layer::backpropagate(){
	this->error_out = VectorNd<float>(1,0);
}

/**
 * method to calculate the jacobian
 * overload this !!
 */
void Layer::calculate_jacobian(){

}

/**
 * returns a copy of the Jacobian
 */
std::vector<float> Layer::get_jacobian(){
	//calculate_jacobian();

	std::vector<float> res = this->Jacobian;
	return res;
}


VectorNd<float> & Layer::calc_output(VectorNd<float> & input){
	this->input = input;
	if( output.getDimensionSize(0) != input.getDimensionSize(0))
		output = VectorNd<float>(2,input.getDimensionSize(0),out_num);

	//std::cout << input.get_data().size() << " " << output.get_data().size() << std::endl;
	process();
	//calculate_jacobian();
	return this->output;
}
/**
 * returns the error vector for backpropagation
 */
VectorNd<float> & Layer::get_error(){
	return this->error_out;
}

/**
 * calculates and returns the error vector for backpropagation
 */
VectorNd<float> & Layer::calc_error(VectorNd<float> &error_inp){
	this->error_inp = error_inp;
	if( error_out.getDimensionSize(0) != error_inp.getDimensionSize(0))
		error_out = VectorNd<float>(2,error_inp.getDimensionSize(0), inp_num);
	backpropagate();
	return this->error_out;
}

void Layer::print_layer(){
	std::cout << "This layer has not implemented this method yet!"<< std::endl;
}

void Layer::print_weights(){
	std::cout << "This layer has not implemented this method yet!" << std::endl;
}

void Layer::set_var_parameters(std::vector<float> & pars){

}

std::vector<float>& Layer::get_var_parameters(){
	//std::vector<float> res();
	//return std::vector<float>;
    //std::vector<float>(0)
    std::vector<float> * res = new std::vector<float>(0);
	return (*res);
	//return nullptr;
	//return nullptr;
}

VectorNd<float> & Layer::get_parameter_gradient(VectorNd<float> & backprop_vec){
	//std::vector<float> ret(0,0.0);
	//VectorNd<float> ret(2,0,0);
	//ret.get_data() = std::vector<float>(0);
	return gradient;
}

bool Layer::get_is_gpu_layer(){
    return this->is_gpu_layer;
}




}
//BOOST_CLASS_EXPORT(MRINN::Layer)
