/*
 * SelfAssemblingSparseLayer.cpp
 *
 *  Created on: Apr 6, 2018
 *      Author: lennart
 */

#include "SelfAssemblingSparseLayer.h"
#include "DenseLayer.h"
#include <iostream>
#include <vector>
//#include <cblas.h>
#include <mkl.h>
#include <cstdlib>
#include <time.h>
#include "../VectorNd.h"

namespace MRINN {

SelfAssemblingSparseLayer::SelfAssemblingSparseLayer(std::vector<int> inp_size, std::vector<int> out_size): Layer(inp_size, out_size) {


	weight_matrix = VectorNd<float>(2,out_num,inp_num);
	connection_matrix = VectorNd<float>(2,out_num,inp_num);
	for( int i = 0; i < this->inp_num; i++){
		for( int j = 0; j < this->out_num; j++){
			weight_matrix(2,j,i) = (((float)rand()/RAND_MAX)-0.5)/10;
			if( rand()/RAND_MAX > 0.5){
				connection_matrix(2,j,i) = 1.0;
			}else{
				connection_matrix(2,j,i) = 0.0;
			}
		}

	}

	gradient = VectorNd<float>(2, 1, weight_matrix.get_data().size());

}
void SelfAssemblingSparseLayer::apply_mask(){


}
std::vector<float> & SelfAssemblingSparseLayer::get_var_parameters(){

	return weight_matrix.get_data();

}

void SelfAssemblingSparseLayer::set_var_parameters(std::vector<float> & pars){
	weight_matrix.get_data().swap(pars);
}
VectorNd<float> & SelfAssemblingSparseLayer::get_parameter_gradient(VectorNd<float> & backprop_vec){

	cblas_sgemm( CblasRowMajor, CblasTrans, CblasNoTrans, backprop_vec.getDimensionSize(1), input.getDimensionSize(1), backprop_vec.getDimensionSize(0), 1.0, &(backprop_vec.get_data()[0]), backprop_vec.getDimensionSize(1), &input.get_data()[0], input.getDimensionSize(1), 0.0,&gradient.get_data()[0], input.getDimensionSize(1));

	return gradient;

}

void SelfAssemblingSparseLayer::process(){
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, input.getDimensionSize(0), out_num, inp_num, 1.0,&(input.get_data()[0]),inp_num, &(weight_matrix.get_data())[0], inp_num, 0.0, &(output.get_data()[0]), out_num );
}


void SelfAssemblingSparseLayer::backpropagate(){
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, error_inp.getDimensionSize(0), inp_num, out_num, 1.0, &error_inp.get_data()[0], out_num, &weight_matrix.get_data()[0], inp_num, 0.0, &(error_out.get_data())[0], inp_num);
}

SelfAssemblingSparseLayer::~SelfAssemblingSparseLayer() {
	// TODO Auto-generated destructor stub
}

} /* namespace MRINN */
