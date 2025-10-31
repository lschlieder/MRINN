/*
 * ReluLayer.cpp
 *
 *  Created on: Nov 20, 2017
 *      Author: lennart
 */
#include <boost/serialization/export.hpp>
#include "ReluLayer.h"
#include <stdexcept>
#include <math.h>

namespace MRINN {



ReluLayer::ReluLayer( std::vector<int> inp_size, std::vector<int> out_size):Layer(inp_size,out_size) {
	if( inp_num != out_num|| inp_dim != out_dim){
		throw std::invalid_argument("input dimension/num is not equal to output dimension/num");
	}
}

ReluLayer::~ReluLayer() {
	// TODO Auto-generated destructor stub
}

/**
 * calculates the jacobian matrix which is a unit matrix multiplied by the derivative of the sigmoid in this case
 */
/*void ReluLayer::calculate_jacobian(){
	for( int i = 0; i < inp_num;i++){
			if( input[i] > 0.0 ){
				Jacobian[i*out_num + i] = 1.0;
			}else{
				Jacobian[i*out_num + i] = 0.0;
			}

	}
}*/
void ReluLayer::process(){
	//int count = 0;
	//#pragma omp parallel for
	for( int j = 0; j < input.getDimensionSize(0); j++){
		const int batch_offset = j*input.getDimensionSize(1);
		//#pragma omp parallel for
		for( int i = 0 ; i < inp_num; i++){
			//output(j,i) = fmax(input(j,i),0.0);
			const int index = batch_offset + i;
			output.get_data()[ index] = fmax(input.get_data()[index],0.0);
		    /*if( input.get_data()[index] > 0.0){
		    	output.get_data()[index] = input.get_data()[index];
		    }else{
		    	output.get_data()[index] = 0;
		    }*/

		}
	}
	//count = count /(float)input.getDimensionSize(0);
	//std::cout << count << "/" << inp_num<< std::endl;
}

void ReluLayer::backpropagate(){
	//#pragma omp parallel for
	for( int j =0; j < error_inp.getDimensionSize(0); j++){
		const int batch_offset = j*error_inp.getDimensionSize(1);
		//#pragma omp parallel for
		for( int i = 0; i < inp_num;i++){
			const int index = batch_offset +i;
			if( input.get_data()[index] > 0.0){
				//error_out(2,j,i) = error_inp(2,j,i) ;
				error_out.get_data()[index] = error_inp.get_data()[index];
			}else{
				//error_out(2,j,i) = 0.0;
				error_out.get_data()[index] = 0.0;//error_inp.get_data()[index];
			}
		}
	}
}

} /* namespace MRINN */
//BOOST_CLASS_EXPORT(MRINN::ReluLayer)
