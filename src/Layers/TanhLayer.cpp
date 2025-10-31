/*
 * TanhLayer.cpp
 *
 *  Created on: Nov 20, 2017
 *      Author: lennart
 */
#include <boost/serialization/export.hpp>
#include "TanhLayer.h"
#include <math.h>
#include <stdexcept>
#include <vector>
#include <iostream>

namespace MRINN {
/*
 * Create TanhLayer. Throws exception if dimensions are not right (can only process input_size == output_size)
 */
TanhLayer::TanhLayer(std::vector<int> inp_size, std::vector<int> out_size): Layer(inp_size,out_size) {
	if( inp_num != out_num|| inp_dim != out_dim){
		throw std::invalid_argument("input dimension/num is not equal to output dimension/num");
	}
	//Jacobian = std::vector<double>(inp_num*out_num,0.0);
}

TanhLayer::~TanhLayer() {
	// TODO Auto-generated destructor stub
}


/*
 * calculates the tanh of every input and puts it in the corresponsing output;
 */
void TanhLayer::process(){
	#pragma omp parallel for
	for( int j=0; j < input.getDimensionSize(0);j++){
		const int batch_offset = j*input.getDimensionSize(1);
		#pragma omp parallel for
		for( int i = 0; i < inp_num; i++){
			output.get_data()[batch_offset+i] = tanh(input.get_data()[batch_offset + i]);
			//output(2,j,i) = tanh(input(2,j,i));
		}
	}
}

void TanhLayer::backpropagate(){
	/*for( int i = 0; i < inp_num;i++){
		for( int j = 0; j < out_num; j++){
			//std::cout << i << " " << j  << " " << i*out_num + j<< std::endl;
			if( i==j ){
			double dtanh = tanh(input[i]);
			Jacobian[i*out_num + i] = 1- dtanh*dtanh;
			} else{
				Jacobian[i*out_num+i] = 0.0;
		}
			//std::cout << 1- dtanh*dtanh;
		}
	}*/
	#pragma omp parallel for
	for( int j=0; j < error_inp.getDimensionSize(0); j++){
		const int batch_offset = j*error_inp.getDimensionSize(1);
		#pragma omp parallel for
		for( int i = 0; i < inp_num;i++){
			float dtanh = tanh(input.get_data()[batch_offset + i]);
			//error_out(2,j,i) = error_inp(2,j,i) * (1-dtanh*dtanh);
			error_out.get_data()[batch_offset+i] = error_inp.get_data()[batch_offset + i] * (1-dtanh*dtanh);
		}
	}
}

} /* namespace MRINN */
//BOOST_CLASS_EXPORT(MRINN::TanhLayer)
