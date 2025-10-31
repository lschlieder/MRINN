/*
 * ExpLayer.cpp
 *
 *  Created on: Jan 16, 2018
 *      Author: lennart
 */

#include "ExpLayer.h"
#include <cmath>
#include <stdexcept>
#include <boost/serialization/export.hpp>

namespace MRINN {

ExpLayer::ExpLayer( std::vector<int> inp_size, std::vector<int> out_size):Layer(inp_size,out_size) {
	if( inp_num != out_num|| inp_dim != out_dim){
		throw std::invalid_argument("input dimension/num is not equal to output dimension/num");
	}
}

void ExpLayer::process(){
	for( int i = 0 ; i < inp_num; i++){
		output.get_data()[i] = exp(-(input.get_data()[i]*input.get_data()[i]));
	}
}

void ExpLayer::backpropagate(){
	for( int i = 0; i < inp_num;i++){
		//float dtanh = tanh(input[i]);
		error_out.get_data()[i] = -exp(-(input.get_data()[i]*input.get_data()[i]))*2*input.get_data()[i] * error_inp.get_data()[i];
	}
}

ExpLayer::~ExpLayer() {
	// TODO Auto-generated destructor stub
}

} /* namespace MRINN */
//BOOST_CLASS_EXPORT(MRINN::ExpLayer)
