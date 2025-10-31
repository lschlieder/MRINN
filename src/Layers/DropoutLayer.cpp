/*
 * DropoutLayer.cpp
 *
 *  Created on: May 9, 2018
 *      Author: lennart
 */
#include <boost/serialization/export.hpp>
#include "DropoutLayer.h"

namespace MRINN{
DropoutLayer::DropoutLayer(std::vector<int> inp_size, std::vector<int> out_size, float rate):Layer(inp_size, out_size) {

	// TODO Auto-generated constructor stub
	this->rate = rate;
	if ( rate > 1 || rate < 0 ){
		throw std::invalid_argument("Rate must be between 0 to 1");
	}

	connection_map = std::vector<int>(inp_num,1);

}

DropoutLayer::~DropoutLayer() {
	// TODO Auto-generated destructor stub
}

void DropoutLayer::process(){
	std::fill(connection_map.begin(),connection_map.end(), 1.0);
	for( int i_map = 0; i_map < input.getDimensionSize(1);i_map++){
		connection_map[i_map] = (rand()/(float)RAND_MAX) < rate;
	}
	for( int i_batch = 0; i_batch < input.getDimensionSize(0);i_batch++){
		const int batch_offset = i_batch* input.getDimensionSize(1);
		for( int i_inp = 0; i_inp < input.getDimensionSize(1); i_inp++){
			output.get_data()[batch_offset +i_inp] = connection_map[i_inp]*input.get_data()[batch_offset+i_inp];
		}
	}
}

void DropoutLayer::backpropagate(){
	for( int i_batch = 0; i_batch < error_inp.getDimensionSize(0); i_batch++){
		const int batch_offset = i_batch * error_inp.getDimensionSize(1);
		for( int i_inp = 0; i_inp < error_inp.getDimensionSize(1); i_inp++){
			error_out.get_data()[batch_offset + i_inp] = connection_map[i_inp] * error_inp.get_data()[batch_offset + i_inp];
		}
	}
}

}/* end namespace MRINN*/

//BOOST_CLASS_EXPORT(MRINN::DropoutLayer)

