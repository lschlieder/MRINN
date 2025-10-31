/*
 * BinaryCrossentropy.cpp
 *
 *  Created on: Apr 17, 2018
 *      Author: lennart
 */

#include "BinaryCrossentropy.h"
#include "../VectorNd.h"
#include <cmath>

using namespace std;
namespace MRINN {
namespace Losses{
BinaryCrossentropy::BinaryCrossentropy() {
	// TODO Auto-generated constructor stub

}

BinaryCrossentropy::~BinaryCrossentropy() {
	// TODO Auto-generated destructor stub
}

float BinaryCrossentropy::get_loss(VectorNd<float> &output, VectorNd<float> &result){
	float res = 0;
	for( int i_batch = 0; i_batch < result.getDimensionSize(0); i_batch++){
		const int batch_offset = i_batch * result.getDimensionSize(1);
		/*for( int i_class = 0; i_class < result.getDimensionSize(1); i_class++){
			//std::cout <<std::endl <<  result.get_data()[batch_offset + i_class] << std::endl;
			float ln = log( std::max(result.get_data()[batch_offset+i_class], (float)1e-15));
			res = res + ln *(output.get_data()[batch_offset+i_class] );
			result.get_data()[batch_offset+i_class] = - output.get_data()[batch_offset+i_class]/result.get_data()[batch_offset + i_class];
		}*/
		res = output.get_data()[batch_offset] * log( std::max(result.get_data()[batch_offset],(float)1e-15)) + ( 1- output.get_data()[batch_offset]) * log(1- std::max(result.get_data()[batch_offset],(float)1e-15));
		result.get_data()[batch_offset] = - output.get_data()[batch_offset]/result.get_data()[batch_offset];
	}
	res = -res/result.getDimensionSize(0);

	return res;
}
} /* namespace Losses */
} /* namespace MRINN */
