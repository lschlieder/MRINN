/*
 * CrossEntropy.cpp
 *
 *  Created on: Apr 5, 2018
 *      Author: lennart
 */

#include "CrossEntropy.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <errno.h>

namespace MRINN {
namespace Losses{
CrossEntropy::CrossEntropy() {
	// TODO Auto-generated constructor stub

}

CrossEntropy::~CrossEntropy() {
	// TODO Auto-generated destructor stub
}

float CrossEntropy::get_loss(VectorNd<float> &output, VectorNd<float> &result){
	//std::cout << "test" << std::endl;
	float res = 0;
	for( int i_batch = 0; i_batch < result.getDimensionSize(0); i_batch++){
		const int batch_offset = i_batch * result.getDimensionSize(1);
		for( int i_class = 0; i_class < result.getDimensionSize(1); i_class++){
			//std::cout <<std::endl <<  result.get_data()[batch_offset + i_class] << std::endl;
			float ln = log( std::max(result.get_data()[batch_offset+i_class],(float) 1e-16 ));
			res = res + ln *(output.get_data()[batch_offset+i_class] );
            
			result.get_data()[batch_offset+i_class] = - (output.get_data()[batch_offset+i_class]/std::max(result.get_data()[batch_offset + i_class], (float)1e-16));

            if( errno == EDOM){
                    std::cout << "EDOM domain error in log in crossentropy " << std::endl;
                }
            if( !std::isfinite(res) ) {
                    std::cout << std::endl << "res is inf in crossentropy " << result.get_data()[batch_offset+i_class] << " "  << ln << " " << output.get_data()[batch_offset+i_class]<< std::endl;
                    
                    res = -MAXFLOAT;
            }
            if( std::isnan(res)){
                std::cout << std::endl << "res is nan crossentropy" << std::endl;
            }
		}
	}
    if( !std::isfinite(res) ) {
        std::cout << std::endl << "res is inf in crossentropy " << std::endl;
        res = -MAXFLOAT;
    }

    /*for( int i =0; i < result.get_data().size();i++){
        if( std::isnan(result.get_data()[i])){
            std::cout << "result vector is nan for crossentropy " << std::endl;
            std::cin.ignore();
        }
    }*/
    
    res = -res/result.getDimensionSize(0);
    //res = -res;
    
	return res;

}


} /* namespace Losses */
} /* namespace MRINN */
