/*
 * Copyright 2019 <copyright holder> <email>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <boost/serialization/export.hpp>
#include "sparse_softmax_crossentropy.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <errno.h>

#include "mkl.h"

namespace MRINN {
namespace Losses{

Sparse_Softmax_Crossentropy::Sparse_Softmax_Crossentropy(){
    //std::cout << "creating sparse crossent" << std::endl;
    probabilities = VectorNd<float>(2,0,0);
}

Sparse_Softmax_Crossentropy::~Sparse_Softmax_Crossentropy(){
    probabilities;
}


float Sparse_Softmax_Crossentropy::get_loss(VectorNd<float> &output, VectorNd<float> &result){
	//std::cout << "test234" << std::endl;
    //std::cout << probabilities.getDimensionSize(0)  << " " <<result.getDimensionSize(0)  << std::endl;
    if(probabilities.getDimensionSize(0) != result.getDimensionSize(0)){
		//norm = std::vector<double>(input.getDimensionSize(0));
        probabilities = VectorNd<float>(2,result.getDimensionSize(0),result.getDimensionSize(1));
        	//std::cout << "test" << std::endl;
	}
		//std::cout << "test" << std::endl;
    for( int i_batch = 0; i_batch< result.getDimensionSize(0);i_batch++){
		float normalizer = 0;
		const int batch_offset = i_batch * result.getDimensionSize(1);
            
        float max_elem = *std::max_element(&result.get_data()[batch_offset], &result.get_data()[batch_offset+result.getDimensionSize(1)]);
		for( int i_inp = 0; i_inp < result.getDimensionSize(1); i_inp++){
            //if( input.get_data()[batch_offset+i_inp] > 32) 
            //    std::cout << input.get_data()[batch_offset+i_inp] << " " << exp(input.get_data()[batch_offset+i_inp]) << std::endl;
			//output.get_data()[batch_offset + i_inp] = exp(std::max(input.get_data()[batch_offset+i_inp], (float) -1e+16));
            probabilities.get_data()[batch_offset + i_inp] = exp(result.get_data()[batch_offset+i_inp] - max_elem  );
            
			normalizer += probabilities.get_data()[batch_offset+i_inp];
            if( errno == ERANGE){
                    std::cout << "ERANGE overflow error " << std::endl;
                    std::cout << result.get_data()[batch_offset+i_inp]<< std::endl;
                    std::cin.ignore();
                }
		}
		//norm[i_batch] = normalizer;
		cblas_sscal(probabilities.getDimensionSize(1), 1/normalizer, &probabilities.get_data()[batch_offset],1);
       /* for( int i_inp = 0; i_inp < result.getDimensionSize(1); i_inp++){
        std::cout <<  probabilities.get_data()[batch_offset + i_inp] << std::endl; 
        }*/
	}

	float res = 0;
	for( int i_batch = 0; i_batch < probabilities.getDimensionSize(0); i_batch++){
		const int batch_offset = i_batch * probabilities.getDimensionSize(1);
		for( int i_class = 0; i_class < probabilities.getDimensionSize(1); i_class++){
			//std::cout <<std::endl <<  result.get_data()[batch_offset + i_class] << std::endl;
			float ln = log( std::max(probabilities.get_data()[batch_offset+i_class],(float) 1e-16 ));
			res = res + ln *(output.get_data()[batch_offset+i_class] );
            
			//result.get_data()[batch_offset+i_class] = - (output.get_data()[batch_offset+i_class]/std::max(result.get_data()[batch_offset + i_class], (float)1e-16));
            result.get_data()[batch_offset+i_class] = (probabilities.get_data()[batch_offset + i_class] - output.get_data()[batch_offset+i_class])/(probabilities.getDimensionSize(0));
            
            //std::cout << result.get_data()[batch_offset+i_class] << std::endl;
            if( errno == EDOM){
                    std::cout << "EDOM domain error in log in crossentropy " << std::endl;
                }
            if( !std::isfinite(res) ) {
                    std::cout << std::endl << "res is inf in crossentropy " << probabilities.get_data()[batch_offset+i_class] << " "  << ln << " " << output.get_data()[batch_offset+i_class]<< std::endl;
                    
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
