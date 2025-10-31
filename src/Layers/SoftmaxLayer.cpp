/*
 * SoftmaxLayer.cpp
 *
 *  Created on: Apr 12, 2018
 *      Author: lennart
 */
#include <boost/serialization/export.hpp>
#include "SoftmaxLayer.h"
#include <cmath>
#include "mkl.h"
#include <errno.h>
#include <algorithm>

namespace MRINN {

SoftmaxLayer::SoftmaxLayer(std::vector<int> inp_size, std::vector<int> out_size):Layer(inp_size, out_size){
	// TODO Auto-generated constructor stub

}

SoftmaxLayer::~SoftmaxLayer() {
	// TODO Auto-generated destructor stub
}

void SoftmaxLayer::process(){
	if(norm.size() != input.getDimensionSize(0)){
		norm = std::vector<double>(input.getDimensionSize(0));
	}
//#pragma omp parallel for
	for( int i_batch = 0; i_batch< input.getDimensionSize(0);i_batch++){
		float normalizer = 0;
		const int batch_offset = i_batch * input.getDimensionSize(1);
            
        float max_elem = *std::max_element(&input.get_data()[batch_offset], &input.get_data()[batch_offset+input.getDimensionSize(1)]);
		for( int i_inp = 0; i_inp < input.getDimensionSize(1); i_inp++){
            //if( input.get_data()[batch_offset+i_inp] > 32) 
            //    std::cout << input.get_data()[batch_offset+i_inp] << " " << exp(input.get_data()[batch_offset+i_inp]) << std::endl;
			//output.get_data()[batch_offset + i_inp] = exp(std::max(input.get_data()[batch_offset+i_inp], (float) -1e+16));
            output.get_data()[batch_offset + i_inp] = exp(input.get_data()[batch_offset+i_inp] - max_elem  );
			normalizer += output.get_data()[batch_offset+i_inp];
            if( errno == ERANGE){
                    std::cout << "ERANGE overflow error " << std::endl;
                    std::cout << input.get_data()[batch_offset+i_inp]<< std::endl;
                    std::cin.ignore();
                }
            /*if( !std::isfinite(normalizer) ) {
                    //std::cout << std::endl << "normalizer is inf" << std::endl;
                    normalizer = MAXFLOAT;
            }
            if( std::isnan(normalizer)){
                std::cout << std::endl << "normalizer is nan " << std::endl;
            }
            if(std::isnan(output.get_data()[batch_offset + i_inp])){
                std::cout << "output is nan " << exp(input.get_data()[batch_offset+i_inp]) << " " << input.get_data()[batch_offset+i_inp] << std::endl; 
                std::cin.ignore();
            }*/
		}
		norm[i_batch] = normalizer;
		cblas_sscal(output.getDimensionSize(1), 1/normalizer, &output.get_data()[batch_offset],1);
	}
}

void SoftmaxLayer::backpropagate(){
#pragma omp parallel for
	for( int i_batch = 0; i_batch < error_inp.getDimensionSize(0);i_batch++){
		const int batch_offset = i_batch*error_inp.getDimensionSize(1);
		for( int i_errorout = 0; i_errorout < error_out.getDimensionSize(1);i_errorout++){
			float err_out = 0;
			for( int i_errorinp = 0; i_errorinp < error_inp.getDimensionSize(1);i_errorinp++){
				//err_out += (-exp(input.get_data()[batch_offset+i_errorinp] +input.get_data()[batch_offset+i_errorout])/(norm[i_batch]*norm[i_batch]) + (i_errorinp==i_errorout)* output.get_data()[batch_offset + i_errorinp])* error_inp.get_data()[batch_offset+i_errorinp];
                err_out +=  output.get_data()[batch_offset+i_errorinp]*((int)(i_errorinp==i_errorout) - output.get_data()[batch_offset+i_errorout]) * error_inp.get_data()[batch_offset+i_errorinp];
                if( errno == ERANGE){
                    std::cout << "ERANGE overflow error " << std::endl;
                }
			}
			error_out.get_data()[batch_offset + i_errorout] = err_out;
		}
	}
}
} /* namespace MRINN */
//BOOST_CLASS_EXPORT(MRINN::SoftmaxLayer)
