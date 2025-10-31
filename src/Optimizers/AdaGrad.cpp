/*
 * AdaGrad.cpp
 *
 *  Created on: Mar 22, 2018
 *      Author: lennart
 */

#include "AdaGrad.h"
#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include "mkl.h"



namespace MRINN {
	namespace Optimizers{
		AdaGrad::AdaGrad(float learning_rate) {
			this->learning_rate = learning_rate;
			gradient_vector = std::vector<float>(0);
			gradient_squared = std::vector<float>(0);
		}

		AdaGrad::~AdaGrad() {
			// TODO Auto-generated destructor stub
		}
		void AdaGrad::set_size(int size){
			gradient_vector.resize(size);
			std::fill(gradient_vector.begin(),gradient_vector.end(),0.0);
			gradient_squared.resize(size);
			std::fill(gradient_squared.begin(), gradient_squared.end(),0.0);
		}
		vector<float> & AdaGrad::get_updated_parameters(vector<float> &parameters, vector<float> &gradient){
			//std::cout << "test" << std::endl;
			std::transform(gradient.begin(),gradient.end(),gradient.begin(), gradient_squared.begin(), std::multiplies<float>());   // gradient  = gradient^2

			cblas_saxpy(gradient_vector.size(), 1.0, &gradient_squared[0],1, &gradient_vector[0],1); 								// gradient_vector = gradient_vector+ gradient

			#pragma omp parallel for
			for( int i=0; i < gradient.size(); i++){
				gradient_squared[i] = sqrt(gradient_vector[i]+EPSILON);
			}

			std::transform(gradient.begin(),gradient.end(), gradient_squared.begin(), gradient.begin(), std::divides<float>());


			cblas_saxpy( parameters.size(), -learning_rate, &gradient[0], 1.0, &parameters[0],1.0);
			//std::transform(gradient_vector.begin(), gradient_vector.end(), gradient_vector.begin(), gradient)

			return parameters;
		}
	}
} /* namespace MRINN */
