/*
 * SGD.cpp
 *
 *  Created on: Mar 21, 2018
 *      Author: lennart
 */

#include "SGD.h"
#include <vector>
#include "mkl.h"
#include <iostream>
#include <cmath>

namespace MRINN {
	namespace Optimizers{
		SGD::SGD(float learning_rate = 0.05, float momentum = 0.9, bool nesterov = false) {
			this->learning_rate = learning_rate;
			this->momentum = momentum;
			vec_momentum = vector<float>(0,0.0);
			old_momentum = vector<float>(0,0.0);
			this->nesterov = nesterov;
		}



		SGD::~SGD() {
			// TODO Auto-generated destructor stub
		}

		/*
		 * Updates the parameters with a standard momentum stochastic gradient descent scheme
		 *
		 * nesterov
		 * v_prev = v # back this up
		 * v = mu * v - learning_rate * dx # velocity update stays the same
		 * x += -mu * v_prev + (1 + mu) * v # position update changes form
		 *
		 *
		 * simple momentum
		 * v = mu*v - learning_rate *dx
		 * x += v
		 */
		vector<float> & SGD::get_updated_parameters(vector<float> &parameters, vector<float> &gradient){
			/*//std::cout << "SGD" << std::endl;
			for( int i=0; i < parameters.size(); i++){
				std::cout << parameters[i] << " ";
			}
			std::cout << std::endl;*/
			//cblas_saxpy(parameters.size(), -0.0001, &gradient[0], 1.0, &(parameters[0]), 1.0);
			if( nesterov)
				std::copy( vec_momentum.begin(), vec_momentum.end(), old_momentum.begin());


			cblas_sscal(vec_momentum.size(), momentum, &vec_momentum[0], 1.0);
			cblas_saxpy( gradient.size(), -learning_rate, &gradient[0], 1.0, &vec_momentum[0],1.0  );
			//std::cout << "test" << std::endl;
			if( nesterov){
				cblas_sscal( vec_momentum.size(), (1+momentum), &vec_momentum[0], 1.0);
				cblas_saxpy( old_momentum.size(), -momentum, &old_momentum[0], 1.0, &vec_momentum[0],1.0);
			}

			cblas_saxpy( parameters.size(), 1.0, &vec_momentum[0], 1.0, &parameters[0],1.0);

			return parameters;
		}

		/*
		 * Updates the parameter size and allocates new memory for the momentum vector
		 */
		void SGD::set_size(int size){
			vec_momentum.clear();
			vec_momentum.resize(size);
			old_momentum.clear();
			old_momentum.resize(size);
		}
	}
} /* namespace MRINN */
