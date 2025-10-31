/*
 * Optimizer.cpp
 *
 *  Created on: Mar 21, 2018
 *      Author: lennart
 */

#include "Optimizer.h"
#include <iostream>

namespace MRINN {
	namespace Optimizers{
		Optimizer::Optimizer() {
			// TODO Auto-generated constructor stub
		
		}
		Optimizer::~Optimizer() {
			// TODO Auto-generated destructor stub
		}

		vector<float> & Optimizer::get_updated_parameters(vector<float> & parameters, vector<float> & gradient){
			return parameters;
		}
		void Optimizer::set_size(int size){

		}
	}
} /* namespace MRINN */
