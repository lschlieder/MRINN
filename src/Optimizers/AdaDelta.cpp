/*
 * AdaDelta.cpp
 *
 *  Created on: Mar 29, 2018
 *      Author: lennart
 */

#include "AdaDelta.h"
#include <stdlib.h>
#include <algorithm>
#include "mkl.h"
#include "iostream"
#include <cmath>

using namespace std;

namespace MRINN {
namespace Optimizers{
AdaDelta::AdaDelta(float learning_rate = 1.0, float momentum= 0.9) {
	// TODO Auto-generated constructor stub
	gradient_vector = std::vector<float>(0);
	gradient_squared = std::vector<float>(0);
	parameters_squared = std::vector<float>(0);
	parameters_momentum_vec = std::vector<float>(0);
	this->momentum = momentum;
	this->learning_rate = learning_rate;

}

AdaDelta::~AdaDelta() {
	// TODO Auto-generated destructor stub

}

void AdaDelta::set_size(int size){
	gradient_vector.resize(size);
	std::fill(gradient_vector.begin(),gradient_vector.end(),0.0);
	gradient_squared.resize(size);
	std::fill(gradient_squared.begin(), gradient_squared.end(),0.0);
	parameters_squared.resize(size);
	std::fill(parameters_squared.begin(), parameters_squared.end(),0.0);
	parameters_momentum_vec.resize(size);
	std::fill(parameters_momentum_vec.begin(), parameters_momentum_vec.end(),0.0);
}

vector<float> & AdaDelta::get_updated_parameters(vector<float> &parameters, vector<float> &gradient){
	// Calculate RMS(Grad)

	std::transform(gradient.begin(),gradient.end(),gradient.begin(), gradient_squared.begin(), std::multiplies<float>());   // gradient  = gradient^2

	cblas_sscal(gradient_vector.size(), momentum, &gradient_vector[0],1);			// graduent_vector = graduent_vector*momentum
	cblas_saxpy(gradient_vector.size(), (1-momentum), &gradient_squared[0],1, &gradient_vector[0],1); 						// gradient_vector = gradient_vector + (1-momentum)*gradient_squared;




	#pragma omp parallel for		//gradient_squared = sqrt(gradient_vector+epsilon)
	for( unsigned int i=0; i < gradient.size(); i++){
		gradient_squared[i] = sqrt(gradient_vector[i]+EPSILON);
		parameters_squared[i] = sqrt(parameters_momentum_vec[i]+EPSILON);
	}
	//std::cout << gradient.size() << " " << parameters_momentum_vec.size() << gradient_vector.size() << " " << gradient_squared.size() << std::endl;
	std::transform(parameters_squared.begin(), parameters_squared.end(), gradient_squared.begin(), gradient_squared.begin(), std::divides<float>());
	//std::cout << gradient[0];
	std::transform(gradient.begin(),gradient.end(), gradient_squared.begin(), gradient.begin(), std::multiplies<float>());
	//std::cout << gradient[0] << std::endl;


	//Calculate RMS(param) (with old parameters!!)

	std::transform(parameters.begin(), parameters.end(), parameters.begin(), parameters_squared.begin(), std::multiplies<float>());
	cblas_sscal(parameters_momentum_vec.size(), momentum, &parameters_momentum_vec[0],1); // parameters_momentum_vec = momentum*parameters_momentum_vec
	cblas_saxpy(parameters_momentum_vec.size(), (1-momentum), &parameters_squared[0],1, &parameters_momentum_vec[0],1 ); // parameters_momentum_vec = parameters_momentum_vec + (1-momentum)*parameters_squared;

	// update parameters
	cblas_saxpy( parameters.size(), -learning_rate, &gradient[0], 1.0, &parameters[0],1.0);

	return parameters;
}

}

} /* namespace MRINN */
