/*
 * Adam.cpp
 *
 *  Created on: Apr 17, 2018
 *      Author: lennart
 */

#include "Adam.h"
#include "mkl.h"
#include <cmath>
#include <iostream>
#include <cmath>

namespace MRINN {
namespace Optimizers{
Adam::Adam(double learning_rate = 0.001, double momentum_one = 0.9, double momentum_two = 0.999) {
	this->momentum_one = momentum_one;
	this->momentum_two = momentum_two;
	this->learning_rate = learning_rate;
	// TODO Auto-generated constructor stub

	momentum_vec_one = vector<float>(0);
	momentum_vec_two = vector<float>(0);
	gradient_squared = vector<float>(0);

}

Adam::~Adam() {
	// TODO Auto-generated destructor stub
}

void Adam::set_size(int size){
	momentum_vec_one.resize(size);
	momentum_vec_two.resize(size);
	gradient_squared.resize(size);
	std::fill(momentum_vec_one.begin(),momentum_vec_one.end(),0.0);
	std::fill(momentum_vec_two.begin(),momentum_vec_two.end(),0.0);
	std::fill(gradient_squared.begin(), gradient_squared.end(),0.0);

	momentum_one_power_t = 1;
	momentum_two_power_t = 1;
}

vector<float> & Adam::get_updated_parameters(vector<float> & parameters, vector<float> & gradient){

	//std::cout << std::endl <<  gradient[0] << " " << parameters[0] << std::endl;
     //std::cout << parameters.size() << " " << gradient.size() << std::endl;
	#pragma omp parallel for
	for( unsigned int i=0; i < gradient.size(); i++){
		gradient_squared[i] = gradient[i]*gradient[i];
		//gradient[i] = gradient[i];
		momentum_vec_one[i] = momentum_one * momentum_vec_one[i];
		momentum_vec_two[i] = momentum_two * momentum_vec_two[i];
	}



	//cblas_saxpy(gradient.size(), )
	cblas_saxpy(gradient.size(),(1-momentum_one), &gradient[0], 1, &momentum_vec_one[0], 1);
	cblas_saxpy(gradient.size(),(1-momentum_two), &gradient_squared[0], 1, &momentum_vec_two[0], 1);
/*
    for( int i =0 ; i < momentum_vec_two.size();i++){
        std::cout << gradient_squared[i]<< " " << (1-momentum_two)*gradient_squared[0] << " " << ((double)1.0-momentum_two) <<" " <<  (1.0-(float)0.999) << std::endl;
    }*/
    
    momentum_one_power_t = momentum_one_power_t * momentum_one;
	momentum_two_power_t = momentum_two_power_t * momentum_two;
	float alpha_t = learning_rate * sqrt( 1 -  momentum_two_power_t )/(1-momentum_one_power_t);
    
	#pragma omp parallel for
	for ( unsigned int i = 0; i < gradient.size();i++){
		gradient[i] = alpha_t * momentum_vec_one[i]/( sqrt(momentum_vec_two[i]) + EPSILON);
        
		//std::cout << parameters[i] << std::endl;
	}
	cblas_saxpy( parameters.size(), -1.0, &gradient[0],1, &parameters[0],1 );
/*
    std::cout << "begin adam output" << std::endl;
    for( int i =0 ; i < momentum_vec_one.size();i++){
        std::cout << momentum_vec_one[i]<< " " ;
    }
    std::cout << std::endl;
    std::cout << std::endl;
    for( int i =0 ; i < momentum_vec_two.size();i++){
        std::cout << momentum_vec_two[i]<< " " ;
    }
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << momentum_one_power_t << " " << momentum_two_power_t<< " " << alpha_t<< std::endl;
    std::cout << std::endl;
    std::cout << "end adam_output " << std::endl;*/
	//std::cout << gradient[0] << " " << parameters[0] << std::endl;
	//std::cin.ignore();




	return parameters;
}

}
} /* namespace MRINN */
