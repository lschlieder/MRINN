/*
 * ScewedDistanceLayer.cpp
 *
 *  Created on: Jan 9, 2018
 *      Author: lennart
 */

#include "ScewedDistanceLayer.h"
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <cblas.h>

namespace MRINN {

ScewedDistanceLayer::ScewedDistanceLayer( std::vector<int> inp_size, std::vector<int> out_size):Layer(inp_size,out_size) {
	//Weights = std::vector<std::vector<float> >(out_num, std::vector<float>(inp_num, 0.0));
	//Positions = std::vector<std::vector<float> >(out_num, std::vector<float>(inp_num,0.0));

	Weights = VectorNd<float>(3,out_num,inp_num,inp_num);
	Positions = VectorNd<float>(2,out_num,inp_num);
	diff_vec = VectorNd<float>(3,1,out_num,inp_num);
	diff_mat = VectorNd<float>(3,1,out_num,inp_num);
	Jacobian = VectorNd<float>(2,out_num,inp_num);
	Jacobian.fill(0.0);
	temp = VectorNd<float>(1,inp_num);
	for( int i = 0 ; i < out_num; i++){
		for( int j = 0; j < inp_num; j++){
			for( int jj = 0; jj < inp_num;jj++){
				Weights(2,i,j,jj) =0.0;
			}
		}
	}

	for( int i = 0 ; i < out_num; i++){
		for( int j = 0; j < inp_num; j++){
			Weights(2,i,j,j) = ((float)rand()/RAND_MAX)/inp_num;
			Positions(2,i,j) = (((float)rand()/RAND_MAX)-0.5)/inp_num;
			//Positions[i][j] = 0.0;
		}
	}

	ones = std::vector<float>(out_num,1.0);


}

void ScewedDistanceLayer::print_weights(){
	/*std::cout << " ScewedDistanceLayer: "  << out_num << " " << inp_num<< std::endl;
	std::cout << " Positions: " << std::endl;

	for( int i_out = 0; i_out < out_num;i_out++){
		for( int i_inp = 0; i_inp < inp_num ; i_inp++){
			std::cout << Positions[i_out][i_inp] << " " ;
			 //params[i_out * inp_num + i_inp] = Positions[i_out][i_inp];
			 //params[ inp_num*out_num + i_out*inp_num + i_inp] = Weights[i_out][i_inp] ;
		}
		std::cout << std::endl;
	}
	std::cout << " Weights: " << std::endl;
	for( int i_out = 0; i_out < out_num;i_out++){
		for( int i_inp = 0; i_inp < inp_num ; i_inp++){
			std::cout << Weights[i_out][i_inp] << " " ;
			 //params[i_out * inp_num + i_inp] = Positions[i_out][i_inp];
			 //params[ inp_num*out_num + i_out*inp_num + i_inp] = Weights[i_out][i_inp] ;
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;*/
}

void ScewedDistanceLayer::set_var_parameters(std::vector<float > & pars){
	/*for( int i_inp = 0; i_inp < inp_num ; i_inp++){
		for( int i_out = 0; i_out < out_num;i_out++){
			Positions[i_out][i_inp] = pars[i_out * inp_num + i_inp];
			//Positions[i_out][i_inp] = 0.5;
			//Weights[i_out][i_inp] = 1.0;
			if( pars[ inp_num*out_num + i_out*inp_num + i_inp] > 0.0){
				Weights[i_out][i_inp] = pars[ inp_num*out_num + i_out*inp_num + i_inp];
			}
				//Weights[i_out][i_inp] =0.0;

		}
	}*/
	std::copy( pars.begin(), pars.begin()+ Positions.get_data().size(), Positions.get_data().begin());
	std::copy( pars.begin()+ Positions.get_data().size(), pars.end(), Weights.get_data().begin());
}

std::vector<float> & ScewedDistanceLayer::get_var_parameters(){
	std::vector<float> params = std::vector<float>((out_num*inp_num) + (out_num*inp_num*inp_num));
	std::copy(Positions.get_data().begin(), Positions.get_data().end(), params.begin());
	std::copy(Weights.get_data().begin(), Weights.get_data().end(), params.begin()+ Positions.get_data().size());

	return params;
	/*for( in& i_inp = 0; i_inp < inp_num ; i_inp++){
		for( int i_out = 0; i_out < out_num;i_out++){
			 params[i_out * inp_num + i_inp] = Positions[i_out][i_inp];
			 params[ inp_num*out_num + i_out*inp_num + i_inp] = Weights[i_out][i_inp] ;
		}
	}*/
	//return params;
}

VectorNd<float> & ScewedDistanceLayer::get_parameter_gradient(VectorNd<float> & backprop_vec){
	VectorNd<float> vec = VectorNd<float>(2,backprop_vec.getDimensionSize(0),(out_num*inp_num) + (out_num*inp_num*inp_num) );
	return vec;
	/*std::vector<float> grad = std::vector<float>(inp_num*out_num*2,0.0);
	for( int i_inp = 0; i_inp < inp_num ; i_inp++){
		for( int i_out = 0; i_out < out_num;i_out++){
			 grad[i_out * inp_num + i_inp] =  (1/output[i_out])* (- Weights[i_out][i_inp]*(input[i_inp] - Positions[i_out][i_inp])) * backprop_vec[i_out];
			 grad[ inp_num*out_num + i_out*inp_num + i_inp] = 0.5* (1/output[i_out]) * (input[i_inp] - Positions[i_out][i_inp]) * (input[i_inp] - Positions[i_out][i_inp])*backprop_vec[i_out];
			 //std::cout << i_out * inp_num + i_inp << " " << inp_num*out_num + i_out*inp_num + i_inp << " " << (1/output[i_out])* (- Weights[i_out][i_inp]*(input[i_inp] - Positions[i_out][i_inp])) << " " << 0.5* (1/output[i_out]) * (input[i_inp] - Positions[i_out][i_inp]) * (input[i_inp] - Positions[i_out][i_inp]) <<  std::endl;
		}
	}
	return grad;*/
}




void ScewedDistanceLayer::process(){

	if( diff_vec.getDimensionSize(0) != input.getDimensionSize(0)){
		diff_vec = VectorNd<float>(3,input.getDimensionSize(0),out_num,inp_num);
		diff_mat = VectorNd<float>(3,input.getDimensionSize(0),out_num,inp_num);
	}
	for( int i_inp = 0; i_inp < input.getDimensionSize(0); i_inp++){
		std::copy( Positions.get_data().begin(), Positions.get_data().end(), &diff_vec(3,i_inp,0,0));
		cblas_sger(CblasRowMajor, Positions.getDimensionSize(0), Positions.getDimensionSize(1), -1.0, &ones[0],1.0, &input(2,i_inp,0),1.0, &diff_vec(3,i_inp,0,0),Positions.getDimensionSize(1));
		for( int i_pos = 0; i_pos < Positions.getDimensionSize(0); i_pos++){
			cblas_sgemv(CblasRowMajor, CblasTrans, Weights.getDimensionSize(1), Weights.getDimensionSize(2), 1.0, &Weights(3,i_pos,0,0), Weights.getDimensionSize(2), &diff_vec(3,i_inp,i_pos,0), 1.0, 0.0, &diff_mat(3,i_inp,i_pos,0), 1.0 );
			output(2,i_inp,i_pos) = cblas_sdot( diff_mat.getDimensionSize(2), &diff_mat(3,i_inp,i_pos,0),1.0, &diff_vec(3,i_inp,i_pos,0),1.0);
		}
	}


	/*for( int i = 0 ; i < out_num; i++){
		std::vector<float> temp_inp = input;
		cblas_daxpy(inp_num,-1.0, &Positions(2,i,0),1.0, &temp_inp[0],1.0);
		cblas_dgemv(CblasRowMajor, CblasNoTrans, inp_num,inp_num,1.0, &Weights(2,out_num,0,0),inp_num, &temp_inp[0],1, 0.0,&temp(1,0),1.0  );
		float dot = cblas_ddot(inp_num, &temp(1,0),1.0, &temp_inp[0],1.0);
		/*float temp = 0.0;
		for( int j = 0; j < inp_num; j++){
			float diff = input[j] - Positions[i][j];
			temp = temp + Weights[i][j] * ( diff*diff);
			//std::cout << Positions[i][j] << std::endl;
		}////////////////////////////////// this was ommented out end here


		//std::cout << temp << std::endl;
		//temp = sqrt(temp);
		output[i] = (dot);
	}*/
}


void ScewedDistanceLayer::backpropagate(){
	//cblas_degmm(CblasRowMajor, CblasNoTrans, CblasNoTrans, diff_vec.getDimensionSize(2), Weights.getDimensionSize(2), Weights.getDimensionSize(1), 1.0, &diff_vec(3,i_inp,0,0), diff_vec.getDimensionSize(1), Weights(3,)  )
	/*for( int i_inp = 0; i_inp < inp_num ; i_inp++){
		float temp = 0.0;
		for( int i_out = 0; i_out < out_num;i_out++){
			float tempsum = 0.0;
			for( int ii = 0; ii < inp_num; ii++){
				tempsum = tempsum + Weights[i_out][ii] *(input[ii] - Positions[i_out][ii]);
			}
			temp = temp + error_inp[i_out] * (1/output[i_out]) * ( tempsum);
			 //grad[i_out * inp_num + i_inp] = (1/output[i_out])* - Weights[i_out][i_inp]*(input[i_inp] - Positions[i_out][i_inp]);
			 //grad[ inp_num*out_num + i_out*inp_num + i_inp] = 0.5* (1/output[i_out]) * (input[i_inp] - Positions[i_out][i_inp]) * (input[i_inp] - Positions[i_out][i_inp]);
		}
		error_out[i_inp] = temp;
	}*/
}
ScewedDistanceLayer::~ScewedDistanceLayer() {
	// TODO Auto-generated destructor stub
}

} /* namespace MRINN */
