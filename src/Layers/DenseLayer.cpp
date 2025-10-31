/*
 * DenseLayer.cpp
 *
 *  Created on: Nov 20, 2017
 *      Author: lennart
 */
#include <boost/serialization/export.hpp>
#include "DenseLayer.h"
#include <iostream>
#include <vector>
//#include <cblas.h>
#include <mkl.h>
#include <cstdlib>
#include <time.h>
#include "../VectorNd.h"
#include <cmath>

//using namespace std;
namespace MRINN {

DenseLayer::DenseLayer( std::vector<int> inp_size, std::vector<int> out_size, bool use_bias): Layer(inp_size, out_size) {
	// TODO Auto-generated constructor stub
	//Layer::Layer(inp_size,out_size);
	//weights = new float*[this->inp_num];
	//std::cout << out_num << " " << inp_num<< std::endl;
	//weights = std::vector<std::vector<float> >(out_num,std::vector<float>(inp_num,0.0));
	this->use_bias = use_bias;
	//std::cout << inp_num+(int)use_bias << " " << inp_num << std::endl;
	weight_matrix = VectorNd<float>(2,out_num,inp_num+(int)use_bias);

	/*if( use_bias){
		//weight_matrix = std::vector<float>(out_num*(inp_num+1),0.0);
		weight_matrix = VectorNd<float>(2,out_num,inp_num+1);
	}else{
		//weight_matrix = std::vector<float>(out_num*inp_num,0.0);
		weight_matrix = VectorNd<float>(2,out_num,inp_num);
	}*/
	//bias = std::vector<float>(out_num,0);
	for( int i = 0; i < this->inp_num + (int)use_bias; i++){
		//weights[i] = std::vector<float>(out_num,0);;
		for( int j = 0; j < this->out_num; j++){
			//weights[j][i] = (((float)rand()/RAND_MAX)-0.5)/inp_num;
			//weight_matrix[j*inp_num]= (((float)rand()/RAND_MAX)-0.5)/inp_num;
			if( i == inp_num){
				//weight_matrix(2,j,i) = (((float)rand()/RAND_MAX)-0.5)/10;
                //weight_matrix(2,j,i) = ((float)rand()/RAND_MAX*2 -0.5) * sqrt(6/(inp_size[1]+inp_size[2]));
				//weight_matrix(2,j,i) = i+1;
				weight_matrix(2,j,i) = 0.0;

			}else{
				//weight_matrix(2,j,i) = (((float)rand()/RAND_MAX)-0.5)/inp_num;
				//weight_matrix(2,j,i) = (((float)rand()/RAND_MAX)-0.5)/10;
               // weight_matrix(2,j,i) = 0.0;
               
                 weight_matrix(2,j,i) = (((float)rand()/RAND_MAX) -0.5) *2* sqrt(6.0/(inp_num+out_num));
                //weight_matrix(2,j,i) = 0.01;
                 //std::cout << weight_matrix(2,j,i) << std::endl;
				//weight_matrix(2,j,i) = i+1;
			}

			//weights[j][i] = 1.0;
			/*if( use_bias){
				weight_matrix(2,j,inp_num) = (((float)rand()/RAND_MAX)-0.5)/inp_num;
			}*/
			//bias[j] = (((float)rand()/RAND_MAX)-0.5)/inp_num;
			//bias[j] = 0.0;
		}

	}

	gradient = VectorNd<float>(2, 1, weight_matrix.get_data().size());
	//Jacobian = std::vector<float>(inp_num*out_num,0);

	//std::cout << Jacobian[4] << std::endl;
	//std::cout << " inp_num: " << inp_num << " out_num: " << out_num<< "mul: " << out_num*inp_num <<   std::endl;
}


DenseLayer::~DenseLayer() {
	// TODO Auto-generated destructor stub
}

void DenseLayer::calculate_jacobian(){
}

void DenseLayer::print_weights(){
	/*std::cout << "Dense Layer:" << std::endl;
	for ( int i = 0; i < out_num;i++){
		for( int j = 0 ; j <= inp_num;j++){ // the <= is due to one bias in the neuron
			if( j != inp_num){
				std::cout << weights[i][j] << " ";

			}else
			{

				//bias[i] = pars[i*(inp_num+1)+j];
				std::cout << "with bias: " << bias[i] << " " ;
			}
		}
		std::cout << std::endl;
	}*/
}

std::vector<float> & DenseLayer::get_var_parameters(){

	return weight_matrix.get_data();

}

void DenseLayer::set_var_parameters(std::vector<float> & pars){

	weight_matrix.get_data().swap(pars);

}

/**
 * returns the parameter gradient for given the backpropagation vector.
 * the backpropagation vector should be the multiplicated jacobi matrices of the layers after this one
 */
VectorNd<float> & DenseLayer::get_parameter_gradient(VectorNd<float> & backprop_vec){
	//std::cout << input.getDimensionSize(1) << std::endl;
    //VectorNd<float> inp(input);
    std::vector<float> inp(input.get_data());
	if(use_bias){
		//std::cout << "test1 " << std::endl;
		//#pragma omp parallel for
		for( int i =0; i < input.getDimensionSize(0);i++){
			//std::cout << i << std::endl;
			inp.insert(inp.begin()+ (input.getDimensionSize(1)+1)*(i+1)-1,1.0);
		}

	}
    //inp.print();
    //std::cin.ignore();
	//std::cout << inp.getDimensionSize(1) << " " << inp.getDimensionSize(0) << " " << backprop_vec.getDimensionSize(1)<< " " << backprop_vec.getDimensionSize(0) << std::endl;


	//const int inp_size = inp.getDimensionSize(1)+use_bias;
	cblas_sgemm( CblasRowMajor, CblasTrans, CblasNoTrans, backprop_vec.getDimensionSize(1), input.getDimensionSize(1)+use_bias, backprop_vec.getDimensionSize(0), 1.0, &(backprop_vec.get_data()[0]), backprop_vec.getDimensionSize(1), &inp[0], input.getDimensionSize(1)+use_bias, 0.0,&gradient.get_data()[0], input.getDimensionSize(1)+use_bias);

    //gradient.print();
    //std::cout <<"gradient dense above" << std::endl;
    //std::cin.ignore();
	return gradient;

}

void DenseLayer::process(){
	//clock_t start,end;
	//start = clock();
	//weight_matrix.print();
	if( use_bias)
	{
		VectorNd<float> inp(input);
		for( int i=0; i < inp.getDimensionSize(0); i++){
			inp.get_data().insert(inp.get_data().begin()+ (inp.getDimensionSize(1)+1)*(i+1)-1,1.0);
		}
		//inp.push_back(1.0);
		//output.fill(0.0);
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, inp.getDimensionSize(0), out_num, inp_num+(int)(use_bias), 1.0,&(inp.get_data()[0]),inp_num+(int)use_bias, &(weight_matrix.get_data())[0], inp_num+(int)use_bias, 0.0, &(output.get_data()[0]), out_num );

        
	}else{
		//cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, input.getDimensionSize(0), out_num, inp_num+(use_bias), 1.0,&(input.get_data()[0]),inp_num+use_bias, &(weight_matrix.get_data())[0], inp_num+use_bias, 0.0, &(output.get_data()[0]), out_num );
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, input.getDimensionSize(0), out_num, inp_num+(use_bias), 1.0,&(input.get_data()[0]),inp_num+use_bias, &(weight_matrix.get_data())[0], inp_num+use_bias, 0.0, &(output.get_data()[0]), out_num );

		//cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, out_num, inp_num+(use_bias), input.getDimensionSize(0), 1.0, &(weight_matrix.get_data())[0], inp_num+use_bias,&(input.get_data()[0]),inp_num+use_bias, 0.0, &(output.get_data()[0]), out_num );

		//const float alpha = 1.0;
		//const float gamma = 0.0;
		//dgemm('N', 'T', out_num, input.getDimensionSize(0), inp_num+(use_bias), &alpha, &(weight_matrix.get_data())[0], inp_num+use_bias, &(input.get_data()[0]),inp_num+use_bias , &gamma, &(output.get_data()[0]), out_num );
	}

	//std::cout << std::endl;
	//std::cout << inp.getDimensionSize(0) << " " << inp.getDimensionSize(1) << std::endl;
	//cblas_dgemv(CblasRowMajor, CblasNoTrans, out_num, inp_num + (int)use_bias, 1.0, &(weight_matrix.get_data())[0], inp_num+(use_bias), &input[0], 1, 0.0, &output[0],1);
	//end = clock();
	/*for( int i = 0; i < this->out_num; i++){

		float out = 0;
		for( int j = 0; j < this->inp_num; j++){
			//std::cout << "out: " << out << " i: " << i   << " j: " << j << std::endl;
			//out = out+ weights[i][j]*input[j];
			out = out + weights.at(i).at(j)*input.at(j);
		}
		out = out+ bias.at(i);
		output[i] = out;

	}*/
	//std::cout << 1000* (end-start)/((float)CLOCKS_PER_SEC) << std::endl;
}

void DenseLayer::backpropagate(){

	/*for( int i = 0; i < out_num;i++){
		for( int j = 0; j < inp_num; j++){
			//std::cout << i << " " << j  << " " << i*out_num + j<< std::endl;
			//Jacobian[i*inp_num + j] = weights[i][j];
			Jacobian.at(i*inp_num + j ) = weights.at(i).at(j);

		}
	}*/
    //error_inp.print();
    //std::cin.ignore();
	if(use_bias){
		VectorNd<float> weights(2,weight_matrix.getDimensionSize(0), weight_matrix.getDimensionSize(1)-1);
		//std::cout << "test" << std::endl;
		//#pragma omp parallel for
		for( int i = 0; i < out_size[0]; i++){

			const int offset = i*(inp_num+use_bias);
			const int weight_offset = i*inp_num;
			//std::cout << i << " " << offset << " " << (i+1)*(inp_num+use_bias)-1 << " " << weight_matrix.get_data().size()<< " " << weights.get_data().size()<<" " << weight_offset<< " " << inp_num <<  std::endl;
            
			std::copy(weight_matrix.get_data().begin()+ offset , weight_matrix.get_data().begin()+(i+1)*(inp_num+use_bias)-1, weights.get_data().begin()+weight_offset);
		}
        //weights.print();
        //std::cin.ignore();

		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, error_inp.getDimensionSize(0), inp_num, out_num, 1.0, &error_inp.get_data()[0], out_num, &weights.get_data()[0], inp_num, 0.0, &(error_out.get_data())[0], inp_num);
        
        

	}else{
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, error_inp.getDimensionSize(0), inp_num, out_num, 1.0, &error_inp.get_data()[0], out_num, &weight_matrix.get_data()[0], inp_num, 0.0, &(error_out.get_data())[0], inp_num);
	}
	//cblas_dgemv(CblasRowMajor, CblasTrans, out_num, inp_num, 1.0, &(weight_matrix.get_data())[0], inp_num, &error_inp[0], 1, 0.0, &error_out[0],1);

	//error_out.print();
    //std::cout << "backprop dense above" <<std::endl;
    //std::cin.ignore();
	//cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, error_inp.getDimensionSize(0), inp_num, out_num, 1.0, &error_inp.get_data()[0], out_num, &weight_matrix.get_data()[0], inp_num, 0.0, &(error_out.get_data())[0], inp_num);
}


} /* namespace MRINN */
//BOOST_CLASS_EXPORT(MRINN::DenseLayer)
