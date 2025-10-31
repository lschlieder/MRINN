/*
 * Convolution2DLayer.cpp
 *
 *  Created on: Dec 5, 2017
 *      Author: lennart
 */
//#include <boost/serialization/export.hpp>
//#include <boost/archive/binary_iarchive.hpp>
//#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/export.hpp>
#include "Convolution2DLayer.h"
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <cblas.h>
#include "mkl_dnn.h"

namespace MRINN {
//BOOST_CLASS_EXPORT_IMPLEMENT(Convolution2DLayer);

#define CHUNK_SIZE 100


Convolution2DLayer::Convolution2DLayer(std::vector<int> inp_size, std::vector<int> out_size, int kernel_size, int kernel_number, crop_method cm):Layer(inp_size, out_size) {
	// TODO Auto-generated constructor stub
	this->kernel_size_x = kernel_size;
	this->kernel_size_y = kernel_size;
	this->kernel_padding_x = kernel_size_x/2;
	this->kernel_padding_y = kernel_size_y/2;
	this->kernel_number = kernel_number;
	this->c_method = cm;

	if( kernel_size%2 == 0){
		std::cout << "Warning: You are using an even kernel size. Usually an odd kernel size is preferred!!!"<< std::endl;
	}

	if( out_size.size() != 3 || inp_size.size() != 3){
		throw std::invalid_argument("input or output size have the wrong dimensions.");
	}
	if( out_size[0] != kernel_number){
		std::cout << out_size[0] << " " << kernel_number << " " << inp_size[0] << std::endl;
		throw std::invalid_argument("output dimension does not check with kernel number. out_dim[0] must be == kernel_num");
	}

	/*if( inp_size.size() != out_size.size){
		throw std::invalid_argument("input and output dimensions do not match. There should be output dimension == input dimension");
	}*/
	/*if( cm == cutoff){
		for( int i = inp_size.size() -1; i > 0; i--){
			if( inp_size[i] != out_size[i] +( kernel_size-1)){
				throw std::invalid_argument("one of the output dimension has the wrong size. Please check that newside = oldside - ( kernelsize- 1) ");
			}
		}
	}else{
		for( int i = inp_size.size() -1; i > 0; i--){
			if( inp_size[i] != out_size[i]){
				throw std::invalid_argument("one of the output dimension has the wrong size. Please check that newside = oldside ");
			}
		}
	}*/
	switch (cm){
		case cutoff: {
			for( int i = inp_size.size() -1; i > 0; i--){
				if( inp_size[i] != out_size[i] +( kernel_size-1)){
					throw std::invalid_argument("one of the output dimension has the wrong size. Please check that newside = oldside - ( kernelsize- 1) ");
				}
			}
		}
		break;
		case valid: {
			for( int i = inp_size.size() -1; i > 0; i--){
				if( inp_size[i] != out_size[i] -( kernel_size-1)){
					throw std::invalid_argument("one of the output dimension has the wrong size. Please check that newside = oldside - ( kernelsize- 1) ");
				}
			}
		}
		break;
		default: {
			for( int i = inp_size.size() -1; i > 0; i--){
				if( inp_size[i] != out_size[i]){
					throw std::invalid_argument("one of the output dimension has the wrong size. Please check that newside = oldside ");
				}
			}
		}
	}

	//int kernel_arr_size = pow(kernel_size,inp_size.size()-1);
	int kernel_arr_size = kernel_size_x * kernel_size_y* inp_size[0];
	//kernels = std::vector< std::vector<float> >( kernel_num, std::vector<float>(kernel_arr_size,1.0));
	//rotated_kernels = std::vector< std::vector<float> >(kernel_num, std::vector<float>(kernel_arr_size,1.0));

	kernels = VectorNd<float>(2, kernel_number, kernel_arr_size);
	rotated_kernels = VectorNd<float>(2,kernel_number,kernel_arr_size);
	//input_matrix = VectorNd<float>(3,1, kernel_num,kernel_arr_size);
	if( c_method == cutoff)
		input_matrix = VectorNd<float>(3,input.getDimensionSize(0) * inp_size[0], kernels.getDimensionSize(1),(inp_size[2]-(kernel_size-1))* (inp_size[1]-(kernel_size-1)));
	if( c_method == valid)
		input_matrix = VectorNd<float>(3,input.getDimensionSize(0) * inp_size[0], kernels.getDimensionSize(1),(inp_size[2]+(kernel_size-1))* (inp_size[1]+(kernel_size-1)) );
	if( c_method == zero_padded || c_method == wrap_around)
		input_matrix = VectorNd<float>(3,input.getDimensionSize(0) * inp_size[0], kernels.getDimensionSize(1),inp_size[1]*inp_size[2]);
	//std::cout << out_size[1]*out_size[2] << " " << (inp_size[2]-(kernel_size-1))* (inp_size[1]-(kernel_size-1)) << std::endl;
	//input_matrix = VectorNd<float>(3,input.getDimensionSize(0) * inp_size[0], kernels.getDimensionSize(1),out_size[1]*out_size[2]);

	input_matrix = VectorNd<float>(3,input.getDimensionSize(0) * inp_size[0], out_size[1]*out_size[2],kernels.getDimensionSize(1));
	//if( c_method == cutoff)
	error_matrix = VectorNd<float>(3,output.getDimensionSize(0)* out_size[0], kernels.getDimensionSize(1), inp_size[1]*inp_size[2]);
	for(int i_kernels = 0; i_kernels < kernel_number;i_kernels++){
		for( int i_size = 0; i_size < kernel_arr_size; i_size++){
			//kernels(2,i_kernels,i_size) = ((float)rand()/RAND_MAX -0.5)/10;
			kernels(2,i_kernels,i_size) = ((float)rand()/RAND_MAX -0.5)/10.0;
			//kernels(2,i_kernels,i_size) = ((float)rand()/RAND_MAX )/100;
			//kernels(2,i_kernels,i_size) = i_size%(kernel_size_x*kernel_size_y);
			//kernels(2,i_kernels,i_size) = i_size;

		//	std::cout << i_size <<" " << i_size%(kernel_size_x*kernel_size_y) << std::endl;
			//kernels(2,i_kernels,i_size) = 1.0;
			rotated_kernels(2,i_kernels,kernel_arr_size - i_size -1)= kernels(2,i_kernels,i_size);

		}
	}

	gradient = VectorNd<float>(2,1,kernel_number*kernel_arr_size,0.0);
	temp_grad_vec = std::vector<float>(out_size[1]*out_size[2],0.0);
	//std::cout << gradient.get_data().size() << std::endl;
	if( c_method == cutoff)
		backwards_c_method = valid;
	if( c_method == zero_padded)
		backwards_c_method = zero_padded;
	if( c_method == wrap_around)
		backwards_c_method = wrap_around;
	if( c_method == valid)
		backwards_c_method = cutoff;



	//Jacobian = std::vector<float>(inp_num*out_num,0.0);

}

Convolution2DLayer::Convolution2DLayer(std::vector<int> inp_size, std::vector<int> out_size, int kernel_size_x, int kernel_size_y, int kernel_number, crop_method cm):Layer(inp_size, out_size){
	// TODO Auto-generated constructor stub
	this->kernel_size_x = kernel_size_x;
	this->kernel_size_y = kernel_size_y;
	this->kernel_padding_x = kernel_size_x/2;
	this->kernel_padding_y = kernel_size_y/2;
	this->kernel_number = kernel_number;
	this->c_method = cm;

	if( kernel_size_x%2 == 0 || kernel_size_y%2 == 0){
		std::cout << "Warning: You are using an even kernel size. Usually an odd kernel size is preferred!!!"<< std::endl;
	}

	if( out_size.size() != 3 || inp_size.size() != 3){
		throw std::invalid_argument("input or output size have the wrong dimensions.");
	}
	if( out_size[0] != kernel_number){
		std::cout << out_size[0] << " " << kernel_number << " " << inp_size[0] << std::endl;
		throw std::invalid_argument("output dimension does not check with kernel number. out_dim[0] must be == kernel_num");
	}

	/*if( inp_size.size() != out_size.size){
		throw std::invalid_argument("input and output dimensions do not match. There should be output dimension == input dimension");
	}*/
	/*if( cm == cutoff){
		for( int i = inp_size.size() -1; i > 0; i--){
			if( inp_size[i] != out_size[i] +( kernel_size-1)){
				throw std::invalid_argument("one of the output dimension has the wrong size. Please check that newside = oldside - ( kernelsize- 1) ");
			}
		}
	}else{
		for( int i = inp_size.size() -1; i > 0; i--){
			if( inp_size[i] != out_size[i]){
				throw std::invalid_argument("one of the output dimension has the wrong size. Please check that newside = oldside ");
			}
		}
	}*/
	switch (cm){
		case cutoff: {
			/*for( int i = inp_size.size() -1; i > 0; i--){
				if( inp_size[i] != out_size[i] +( kernel_size-1)){
					throw std::invalid_argument("one of the output dimension has the wrong size. Please check that newside = oldside - ( kernelsize- 1) ");
				}
			}*/
			if( inp_size[2] != out_size[2] + (kernel_size_x -1) ||  inp_size[1] != out_size[1] + (kernel_size_y -1) ){
				throw std::invalid_argument("one of the output dimension has the wrong size. Please check that newside = oldside - ( kernelsize- 1) ");
			}
		}
		break;
		case valid: {
			/*for( int i = inp_size.size() -1; i > 0; i--){
				if( inp_size[i] != out_size[i] -( kernel_size-1)){
					throw std::invalid_argument("one of the output dimension has the wrong size. Please check that newside = oldside - ( kernelsize- 1) ");
				}
			}*/
			if( inp_size[1] != out_size[1] -( kernel_size_y-1) || inp_size[2] != out_size[2] -( kernel_size_x-1)){
				throw std::invalid_argument("one of the output dimension has the wrong size. Please check that newside = oldside - ( kernelsize- 1) ");
			}
		}
		break;
		default: {
			for( int i = inp_size.size() -1; i > 0; i--){
				if( inp_size[i] != out_size[i]){
					throw std::invalid_argument("one of the output dimension has the wrong size. Please check that newside = oldside ");
				}
			}
		}
	}

	//int kernel_arr_size = pow(kernel_size,inp_size.size()-1);
	int kernel_arr_size = kernel_size_x * kernel_size_y* inp_size[0];
	//kernels = std::vector< std::vector<float> >( kernel_num, std::vector<float>(kernel_arr_size,1.0));
	//rotated_kernels = std::vector< std::vector<float> >(kernel_num, std::vector<float>(kernel_arr_size,1.0));

	kernels = VectorNd<float>(2, kernel_number, kernel_arr_size);
	rotated_kernels = VectorNd<float>(2,kernel_number,kernel_arr_size);
	//input_matrix = VectorNd<float>(3,1, kernel_num,kernel_arr_size);
	if( c_method == cutoff)
		input_matrix = VectorNd<float>(3,input.getDimensionSize(0) * inp_size[0], kernels.getDimensionSize(1),(inp_size[2]-(kernel_size_x-1))* (inp_size[1]-(kernel_size_y-1)));
	if( c_method == valid)
		input_matrix = VectorNd<float>(3,input.getDimensionSize(0) * inp_size[0], kernels.getDimensionSize(1),(inp_size[2]+(kernel_size_x-1))* (inp_size[1]+(kernel_size_y-1)) );
	if( c_method == zero_padded || c_method == wrap_around)
		input_matrix = VectorNd<float>(3,input.getDimensionSize(0) * inp_size[0], kernels.getDimensionSize(1),inp_size[1]*inp_size[2]);
	//std::cout << out_size[1]*out_size[2] << " " << (inp_size[2]-(kernel_size-1))* (inp_size[1]-(kernel_size-1)) << std::endl;
	//input_matrix = VectorNd<float>(3,input.getDimensionSize(0) * inp_size[0], kernels.getDimensionSize(1),out_size[1]*out_size[2]);

	input_matrix = VectorNd<float>(3,input.getDimensionSize(0) * inp_size[0], out_size[1]*out_size[2],kernels.getDimensionSize(1));
	//if( c_method == cutoff)
	error_matrix = VectorNd<float>(3,output.getDimensionSize(0)* out_size[0], kernels.getDimensionSize(1), inp_size[1]*inp_size[2]);
	for(int i_kernels = 0; i_kernels < kernel_number;i_kernels++){
		for( int i_size = 0; i_size < kernel_arr_size; i_size++){
			kernels(2,i_kernels,i_size) = ((float)rand()/RAND_MAX -0.5)/10;
			//kernels(2,i_kernels,i_size) = ((float)rand()/RAND_MAX )/100;
			//kernels(2,i_kernels,i_size) = 1.0;
			//kernels(2,i_kernels,i_size) = 1.0;
			rotated_kernels(2,i_kernels,kernel_arr_size - i_size -1)= kernels(2,i_kernels,i_size);

		}
	}
	gradient = VectorNd<float>(2,1,kernel_number*kernel_arr_size,0.0);

	temp_grad_vec = std::vector<float>(out_size[1]*out_size[2],0.0);

	if( c_method == cutoff)
		backwards_c_method = valid;
	if( c_method == zero_padded)
		backwards_c_method = zero_padded;
	if( c_method == wrap_around)
		backwards_c_method = wrap_around;
	if( c_method == valid)
		backwards_c_method = cutoff;



	//Jacobian = std::vector<float>(inp_num*out_num,0.0);

}

Convolution2DLayer::~Convolution2DLayer() {
	// TODO Auto-generated destructor stub
}

/*std::vector<float> Convolution2DLayer::convolution(std::vector<float> inp){

}*/
void Convolution2DLayer::calculate_jacobian(){

	/*for( int i_out = 0; i_out < out_num; i_out++){
		int x_out = i_out%out_size[1];
		int y_out = i_out/out_size[1];
			for( int i_kern = 0; i_kern < kernels.size(); i_kern++){
				for( int i_inp = 0; i_inp < inp_num; i_inp++){
					int x_inp = i_inp%inp_size[1];
					int y_inp = i_inp/inp_size[1];

					int x_diff = x_inp - x_out;
					int y_diff = y_inp - y_out;

					int pos_kernel_x = x_diff + kernel_size/2;
					int pos_kernel_y = y_diff + kernel_size/2;
					switch( this->c_method){
						case zero_padded: {
							if (pos_kernel_x < kernel_size && pos_kernel_y < kernel_size && pos_kernel_x >=0 && pos_kernel_y >= 0){
								Jacobian[i_kern*(kernel_number*out_size[1]*out_size[2])+i_out * inp_num + i_inp] = kernels[i_kern][pos_kernel_y * kernel_size + pos_kernel_x];
								//std::cout << kernels[i_kern][pos_kernel_y * kernel_size + pos_kernel_x] << " " ;
							}else{
								Jacobian[i_kern*(kernel_number*out_size[1]*out_size[2])+i_out * inp_num + i_inp] = 0.0;
								//std::cout << "0 ";
							}
							break;
						}
						case wrap_around: {
							if( pos_kernel_x >= inp_size[2])
								pos_kernel_x = pos_kernel_x % inp_size[2];

							if( pos_kernel_y >= inp_size[1])
								pos_kernel_y = pos_kernel_y % inp_size[1];

							if (pos_kernel_x < kernel_size && pos_kernel_y < kernel_size && pos_kernel_x >=0 && pos_kernel_y >= 0){
								Jacobian[i_kern*(kernel_number*out_size[1]*out_size[2])+i_out * inp_num + i_inp] = kernels[i_kern][pos_kernel_y * kernel_size + pos_kernel_x];
								//std::cout << kernels[i_kern][pos_kernel_y * kernel_size + pos_kernel_x] << " " ;
							}else{
								Jacobian[i_kern*(kernel_number*out_size[1]*out_size[2])+i_out * inp_num + i_inp] = 0.0;
								//std::cout << "0 ";
							}
							break;
						}
						case cutoff: {
							int x_diff = x_inp - (x_out + kernel_size/2);
							int y_diff = y_inp - (y_out + kernel_size/2);

							int pos_kernel_x = x_diff + kernel_size/2;
							int pos_kernel_y = y_diff + kernel_size/2;

							if (pos_kernel_x < kernel_size && pos_kernel_y < kernel_size && pos_kernel_x >=0 && pos_kernel_y >= 0){
								Jacobian[i_kern*(kernel_number*out_size[1]*out_size[2])+i_out * inp_num + i_inp] = kernels[i_kern][pos_kernel_y * kernel_size + pos_kernel_x];
								//std::cout << kernels[i_kern][pos_kernel_y * kernel_size + pos_kernel_x] << " " ;
							}else{
								Jacobian[i_kern*(kernel_number*out_size[1]*out_size[2])+i_out * inp_num + i_inp] = 0.0;
								//std::cout << "0 ";
							}
							break;
						}
					}

				}
				//std::cout << std::endl;
			}
		}*/

}

void Convolution2DLayer::set_var_parameters(std::vector<float> & pars){
	/*for( int i_kern = 0; i_kern < kernels.size(); i_kern++){
		for( int ix = 0; ix < kernel_size; ix++){
			for( int iy = 0; iy < kernel_size; iy++){
				kernels[i_kern][iy*kernel_size+ix] = pars[i_kern*(kernel_size*kernel_size) + iy*(kernel_size) + ix];
			}
		}
	}*/

	std::copy( pars.begin(),pars.end(), &kernels(2,0,0));


	//calculate the rotated kernels;

	const int kernel_size = kernel_size_x * kernel_size_y;
	#pragma omp parallel for
	for( int i = 0; i < kernels.getDimensionSize(0); i++){
		const int kernel_offset = i * kernels.getDimensionSize(1);
		for( int j = 0; j < inp_size[0]; j++){
			//std::cout << i << " " << kernels.getDimensionSize(0) << " " << j*kernel_size << " " << (j+1)*kernel_size << " " << kernels.getDimensionSize(1)<< std::endl;
			//std::reverse_copy(&kernels(2,i,j*kernel_size), &kernels(2,i,(j+1)*kernel_size), &rotated_kernels(2,i,j*kernel_size) );
			//std::reverse_copy( &(kernels.get_data()[kernel_offset + j*kernel_size]), &(kernels.get_data()[kernel_offset+(j+1)kernel_size]), &(rotated_kernels.get_data()[kernel_offset + i*kernel_size]));
			std::reverse_copy( &kernels.get_data()[kernel_offset+ j*kernel_size], &kernels.get_data()[kernel_offset + (j+1)*kernel_size], &rotated_kernels.get_data()[kernel_offset + j*kernel_size]);
		}
	}


	// calculate the rotated kernels;
	//int kernel_arr_size = pow(kernel_size,inp_size.size()-1);
	//rotated_kernels = std::vector<std::vector<float> >(kernel_number, std::vector<float>(kernel_arr_size, 0.0));

	/*for( int i_kernel = 0; i_kernel < this->kernel_number; i_kernel++){
		for( int i_ky = 0; i_ky < this->kernel_size; i_ky++ ){
			for( int i_kx = 0; i_kx < kernel_size; i_kx++){
				int rotated_kx = kernel_size -1 -i_kx;
				int rotated_ky = kernel_size -1 -i_ky;

				rotated_kernels[i_kernel][i_ky*kernel_size + i_kx] = kernels[i_kernel][rotated_ky*kernel_size +rotated_kx];

				//std::cout << i_kx << " " << i_ky << " " << rotated_kx << " " << rotated_ky << " " << rotated_kernels[i_kernel][i_ky*kernel_size + i_kx] << " " << kernels[i_kernel][rotated_ky*kernel_size +rotated_kx] << " " << rotated_kx*kernel_size + rotated_ky <<   std::endl;
			}
		}
	}*/
}

std::vector<float> & Convolution2DLayer::get_var_parameters(){
   /*	std::vector<float> ret(kernel_number*kernel_size*kernel_size,0.0);
	for( unsigned int i_kern = 0; i_kern < kernels.size(); i_kern++){
		for( int ix = 0; ix < kernel_size; ix++){
			for( int iy = 0; iy < kernel_size; iy++){
				ret[i_kern*(kernel_size*kernel_size) + iy*(kernel_size) + ix] = kernels[i_kern][iy*kernel_size + ix];
			}
		}
	}*/
	//kernels.print();
	return kernels.get_data();


	//return ret;
}

float Convolution2DLayer::partial_derivative( int out_x, int out_y, int kern_x, int kern_y, int i_pic, int i_batch){
/*
	int inp_x;
	int inp_y;
	switch( this->c_method){
		case zero_padded:{
			inp_x = (kern_x - kernel_size/2) + out_x;
			inp_y = (kern_y - kernel_size/2) + out_y;
			break;
		}
		case wrap_around:{
			inp_x = (kern_x - kernel_size/2) + out_x;
			inp_y = (kern_y - kernel_size/2) + out_y;

			inp_x = (inp_size[2] + (inp_x%inp_size[2])) % inp_size[2];
			inp_y = (inp_size[1] + (inp_y%inp_size[1])) % inp_size[1];
			break;
		}
		case cutoff:{
			inp_x = kern_x + out_x;
			inp_y = kern_y + out_y;
			break;
		}
		default: {

		}
	}

	return input.get_data()[i_batch*(input.getDimensionSize(1))+ i_pic*( inp_size[1] * inp_size[2]) + inp_y*(inp_size[2]) + inp_x];*/

}


VectorNd<float> & Convolution2DLayer::get_parameter_gradient(VectorNd<float> &backprop_vec){
	//VectorNd<float> ret(2,1,kernel_number*kernel_size*kernel_size,0.0);
	gradient.fill(0.0);
	/*if( temp_grad_vec.size()!= backprop_vec.getDimensionSize(1)){
		temp_grad_vec = std::vector<float>(backprop_vec.getDimensionSize(1));
	}*/

	for( int i_batch = 0; i_batch < backprop_vec.getDimensionSize(0);i_batch++){
		const int backprop_batch_offset = i_batch * backprop_vec.getDimensionSize(1);
		const int input_batch_offset = i_batch * input.getDimensionSize(1);

		for( int inp_number = 0; inp_number < inp_size[0]; inp_number++){
			const int inp_offset = input_batch_offset + inp_number * inp_size[1] * inp_size[2];
			const int inp_grad_offset = inp_number * kernel_size_x*kernel_size_y;
			for( int out_number = 0; out_number < out_size[0]; out_number++){
				const int back_offset = backprop_batch_offset + out_number * out_size[1]*out_size[2];
				const int kernel_offset = inp_grad_offset + out_number * kernels.getDimensionSize(1);
				//std::cout << (out_size[2] + kernel_size_x - inp_size[2]-1 ) /2 << " " << (out_size[1] + kernel_size_y - inp_size[1] -1)/2 << "  " << out_size[1] << " " << out_size[2] << " " << inp_size[1] << " " << inp_size[2]  << std::endl;
				//std::cout << inp_offset << " " << input.get_data().size() << "  " << back_offset << " " << backprop_vec.get_data().size() << "  " << kernel_offset << " " << gradient.get_data().size() <<  std::endl;
				Convolution2D(&input.get_data()[inp_offset], inp_size[2], inp_size[1], &backprop_vec.get_data()[back_offset], out_size[1], out_size[2] , zero_padded, (out_size[2] + kernel_size_x - inp_size[2] -1) /2, (out_size[1] + kernel_size_y - inp_size[1] -1)/2, &gradient.get_data()[kernel_offset]  );
			}
		}
	}
	return gradient;
/*
	//backprop_vec.print();
	const int output_size = (out_size[1]*out_size[2]);
	#pragma omp parallel for
	for( int i_batch = 0; i_batch < backprop_vec.getDimensionSize(0); i_batch++){
		//#pragma omp parallel for

		const int batch_offset = i_batch*backprop_vec.getDimensionSize(0);
		#pragma omp parallel for
		for( int i_kern = 0; i_kern < kernels.getDimensionSize(0); i_kern++){

			const int kernel_offset = i_kern*(output_size);
			const int kernel_offset_grad = i_kern *(kernel_size*kernel_size);
			const int i_pic = i_kern / out_size[0];
			const int pic_offset = batch_offset+i_pic*( inp_size[1] * inp_size[2]);
			for( int kern_y = 0; kern_y < kernel_size; kern_y++){
				const int kernel_offset_grad_ky = kernel_offset_grad+kern_y*kernel_size;
				for( int kern_x = 0; kern_x < kernel_size; kern_x++){
					//std::fill(temp_grad_vec.begin(), temp_grad_vec.end(),0.0);
					//std::cout << out_size[2]*out_size[1] << " " << temp_grad_vec.size() << std::endl;
					for( int i_out= 0; i_out < out_size[2]*out_size[1]; i_out++){
						int out_x = i_out % out_size[1];
						int out_y = i_out / out_size[1];

						int inp_x;
						int inp_y;
						switch( this->c_method){
							case zero_padded:{
								inp_x = (kern_x - kernel_size/2) + out_x;
								inp_y = (kern_y - kernel_size/2) + out_y;
								break;
							}
							case wrap_around:{
								inp_x = (kern_x - kernel_size/2) + out_x;
								inp_y = (kern_y - kernel_size/2) + out_y;

								inp_x = (inp_size[2] + (inp_x%inp_size[2])) % inp_size[2];
								inp_y = (inp_size[1] + (inp_y%inp_size[1])) % inp_size[1];
								break;
							}
							case cutoff:{
								inp_x = kern_x + out_x;
								inp_y = kern_y + out_y;
								break;
							}
							default: {
								inp_x = -1;
								inp_y = -1;
							}
						}

						//return input.get_data()[i_batch*(input.getDimensionSize(1))+ i_pic*( inp_size[1] * inp_size[2]) + inp_y*(inp_size[2]) + inp_x];
						//temp_grad_vec[kernel_offset + i_out]= partial_derivative(out_x,out_y,kern_x,kern_y,i_pic, i_batch);
						//temp_grad_vec[kernel_offset+i_out] = input.get_data()[pic_offset + inp_y*(inp_size[2])+ inp_x] * backprop_vec.get_data()[kernel_offset + i_out];
						gradient.get_data()[kernel_offset_grad_ky + kern_x] += input.get_data()[pic_offset + inp_y*(inp_size[2])+ inp_x] * backprop_vec.get_data()[kernel_offset + i_out];
					}


		////////////////////////
					//float temp_res;
					//float temp_res = cblas_ddot( temp_grad_vec.size(), &temp_grad_vec[0],1.0, &backprop_vec(2,i_batch,0),1.0);
					//gradient.get_data()[kernel_offset_grad_ky+  kern_x] += temp_res;

				}
			}
		}

	}
	//ret.print();
	//std::cin.ignore();

	return gradient;
	/*VectorNd<float> ret(2,1, kernels.get_data().size());
	ret.fill(0.0);
	return ret;*/
}



void Conv2D(float* inp, int inp_size_x, int inp_size_y,float * kernel, int kernel_size_x, int kernel_size_y, int zero_padding_x, int zero_padding_y, float* result){
	//result = std::vector<float>(inp_size_x* inp_size_y,0.0);
	//#pragma omp parallel for
	for( int i_y = 0; i_y <inp_size_y ; i_y++ ){

		for( int i_x = 0; i_x < inp_size_x; i_x++){
			float res = 0.0;

			const int ixtemp = i_x - zero_padding_x;

				//#pragma omp parallel for
				//std::cout << i_y << std::endl;
				for( int i_kernely = 0; i_kernely < kernel_size_y; i_kernely++){
					const int kernel_offsety = (i_kernely ) *  kernel_size_x;
					const int inp_y = i_y+(i_kernely-zero_padding_y);
					const int inp_offset = (inp_y)*(inp_size_x);
					//std::cout << inp_y << " ";
					if( inp_y >= 0 && inp_y < inp_size_y){
						for( int i_kernelx = 0; i_kernelx < kernel_size_x; i_kernelx++){
							//const int inp_x = i_x + (i_kernelx - kernel_size/2);
							const int inp_x = ixtemp + i_kernelx;
							//std::cout << inp_x << std::endl;

							if( inp_x >= 0 && inp_x < inp_size_x){
								res += kernel[kernel_offsety + i_kernelx]*inp[inp_offset + inp_x] ;
							}
						}
					}
				}
				std::cout << i_y << " " << i_x << std::endl;
				const int output_offset = i_y*(inp_size_x)+ i_x;
				result[output_offset] += res;


		}
	}
}

void Convolution2DLayer::Convolution2D(float* inp, int inp_size_x, int inp_size_y,float * kernel, int kernel_size_x, int kernel_size_y, crop_method cm, int zero_padding_x, int zero_padding_y, float* result){
	//std::vector<float> result;
	//const int block_size = std::min(CHUNK_SIZE, inp_size_y);
	switch( cm){
		case cutoff: {
			//result = std::vector<float>((inp_size_x-(kernel_size-1))* (inp_size_y-(kernel_size-1)),0.0);
			//#pragma omp parallel for
			for( int i_y = 0 ; i_y < inp_size_y-(kernel_size_y-1); i_y++){
			//	#pragma omp parallel for
				for( int i_x = 0; i_x < inp_size_x-(kernel_size_x-1); i_x++){
					float res = 0.0;
			//		#pragma omp parallel for
					for( int i_kernely = 0; i_kernely < kernel_size_y; i_kernely++){
						const int i_kernel_offset = i_kernely*kernel_size_x;
						const int i_input_offset = (i_y+i_kernely)*(inp_size_x) + i_x ;
						for( int i_kernelx = 0; i_kernelx < kernel_size_x; i_kernelx++){
							//sliding_window[ i_kernel_offset + i_kernelx] = inp[(i_y+i_kernely)*(inp_size_x)+(i_x+i_kernelx) ];
							res += inp[i_input_offset + i_kernelx]* kernel[i_kernel_offset + i_kernelx];
							//std::cout << i_y << " " << i_x << " " << i_kernely << " " << i_kernelx << "   " << inp[i_input_offset + i_kernelx] << " " << kernel[i_kernel_offset + i_kernelx] << " " << i_input_offset + i_kernelx << std::endl;
						}
					}
					//std::cout << i_y << " " << i_x << std::endl;
					const int output_offset = i_y*((inp_size_x)-(kernel_size_x-1))+ i_x;
					result[output_offset] += res;
				}
			}
		}
		break;
		case valid:{
			//result = std::vector<float>((inp_size_x+(kernel_size-1))* (inp_size_y+(kernel_size-1)),0.0);
			//#pragma omp parallel for
			for( int i_y = 0 ; i_y < inp_size_y+(kernel_size_y-1); i_y++){
			//	#pragma omp parallel for
				for( int i_x = 0; i_x < inp_size_x+(kernel_size_x-1); i_x++){
					float res = 0.0;
					//#pragma omp parallel for
					for( int i_kernely = 0; i_kernely < kernel_size_y; i_kernely++){
						//const int y_k = ;
						const int kernel_offsety = (i_kernely ) *  kernel_size_x;
						const int inp_y = i_y+(i_kernely-(kernel_size_y-1));
						const int inp_offset = (inp_y)*(inp_size_x);
						if( inp_y >= 0 && inp_y < inp_size_y){
							for( int i_kernelx = 0; i_kernelx < kernel_size_x; i_kernelx++){
								//int x_k = i_kernelx - kernel_size/2;
								const int inp_x = i_x + (i_kernelx - (kernel_size_x-1));
								if( inp_x >= 0 && inp_x < inp_size_x){
									res += kernel[kernel_offsety + i_kernelx]*inp[inp_offset + inp_x] ;
								}
							}
						}
					}
					const int output_offset = i_y*((inp_size_x)+(kernel_size_x-1))+ i_x;
					result[ output_offset] += res; //cblas_ddot(sliding_window.size(), &sliding_window[0],1.0,&kernel[0],1.0);
				}
			}
		}
		break;
		case zero_padded:{
			//result = std::vector<float>(inp_size_x* inp_size_y,0.0);
			//#pragma omp parallel for
			//std::cout << inp_size_y << " " << kernel_size_y << " " << zero_padding_y << " " <<inp_size_x << " " << kernel_size_x << " " << zero_padding_x<< std::endl;
			const int boundary_y = inp_size_y-kernel_size_y + (2*zero_padding_y)+1;
			const int boundary_x = inp_size_x-kernel_size_x + (2*zero_padding_x)+1;

		//	#pragma omp parallel for
			for( int i_y = 0; i_y <boundary_y ; i_y++ ){
				for( int i_x = 0; i_x < boundary_x; i_x++){
					float res = 0.0;
					const int ixtemp = i_x - zero_padding_x;
					//std::cout << i_y << " " << i_x << std::endl << std::endl;
						//#pragma omp parallel for
						//std::cout << i_y << std::endl;
						for( int i_kernely = 0; i_kernely < kernel_size_y; i_kernely++){
							const int kernel_offsety = (i_kernely ) *  kernel_size_x;
							const int inp_y = i_y+(i_kernely-zero_padding_y);
							const int inp_offset = (inp_y)*(inp_size_x);
							//std::cout << inp_y << " ";
							//std::cout << inp_y<< " ";
							if( inp_y >= 0 && inp_y < inp_size_y){
								for( int i_kernelx = 0; i_kernelx < kernel_size_x; i_kernelx++){
									//const int inp_x = i_x + (i_kernelx - kernel_size/2);
									const int inp_x = ixtemp + i_kernelx;
									//std::cout << inp_x << std::endl;
								//	std::cout << inp_x << " " ;

									if( inp_x >= 0 && inp_x < inp_size_x){
										res += kernel[kernel_offsety + i_kernelx]*inp[inp_offset + inp_x] ;
									}
								}
							}
						}
						//std::cout << std::endl;
						//std::cout << i_y << " " << i_x << " " << inp_size_y -kernel_size_y + (2*zero_padding_y)+1 << " " << inp_size_x-kernel_size_x + (2*zero_padding_x)+1<< " " << ixtemp<< " " << i_y- zero_padding_y<< "    " << res<< std::endl;

						const int output_offset = i_y*boundary_x+ i_x;
						result[output_offset] += res;


				}
			}
		}
		break;
		case wrap_around:{
			//result = std::vector<float>(inp_size_x* inp_size_y,0.0);
			//#pragma omp parallel for
			for( int i_y = 0 ; i_y < inp_size_y; i_y++){
			//	#pragma omp parallel for
				for( int i_x = 0; i_x < inp_size_x; i_x++){
					float res = 0.0;
				//	#pragma omp parallel for
					for( int i_kernely = 0; i_kernely < kernel_size_y; i_kernely++){
						const int kernel_offsety = (i_kernely ) *  kernel_size_x;
						const int inp_y = (inp_size_y + (( i_y+(i_kernely-kernel_size_y/2))%inp_size_y)) % inp_size_y;
						const int inp_offset = (inp_y)*(inp_size_y);
						for( int i_kernelx = 0; i_kernelx < kernel_size_x; i_kernelx++){
							const int inp_x =(inp_size_x + ((i_x + (i_kernelx - kernel_size_x/2))%inp_size_x)) % inp_size_x;
							res += kernel[kernel_offsety + i_kernelx]*inp[inp_offset + inp_x ] ;
						}
					}
					const int output_offset = i_y*(inp_size_x)+ i_x;
					result[ output_offset] += res;
				}
			}

		}
		break;
	}
}
/*
void Convolution2DLayer::Convolution2D(float* inp, int inp_size_x, int inp_size_y, int inp_image_num ,float * kernel, int kernel_size, crop_method cm, float* result){
	//std::vector<float> result;
	//const int block_size = std::min(CHUNK_SIZE, inp_size_y);
	switch( cm){
		case cutoff: {
			//result = std::vector<float>((inp_size_x-(kernel_size-1))* (inp_size_y-(kernel_size-1)),0.0);
			//#pragma omp parallel for
			for( int i_y = 0 ; i_y < inp_size_y-(kernel_size-1); i_y++){
			//	#pragma omp parallel for
				for( int i_x = 0; i_x < inp_size_x-(kernel_size-1); i_x++){
					float res = 0.0;
			//		#pragma omp parallel for
					for( int i_image = 0; i_image< inp_image_num; i_image++){
						const int image_offset = i_image * inp_size_y*inp_size_x;
						const int kernel_image_offset = i_image* kernel_size*kernel_size;
						for( int i_kernely = 0; i_kernely < kernel_size; i_kernely++){
							const int i_kernel_offset = i_kernely*kernel_size + kernel_image_offset;
							const int i_input_offset = (i_y+i_kernely)*(inp_size_x) + i_x + image_offset;
							for( int i_kernelx = 0; i_kernelx < kernel_size; i_kernelx++){
								//sliding_window[ i_kernel_offset + i_kernelx] = inp[(i_y+i_kernely)*(inp_size_x)+(i_x+i_kernelx) ];



								res += inp[i_input_offset + i_kernelx]* kernel[i_kernel_offset + i_kernelx];
							}
						}
					}
					const int output_offset = i_y*((inp_size_x)-(kernel_size-1))+ i_x;
					result[output_offset] += res;
				}
			}
		}
		break;
		case valid:{
			//result = std::vector<float>((inp_size_x+(kernel_size-1))* (inp_size_y+(kernel_size-1)),0.0);
			//#pragma omp parallel for
			for( int i_y = 0 ; i_y < inp_size_y+(kernel_size-1); i_y++){
			//	#pragma omp parallel for
				for( int i_x = 0; i_x < inp_size_x+(kernel_size-1); i_x++){
					float res = 0.0;
					//#pragma omp parallel for
					for( int i_image = 0; i_image< inp_image_num; i_image++){
						const int image_offset = i_image * inp_size_y*inp_size_x;
						const int kernel_image_offset = i_image* kernel_size*kernel_size;
						for( int i_kernely = 0; i_kernely < kernel_size; i_kernely++){
							//const int y_k = ;
							const int kernel_offsety = (i_kernely ) *  kernel_size+ kernel_image_offset;
							const int inp_y = i_y+(i_kernely-kernel_size/2);
							const int inp_offset = (inp_y)*(inp_size_y) + image_offset;
							if( inp_y >= 0 && inp_y < inp_size_y){
								for( int i_kernelx = 0; i_kernelx < kernel_size; i_kernelx++){
									//int x_k = i_kernelx - kernel_size/2;
									const int inp_x = i_x + (i_kernelx - kernel_size/2);
									if( inp_x >= 0 && inp_x < inp_size_x){
										res += kernel[kernel_offsety + i_kernelx]*inp[inp_offset + inp_x] ;
									}
								}
							}
						}
					}
					const int output_offset = i_y*(inp_size_x)+ i_x;
					result[ output_offset] += res; //cblas_ddot(sliding_window.size(), &sliding_window[0],1.0,&kernel[0],1.0);
				}
			}
		}
		break;
		case zero_padded:{
			//result = std::vector<float>(inp_size_x* inp_size_y,0.0);
			#pragma omp parallel for
			for( int body_start_index = 0 ; body_start_index < inp_size_y; body_start_index+= CHUNK_SIZE){
				//#pragma omp parallel for
				//std::cout << body_start_index << std::endl;
				const int body_end_index = body_start_index + CHUNK_SIZE;
				for( int i_x = 0; i_x < inp_size_x; i_x++){
					float res = 0.0;

					const int ixtemp = i_x - kernel_size/2;
					for( int i_y = body_start_index; i_y < body_end_index && i_y< inp_size_y; i_y+=1 ){
						//#pragma omp parallel for
						//std::cout << i_y << std::endl;
						for( int i_image = 0; i_image< inp_image_num; i_image++){
							const int image_offset = i_image * inp_size_y*inp_size_x;
							const int kernel_image_offset = i_image* kernel_size*kernel_size;
							for( int i_kernely = 0; i_kernely < kernel_size; i_kernely++){
								const int kernel_offsety = (i_kernely ) *  kernel_size + kernel_image_offset;
								const int inp_y = i_y+(i_kernely-kernel_size/2);
								const int inp_offset = (inp_y)*(inp_size_y)+image_offset;
								//std::cout << inp_y << " ";
								if( inp_y >= 0 && inp_y < inp_size_y){
									for( int i_kernelx = 0; i_kernelx < kernel_size; i_kernelx++){
										//const int inp_x = i_x + (i_kernelx - kernel_size/2);
										const int inp_x = ixtemp + i_kernelx;
										//std::cout << inp_x << std::endl;

										if( inp_x >= 0 && inp_x < inp_size_x){
											res += kernel[kernel_offsety + i_kernelx]*inp[inp_offset + inp_x] ;
										}
									}
								}
							}
						}
						const int output_offset = i_y*(inp_size_x)+ i_x;
						result[output_offset] += res;

					}
				}
			}
		}
		break;
		case wrap_around:{
			//result = std::vector<float>(inp_size_x* inp_size_y,0.0);
			//#pragma omp parallel for
			for( int i_y = 0 ; i_y < inp_size_y; i_y++){
			//	#pragma omp parallel for
				for( int i_x = 0; i_x < inp_size_x; i_x++){
					float res = 0.0;
				//	#pragma omp parallel for
					for( int i_image = 0; i_image< inp_image_num; i_image++){
						const int image_offset = i_image * inp_size_y*inp_size_x;
						const int kernel_image_offset = i_image* kernel_size*kernel_size;
						for( int i_kernely = 0; i_kernely < kernel_size; i_kernely++){
							const int kernel_offsety = (i_kernely ) *  kernel_size + kernel_image_offset;
							const int inp_y = (inp_size_y + (( i_y+(i_kernely-kernel_size/2))%inp_size_y)) % inp_size_y;
							const int inp_offset = (inp_y)*(inp_size_y)+  image_offset;
							for( int i_kernelx = 0; i_kernelx < kernel_size; i_kernelx++){
								const int inp_x =(inp_size_x + ((i_x + (i_kernelx - kernel_size/2))%inp_size_x)) % inp_size_x;
								res += kernel[kernel_offsety + i_kernelx]*inp[inp_offset + inp_x ] ;
							}
						}
					}
					const int output_offset = i_y*(inp_size_x)+ i_x;
					result[ output_offset] += res;
				}
			}

		}
		break;
	}
}*/



void Convolution2DLayer::backpropagate(){

	//resize_error_matrix();
	//fill_error_matrix();
	//error_out.fill(0.0);
	//std::cout << "before backpropagate" << std::endl;
	error_out.fill(0.0);

	const int error_inp_size = out_size[1] * out_size[2];
	const int error_out_size = inp_size[2]* inp_size[1];
	//std::cout << error_inp.getDimensionSize(0) << " " << error_inp.getDimensionSize(1) << std::endl;
	#pragma omp parallel for
	for( int i_batch = 0; i_batch < error_inp.getDimensionSize(0); i_batch++){
		const int input_batch_offset = i_batch * error_inp.getDimensionSize(1);
		const int output_batch_offset =i_batch *error_out.getDimensionSize(1);
		#pragma omp parallel for
		for( int i_kernel = 0; i_kernel < kernel_number; i_kernel++){
			const int kernel_stack_offset = i_kernel * kernels.getDimensionSize(1);
			const int error_inp_offset = i_kernel* error_inp_size + input_batch_offset;
			for( int i_inp_stack = 0; i_inp_stack < inp_size[0]; i_inp_stack++){
					const int kernel_offset = kernel_stack_offset  + i_inp_stack*kernel_size_x*kernel_size_y;
					const int error_out_offset = i_inp_stack * error_out_size + output_batch_offset;
					//std::cout << error_inp_offset << " " << error_inp.get_data().size()<< " " << kernel_offset << " "<< kernels.get_data().size() << " "  << error_out_offset <<" " <<  error_out.get_data().size() << std::endl;
					Convolution2D(&error_inp.get_data()[error_inp_offset], out_size[2],out_size[1], &rotated_kernels.get_data()[kernel_offset], kernel_size_x,kernel_size_y, backwards_c_method,kernel_size_x/2, kernel_size_y/2, &error_out.get_data()[error_out_offset]);
					//std::cout << i_batch << " " <<i_pics << " " << i_kernel << "	" << batch_offset << " " <<index << " " << index_kernel << " " << error_index << " " << error_inp.getDimensionSize(0) << " " << error_inp.getDimensionSize(1) << std::endl;

			}
		}
	}
}

void Convolution2DLayer::print_weights(){
	/*std::cout << std::endl;
	for( int i_kernel = 0; i_kernel< kernels.size(); i_kernel++){
		std::cout << "Kernel number: " << i_kernel << std::endl;
		for( int i_ky = 0; i_ky < kernel_size; i_ky++){

			for( int i_kx = 0; i_kx < kernel_size; i_kx++){
				std::cout << kernels[i_kernel][i_ky*kernel_size + i_kx] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}*/
}

void Convolution2DLayer::resize_input_matrix(){
	if( input.getDimensionSize(0) != input_matrix.getDimensionSize(0)/inp_size[0]){
		//if( c_method == cutoff)
		//	input_matrix = VectorNd<float>(3,input.getDimensionSize(0) * inp_size[0], kernels.getDimensionSize(1),(inp_size[2]-(kernel_size-1))* (inp_size[1]-(kernel_size-1)));
		//if( c_method == valid)
		//	input_matrix = VectorNd<float>(3,input.getDimensionSize(0) * inp_size[0], kernels.getDimensionSize(1),(inp_size[2]+(kernel_size-1))* (inp_size[1]+(kernel_size-1)) );
		//if( c_method == zero_padded || c_method == wrap_around)
		//	input_matrix = VectorNd<float>(3,input.getDimensionSize(0) * inp_size[0], kernels.getDimensionSize(1),inp_size[1]*inp_size[2]);
		input_matrix = VectorNd<float>( 3, input.getDimensionSize(0) * inp_size[0], out_size[1]*out_size[2], kernels.getDimensionSize(1) );
		//input_matrix = VectorNd<float>( 3, input.getDimensionSize(0) * inp_size[0],kernels.getDimensionSize(1), out_size[1]*out_size[2]);

	}
}

void Convolution2DLayer::resize_error_matrix(){
	if( error_inp.getDimensionSize(0) != error_matrix.getDimensionSize(0)/out_size[0]){
		error_matrix = VectorNd<float>(3,output.getDimensionSize(0)* out_size[0], kernels.getDimensionSize(1), inp_size[1]*inp_size[2]);
	}

}


void Convolution2DLayer::fill_input_matrix(){
/*	resize_input_matrix();

	for( int i_pic = 0 ; i_pic < input.getDimensionSize(0)*inp_size[0]; i_pic++){
		const int pic_offset = i_pic* (input_matrix.getDimensionSize(2)*input_matrix.getDimensionSize(1));
		const int inp_pic_offset = i_pic/inp_size[0] * input.getDimensionSize(1);
		if( c_method == cutoff){
				//std::vector<float> aa(32);

				for( int i_y = 0 ; i_y < inp_size[1]-(kernel_size-1); i_y++){
					const int y_offset = (i_y*(inp_size[2]-(kernel_size-1))) ;
					for( int i_x = 0; i_x < inp_size[2]-(kernel_size-1); i_x++){
						const int img_offset = + pic_offset+ (y_offset + i_x)*input_matrix.getDimensionSize(2);
						for( int i_kernely = 0; i_kernely < kernel_size; i_kernely++){
							const int kernel_y_offset = img_offset + i_kernely*kernel_size;
							const int inp_kernel_offset = inp_pic_offset + (i_y+i_kernely)*(inp_size[2]) +i_x;
							for( int i_kernelx = 0; i_kernelx < kernel_size; i_kernelx++){
								//const int kernel_offset = kernel_y_offset + i_kernelx
								//int pic = i_pic/inp_size[0];
								//int kernel =  (i_y+i_kernely)*(inp_size[2])+(i_x+i_kernelx) ;
							    // float a = input(2,0,0);
							    //float a = input.get_data()[i_pic/inp_size[0] * input.getDimensionSize(1) + (i_y+i_kernely)*(inp_size[2])+(i_x+i_kernelx)];
							    // float a = aa[i_pic/inp_size[0] * input.getDimensionSize(1) + (i_y+i_kernely)*(inp_size[2])+(i_x+i_kernelx)];
								//sliding_window[i_kernely*kernel_size + i_kernelx] = inp[(i_y+i_kernely)*(inp_size_x)+(i_x+i_kernelx) ];
								//std::cout << i_kernely*kernel_size + i_kernelx <<  " "  << (i_y*(inp_size[2]-(kernel_size-1))) + i_x << std::endl;
								//input_matrix(3,i_pic,i_kernely*kernel_size + i_kernelx, (i_y*(inp_size[2]-(kernel_size-1))) + i_x) = input(2,i_pic/inp_size[0], (i_y+i_kernely)*(inp_size[2])+(i_x+i_kernelx) );
							    //std::cout << "test" << std::endl;
								//std::cout << (i_y*(inp_size[2]-(kernel_size-1)))+ i_x << " " << input_matrix.getDimensionSize(1) << " " << i_kernely*kernel_size + i_kernelx << " " << input_matrix.getDimensionSize(2) << std::endl;
								//input_matrix(3,i_pic, (i_y*(inp_size[2]-(kernel_size-1)))+ i_x, i_kernely*kernel_size + i_kernelx ) = input(2,i_pic/inp_size[0], (i_y+i_kernely)*(inp_size[2])+(i_x+i_kernelx) );

								//input_matrix.get_data()[kernel_y_offset+  i_kernelx  ] = input(2,i_pic/inp_size[0], (i_y+i_kernely)*(inp_size[2])+(i_x+i_kernelx) );

								input_matrix.get_data()[kernel_y_offset+  i_kernelx  ] = input.get_data()[inp_kernel_offset + i_kernelx];
							    //std::cout << "test" << std::endl;
								//input_matrix.get_data()[kernel_y_offset + i_kernelx] = input.get_data()[inp_kernel_offset+i_kernelx];
								//input_matrix.get_data()[pic_offset+ ((i_kernely*kernel_size + i_kernelx )* input_matrix.getDimensionSize(2)) + ((i_y*(inp_size[2]-(kernel_size-1))) + i_x ) ] = input.get_data()[i_pic/inp_size[0] * input.getDimensionSize(1) + (i_y+i_kernely)*(inp_size[2])+(i_x+i_kernelx)];
							}
						}
					}
				}
			}else{
				if( c_method == valid){
					for( int i_y = 0 ; i_y < inp_size[1]+(kernel_size-1); i_y++){
						for( int i_x = 0; i_x < inp_size[2]+(kernel_size-1); i_x++){;
							for( int i_kernely = -kernel_size/2; i_kernely < kernel_size/2 + kernel_size%2; i_kernely++){
								for( int i_kernelx = -kernel_size/2; i_kernelx < kernel_size/2 + kernel_size%2; i_kernelx++){
									int x_k = i_kernelx + kernel_size/2;
									int y_k = i_kernely + kernel_size/2;
									if( i_y + i_kernely < 0 || i_x + i_kernelx < 0 || i_y + i_kernely >= inp_size[1] || i_x + i_kernelx >= inp_size[2]){
										input_matrix(3,i_pic, y_k*kernel_size+ x_k, i_y* inp_size[2] + i_x) = 0.0;
									}else{
										//sliding_window[y_k*kernel_size + x_k] = inp[(i_y+i_kernely)*(inp_size_y)+(i_x+i_kernelx) ];
										input_matrix(3,i_pic, y_k*kernel_size+ x_k, i_y* inp_size[2] + i_x) = input(2,i_pic/inp_size[0],(i_y+i_kernely)*(inp_size[1])+(i_x+i_kernelx)  );

									}
								}
							}
						}
					}
				}else{
					for( int i_y = 0 ; i_y < inp_size[1]; i_y++){
						for( int i_x = 0; i_x < inp_size[2]; i_x++){

							for( int i_kernely = -kernel_size/2; i_kernely < kernel_size/2 + kernel_size%2; i_kernely++){
								for( int i_kernelx = -kernel_size/2; i_kernelx < kernel_size/2 + kernel_size%2; i_kernelx++){
									int x_k = i_kernelx + kernel_size/2;
									int y_k = i_kernely + kernel_size/2;
									//std::cout << i_kernely << " " << i_kernelx << " " << i_y << " " << i_x << " " << x_k << " " << y_k << std::endl;
									if( i_y + i_kernely < 0 || i_x + i_kernelx < 0 || i_y + i_kernely >= inp_size[1] || i_x + i_kernelx >= inp_size[2]){
										input_matrix(3, i_pic/inp_size[0], y_k*kernel_size +x_k, i_y*inp_size[2] + i_x) = 0.0;

										if( c_method == zero_padded){
											//sliding_window[y_k*kernel_size + x_k] = 0.0;
											//std::cout << i_y*inp_size[2] + i_x << " " << y_k*kernel_size +x_k  << " " << kernels.getDimensionSize(1)<< std::endl;
											input_matrix(3, i_pic/inp_size[0], y_k*kernel_size +x_k, i_y*inp_size[2] + i_x) = 0.0;

										}else
										if( c_method == wrap_around){
											int x = (inp_size[2] + ((i_x+i_kernelx)%inp_size[2])) % inp_size[2]; //wrapp around in both dimensions
											int y = (inp_size[1] + ((i_y+i_kernely)%inp_size[1])) % inp_size[1];
											//sliding_window[y_k*kernel_size + x_k] = inp[ y*(inp_size_x)+x ];
											input_matrix(3,i_pic/inp_size[0], y_k*kernel_size +x_k, i_y*inp_size[2] + i_x) = input(2,i_pic/inp_size[0], y*(inp_size[2])+x );

										}
									}else{
										//std::cout << i_kernely << " " << i_kernelx << " " << i_y << " " << i_x << " " << x_k << " " << y_k << std::endl;

										//sliding_window[y_k*kernel_size + x_k] = inp[(i_y+i_kernely)*(inp_size_y)+(i_x+i_kernelx) ];
										input_matrix(3,i_pic/inp_size[0], y_k*kernel_size +x_k, i_y*inp_size[2] + i_x) = input(2,i_pic/inp_size[0], (i_y+i_kernely)*(inp_size[1])+(i_x+i_kernelx) );
									}
								}
							}

						}
					}
				}
			}
	}*/
}

void Convolution2DLayer::fill_error_matrix(){
	/*
	resize_error_matrix();
	crop_method cm;
	if( c_method == cutoff)
		cm = valid;
	if( c_method == zero_padded)
		cm = zero_padded;
	if( c_method == valid)
		cm = cutoff;
	if( c_method == wrap_around)
		cm = wrap_around;

	for( int i_pic = 0 ; i_pic <error_inp.getDimensionSize(0)*out_size[0]; i_pic++){
		if( cm == cutoff){
				for( int i_y = 0 ; i_y < out_size[1]-(kernel_size-1); i_y++){
					for( int i_x = 0; i_x < out_size[2]-(kernel_size-1); i_x++){

						for( int i_kernely = 0; i_kernely < kernel_size; i_kernely++){
							for( int i_kernelx = 0; i_kernelx < kernel_size; i_kernelx++){
								//sliding_window[i_kernely*kernel_size + i_kernelx] = inp[(i_y+i_kernely)*(inp_size_x)+(i_x+i_kernelx) ];
								//std::cout << i_kernely*kernel_size + i_kernelx <<  " "  << (i_y*(inp_size[2]-(kernel_size-1))) + i_x << std::endl;
								error_matrix(3,i_pic,i_kernely*kernel_size + i_kernelx, (i_y*(out_size[2]-(kernel_size-1))) + i_x) = error_inp(2,i_pic/out_size[0], (i_y+i_kernely)*(out_size[2])+(i_x+i_kernelx) );
							}
						}
					}
				}
			}else{
				if( cm == valid){
					for( int i_y = 0 ; i_y < out_size[1]+(kernel_size-1); i_y++){
						for( int i_x = 0; i_x < out_size[2]+(kernel_size-1); i_x++){;
							for( int i_kernely = -kernel_size/2; i_kernely < kernel_size/2 + kernel_size%2; i_kernely++){
								for( int i_kernelx = -kernel_size/2; i_kernelx < kernel_size/2 + kernel_size%2; i_kernelx++){
									int x_k = i_kernelx + kernel_size/2;
									int y_k = i_kernely + kernel_size/2;
									if( i_y + i_kernely < 0 || i_x + i_kernelx < 0 || i_y + i_kernely >= out_size[1] || i_x + i_kernelx >= out_size[2]){
										error_matrix(3,i_pic, y_k*kernel_size+ x_k, i_y* out_size[2] + i_x) = 0.0;
									}else{
										//sliding_window[y_k*kernel_size + x_k] = inp[(i_y+i_kernely)*(inp_size_y)+(i_x+i_kernelx) ];
										error_matrix(3,i_pic, y_k*kernel_size+ x_k, i_y* out_size[2] + i_x) = error_inp(2,i_pic/out_size[0],(i_y+i_kernely)*(out_size[1])+(i_x+i_kernelx)  );

									}
								}
							}
						}
					}
				}else{
					for( int i_y = 0 ; i_y < out_size[1]; i_y++){
						for( int i_x = 0; i_x < out_size[2]; i_x++){

							for( int i_kernely = -kernel_size/2; i_kernely < kernel_size/2 + kernel_size%2; i_kernely++){
								for( int i_kernelx = -kernel_size/2; i_kernelx < kernel_size/2 + kernel_size%2; i_kernelx++){
									int x_k = i_kernelx + kernel_size/2;
									int y_k = i_kernely + kernel_size/2;
									//std::cout << i_kernely << " " << i_kernelx << " " << i_y << " " << i_x << " " << x_k << " " << y_k << std::endl;
									if( i_y + i_kernely < 0 || i_x + i_kernelx < 0 || i_y + i_kernely >= out_size[1] || i_x + i_kernelx >= out_size[2]){
										if( cm == zero_padded){
											//sliding_window[y_k*kernel_size + x_k] = 0.0;
											//std::cout << i_y*out_size[2] + i_x << " " << y_k*kernel_size +x_k  << " " << kernels.getDimensionSize(1)<< std::endl;
											error_matrix(3, i_pic/out_size[0], y_k*kernel_size +x_k, i_y*out_size[2] + i_x) = 0.0;

										}else
										if( cm == wrap_around){
											int x = (out_size[2] + ((i_x+i_kernelx)%out_size[2])) % out_size[2]; //wrapp around in both dimensions
											int y = (out_size[1] + ((i_y+i_kernely)%out_size[1])) % out_size[1];
											//sliding_window[y_k*kernel_size + x_k] = inp[ y*(inp_size_x)+x ];
											error_matrix(3,i_pic/out_size[0], y_k*kernel_size +x_k, i_y*out_size[2] + i_x) = error_inp(2,i_pic/out_size[0], y*(out_size[2])+x );

										}
									}else{
										//std::cout << i_kernely << " " << i_kernelx << " " << i_y << " " << i_x << " " << x_k << " " << y_k << std::endl;

										//sliding_window[y_k*kernel_size + x_k] = inp[(i_y+i_kernely)*(inp_size_y)+(i_x+i_kernelx) ];
										error_matrix(3,i_pic/out_size[0], y_k*kernel_size +x_k, i_y*out_size[2] + i_x) = input(2,i_pic/out_size[0], (i_y+i_kernely)*(out_size[1])+(i_x+i_kernelx) );
									}
								}
							}

						}
					}
				}
			}
	}*/
}

void Convolution2DLayer::process(){
	//resize_input_matrix();
	/*fill_input_matrix();
	for( int i_batch = 0; i_batch < input.getDimensionSize(0); i_batch++){
		for( int i_pics = 0; i_pics < inp_size[0] ; i_pics++){
			cblas_dgemm(CblasRowMajor, CblasNoTrans,CblasTrans, kernels.getDimensionSize(0), input_matrix.getDimensionSize(2), kernels.getDimensionSize(1), 1.0, &kernels(2,0,0), kernels.getDimensionSize(1), &input_matrix(3,i_pics*i_batch,0,0), input_matrix.getDimensionSize(2), 0.0, &output(2,i_batch, i_pics*out_size[1]*out_size[2]) , input_matrix.getDimensionSize(2));
		}
	}*/
	output.fill(0.0);
	const int img_size = inp_size[2]* inp_size[1];
	const int out_img_size = out_size[1] * out_size[2];

	const int kernel_squared = kernel_size_x*kernel_size_y;
	//std::cout << input.getDimensionSize(0) << " " << input.getDimensionSize(1) << std::endl;
	//std::cout << "before conv" << std::endl;
	#pragma omp parallel for
	for( int i_batch = 0; i_batch < input.getDimensionSize(0); i_batch++){
		//#pragma omp parallel for
		//for( int i_pics = 0;i_pics < inp_size[0]; i_pics++){
		//	const int offset_pic = i_pics * img_size;
		const int inp_batch_offset = i_batch * input.getDimensionSize(1);
		const int out_batch_offset = i_batch * output.getDimensionSize(1);
		#pragma omp parallel for
		for( int i_kernels = 0; i_kernels < kernels.getDimensionSize(0); i_kernels++){
			const int kernel_offset = i_kernels*kernels.getDimensionSize(1);
			const int output_offset = out_batch_offset+ i_kernels * out_img_size;
			for( int i_inp_stack = 0; i_inp_stack < inp_size[0]; i_inp_stack++){
				const int kernels_stack_offset = kernel_offset + i_inp_stack*kernel_squared;
				const int inp_offset = i_inp_stack*img_size+inp_batch_offset;

				Convolution2D(&input.get_data()[inp_offset],inp_size[2], inp_size[1], &kernels.get_data()[kernels_stack_offset], kernel_size_x, kernel_size_y, c_method,kernel_size_x/2, kernel_size_y/2, &output.get_data()[output_offset]);


				//Convolution2D(&input.get_data[inp_offset],inp_size[2], inp_size[1], &kernels.get_data()[inp_batch_offset], kernel_size, c_method, &output(2,i_batch, i_kernels*out_img_size));

			//Convolution2D(&input(2,i_batch, offset_pic), inp_size[2], inp_size[1], &kernels(2,i_kernels,0), kernel_size, c_method, &output(2,i_batch, (i_pics+i_kernels)*(out_img_size))  );
				//Convolution2D(&input.get_data()[i_batch*input.getDimensionSize(0)+ offset_pic], inp_size[2], inp_size[1], &kernels.get_data()[i_kernels*kernels.getDimensionSize(0)], kernel_size, c_method, &output.get_data()[i_batch*output.getDimensionSize(0) + (i_pics+i_kernels)*(out_img_size)]  );


			}
		}
	}

}

} /* namespace MRINN */
//BOOST_CLASS_EXPORT(MRINN::Convolution2DLayer)
