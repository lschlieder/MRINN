/*
 * VectorNd.cpp
 *
 *  Created on: Jan 24, 2018
 *      Author: lennart
 */

#include "VectorNd.h"
#include <stdexcept>
#include <iostream>


namespace MRINN {

template < typename T>
VectorNd< T>::VectorNd(){
	dims = 0;
	dim_vec = std::vector<int>(dims,0);
	data = std::vector<T>(dims);
}

template< typename T>
VectorNd<T>::VectorNd(VectorNd<T>  * vec){
	dims = vec.dims;
	length = vec.length;
	data = std::vector<T>(vec.get_data());
	dim_vec = std::vector<int>(vec.get_dim_vec());
}

template < typename T>
VectorNd< T>::VectorNd(int dims,...) {
	va_list args;
	va_start(args,dims);
	this->dims = dims;
	dim_vec = std::vector<int>(dims,0);
	int vector_length = 1;
	for( int i = 0; i < dims; i++){
		dim_vec[i] = va_arg(args, int);
		if( dim_vec[i] <= 0){
			vector_length = 0;
			for( int j = 0; j < dims; j++){
				dim_vec[i] = 0;
			}
			break;
			//throw std::invalid_argument("input dimensions must be greater than zero!");
		}
		vector_length = vector_length * dim_vec[i];
	}
	length = vector_length;
	data = std::vector<T>(vector_length);
	va_end(args);
}

template < typename T>
T & VectorNd<T>::operator()(int dims,...){
	va_list args;
	va_start(args,dims);
	std::vector<int> access_vec(dims);

	int inp_num;
	int access_num =0;
	int dim_num = 1;
	for( int i =0; i <dims;i++){
		access_vec[i] = va_arg(args,int);
		inp_num = access_vec[i];
		//inp_num = va_arg(args,int);
		if( inp_num >= dim_vec[i] || inp_num < 0){
			throw std::invalid_argument("Out of bounds error for vector access!");
		}
		//dim_num = dim_num/dim_vec[dims-1-i];
		//access_num += inp_num * dim_num;
	}
	//int access_num = 0;
	//int dim_num = 1;
	//dim_num = 1;
	for( int i=dims-1; i >=0; i--){
	//for( int i=dims;  i--;){
		access_num += access_vec[i] * dim_num;
		dim_num = dim_num* dim_vec[i];
	}
	va_end(args);
	return data[access_num];
}

template < typename T>
void VectorNd<T>::fill(T inp){
	std::fill(data.begin(), data.end(), inp);
}

template <typename T>
void VectorNd<T>::print(){
	//std::cout << std::endl;
	for( int i = 0; i < data.size(); i++){
		int dim = 1;
		for( int j = 1; j < dim_vec.size();j++){
			dim = dim*dim_vec[j];
			if( i % dim ==0 ){
				std::cout << std::endl;
			}
		}
		std::cout << data[i] << " ";
		
	}
	std::cout << std::endl;
}

template < typename T>
int VectorNd<T>::getDimensionSize(int dim){
	return dim_vec[dim];
}


template < typename T>
std::vector<T> & VectorNd<T>::get_data(){
	return data;
}

template < typename T>
std::vector<int> & VectorNd<T>::get_dim_vec(){
	return dim_vec;
}

template < typename T>
VectorNd<T>::~VectorNd() {
	// TODO Auto-generated destructor stub
}

} /* namespace MRINN */
