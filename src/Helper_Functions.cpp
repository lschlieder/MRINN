/*
 * Helper_Functions.cpp
 *
 *  Created on: Nov 15, 2017
 *      Author: lennart
 */
#include <vector>
#include "VectorNd.h"
namespace MRINN{

/*	template<typename T>
//	std::vector<T> v(int dims,  T* size){
//		std::vector<T> vec(dims);
//		for(int i = 0; i < dims;i++){
//			vec[i] = size[i];
//		}
//		return vec;
//	}
*/
	/*std::vector<int> v(int dims,  int* size){
		std::vector<int> vec(dims);
		for(int i = 0; i < dims;i++){
			vec[i] = size[i];
		}
		return vec;
	}*/
	
	template <class T>
	T max(VectorNd<T> arr){
        T max = arr.get_data()[0];
        for( int i =0; i < arr.get_data().size(); i++){
            if (arr.get_data()[i] > max)
                max = arr.get_data()[i];
        }
        return max ;
    }
    
    template <class T>
	T min(VectorNd<T> arr){
        T min = arr.get_data()[0];
        for( int i =0; i < arr.get_data().size(); i++){
            if (arr.get_data()[i] < min)
                min = arr.get_data()[i];
        }
        return min;
    }
}



