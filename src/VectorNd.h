/*
 * VectorNd.h
 *
 *  Created on: Jan 24, 2018
 *      Author: lennart
 */

#ifndef VECTORND_H_
#define VECTORND_H_

#include <stdarg.h>
#include <vector>
#include <algorithm>
//#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
//#include <boost/serialization/export.hpp>

namespace MRINN {

template < typename T>
class VectorNd{
public:
	// needed for serialization
	friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & length;
        ar & dims;
        ar & dim_vec;
        ar & data;
    }

	VectorNd(int dims,...);
	VectorNd(VectorNd<T> * vec);
	VectorNd();
	virtual ~VectorNd();

	T & operator()(int dims, ...);
	int getDimensionSize(int dim);
	std::vector<int> & get_dim_vec();
	void fill(T inp);
	void print();


	std::vector<T> & get_data();



protected:
	int length;
	int dims;
	std::vector<int> dim_vec;
	std::vector<T> data;


};



} /* namespace MRINN */
#include "VectorNd.tpp"
#endif /* VECTORND_H_ */
