/*
 * UpSampling2D.h
 *
 *  Created on: Apr 23, 2018
 *      Author: lennart
 */

#ifndef UPSAMPLING2D_H_
#define UPSAMPLING2D_H_

#include "Layer.h"

namespace MRINN {

class UpSampling2D: public Layer {
public:
	friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
    	//ar & boost::serialization::base_object<Layer>(*this);
    	ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Layer);
    	ar & pooling_size_x;
    	ar & pooling_size_y;
    }
	virtual ~UpSampling2D();
	UpSampling2D(std::vector<int> inp_size, std::vector<int> out_size, int pooling_size_x, int pooling_size_y);

protected:
	int pooling_size_x;
	int pooling_size_y;
	void process();
	void backpropagate();
};

} /* namespace MRINN */

#endif /* UPSAMPLING2D_H_ */
