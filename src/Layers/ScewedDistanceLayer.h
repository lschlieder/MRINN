/*
 * ScewedDistanceLayer.h
 *
 *  Created on: Jan 9, 2018
 *      Author: lennart
 */

#ifndef SCEWEDDISTANCELAYER_H_
#define SCEWEDDISTANCELAYER_H_

#include "Layer.h"
#include "../VectorNd.h"

namespace MRINN {

class ScewedDistanceLayer: public Layer {
public:
	friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
    	//ar & boost::serialization::base_object<Layer>(*this);
    	ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Layer);
    	ar & ones;
    	ar & diff_vec;
    	ar & diff_mat;
    	ar & Positions;
    	ar & Weights;
    	ar & Jacobian;
    	ar & temp;
    }
	void set_var_parameters(std::vector<float> & pars);
	std::vector<float> & get_var_parameters();
	VectorNd<float> & get_parameter_gradient(VectorNd<float> & backprop_vec);
	ScewedDistanceLayer( std::vector<int> inp_size, std::vector<int> out_size);

	void print_weights();

	virtual ~ScewedDistanceLayer();

protected:
	//std::vector< std::vector<float> > Positions;
	//std::vector< std::vector<float> > Weights;

	std::vector<float> ones;
	VectorNd<float> diff_vec;
	VectorNd<float> diff_mat;

	VectorNd<float> Positions;
	VectorNd<float> Weights;

	VectorNd<float> Jacobian;


	VectorNd<float> temp;

	void process();
	void backpropagate();



};

} /* namespace MRINN */

#endif /* SCEWEDDISTANCELAYER_H_ */
