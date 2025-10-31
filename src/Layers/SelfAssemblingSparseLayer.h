/*
 * SelfAssemblingSparseLayer.h
 *
 *  Created on: Apr 6, 2018
 *      Author: lennart
 */

#ifndef SELFASSEMBLINGSPARSELAYER_H_
#define SELFASSEMBLINGSPARSELAYER_H_

#include "Layer.h"
#include <vector>
#include "../VectorNd.h"

namespace MRINN {

class SelfAssemblingSparseLayer: public Layer {
public:
	void set_var_parameters(std::vector<float> & pars);
	std::vector<float> & get_var_parameters();

	VectorNd<float> & get_parameter_gradient(VectorNd<float> & backprop_vec);
	SelfAssemblingSparseLayer( std::vector<int> inp_size, std::vector<int> out_size);
	virtual ~SelfAssemblingSparseLayer();
protected:
	VectorNd<float> input_after_connection;
	VectorNd<float> connection_matrix;
	VectorNd<float> weight_matrix;
	void apply_mask();
	void backpropagate();
	void process();
};

} /* namespace MRINN */

#endif /* SELFASSEMBLINGSPARSELAYER_H_ */
