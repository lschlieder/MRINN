/*
 * AdaDelta.h
 *
 *  Created on: Mar 29, 2018
 *      Author: lennart
 */

#ifndef ADADELTA_H_
#define ADADELTA_H_

#include "Optimizer.h"

using namespace std;

namespace MRINN {
namespace Optimizers{
class AdaDelta: public Optimizer {
public:
	AdaDelta(float learning_rate, float momentum);
	virtual ~AdaDelta();
	virtual void set_size(int size);
	vector<float> & get_updated_parameters(vector<float> & parameters, vector<float> & gradient);
protected:
	vector<float> gradient_vector;
	vector<float> gradient_squared;

	vector<float> parameters_squared;
	vector<float> parameters_momentum_vec;
	float momentum;
	float learning_rate;
};

} /* namespace Optimizers */

} /* namespace MRINN */

#endif /* ADADELTA_H_ */
