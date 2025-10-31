/*
 * AdaMax.h
 *
 *  Created on: Apr 17, 2018
 *      Author: lennart
 */

#ifndef ADAMAX_H_
#define ADAMAX_H_

#include "Optimizer.h"

namespace MRINN {
namespace Optimizers{


class AdaMax: public Optimizer {
public:
	AdaMax(float learning_rate, float momentum_one, float momentum_two);
	virtual ~AdaMax();

	vector<float> & get_updated_parameters(vector<float> &parameters, vector<float> &gradient);
	void set_size(int size);
protected:
	float learning_rate;
	float momentum_one;
	float momentum_one_power_t = 1;
	float momentum_two;
	vector<float> momentum_vec_one;
	vector<float> momentum_vec_two;
	vector<float> gradient_squared;
};

} /* namespace Optmimizers */
} /* namespace MRINN */

#endif /* ADAMAX_H_ */
