/*
 * Adam.h
 *
 *  Created on: Apr 17, 2018
 *      Author: lennart
 */

#ifndef ADAM_H_
#define ADAM_H_

#include "Optimizer.h"
#include <vector>

using namespace std;
namespace MRINN {
namespace Optimizers{
class Adam: public Optimizer {
public:
	Adam(double learning_rate, double momentum_one, double momentum_two);
	virtual ~Adam();

	vector<float> & get_updated_parameters(vector<float> &parameters, vector<float> &gradient);
	void set_size(int size);
protected:
	double learning_rate;
	double momentum_one;
	double momentum_one_power_t = 1;
	double momentum_two;
	double momentum_two_power_t = 1;
	vector<float> momentum_vec_one;
	vector<float> momentum_vec_two;
	vector<float> gradient_squared;

};


} /* namespace Optimizers */
} /* namespace MRINN */

#endif /* ADAM_H_ */
