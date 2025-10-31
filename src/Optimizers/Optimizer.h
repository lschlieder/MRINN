/*
 * Optimizer.h
 *
 *  Created on: Mar 21, 2018
 *      Author: lennart
 */

#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include <vector>

using namespace std;

static const float EPSILON = 0.00000001f;

namespace MRINN {
	namespace Optimizers{
		class Optimizer{
		public:
			Optimizer();
			virtual ~Optimizer();
			virtual vector<float> & get_updated_parameters(vector<float> & parameters, vector<float> & gradient);
			virtual void set_size(int size);
		};

	} /* namespace MRINN */
} /* namespace MRINN */
#endif /* OPTIMIZER_H_ */
