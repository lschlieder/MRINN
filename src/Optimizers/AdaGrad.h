/*
 * AdaGrad.h
 *
 *  Created on: Mar 22, 2018
 *      Author: lennart
 */

#ifndef ADAGRAD_H_
#define ADAGRAD_H_

#include "Optimizer.h"
#include <vector>

using namespace std;

namespace MRINN {
	namespace Optimizers{
		class AdaGrad: public Optimizer {
		public:
			AdaGrad(float learning_rate );
			virtual ~AdaGrad();
			virtual void set_size(int size);
			vector<float> & get_updated_parameters(vector<float> & parameters, vector<float> & gradient);
		protected:
			vector<float> gradient_vector;
			vector<float> gradient_squared;
			float learning_rate;

		};
	} /*namespace Optimizers */
} /* namespace MRINN */

#endif /* ADAGRAD_H_ */
