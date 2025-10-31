/*
 * SGD.h
 *
 *  Created on: Mar 21, 2018
 *      Author: lennart
 */

#ifndef SGD_H_
#define SGD_H_

#include "Optimizer.h"
#include <vector>
using namespace std;
namespace MRINN {
	namespace Optimizers{
		class SGD: public Optimizer {
		public:
			SGD(float learning_rate, float momentum, bool nesterov);
			virtual ~SGD();
			void set_size(int size);
		    //vector<float> & get_updated_parameters(vector<float> & parameters, vector<float> & gradient);
			vector<float> & get_updated_parameters(vector<float> & parameters, vector<float> &gradient);

		protected:
			float learning_rate = 0.05;
			float momentum = 0.9;
			bool nesterov= false;
			vector<float> vec_momentum; //holds the momentum
			vector<float> old_momentum; //needed for nesterov momemtum
		};
	}

} /* namespace MRINN */

#endif /* SGD_H_ */
