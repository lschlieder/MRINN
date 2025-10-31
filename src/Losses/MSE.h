/*
 * MSE.h
 *
 *  Created on: Apr 3, 2018
 *      Author: lennart
 */

#ifndef MSE_H_
#define MSE_H_

#include "Loss.h"

namespace MRINN {
namespace Losses{
class MSE: public Loss {
public:
	float get_loss(VectorNd<float> &output, VectorNd<float> &result); //returns the loss as return parameter. result should contain the derivative of the error as a pass through parameter
	MSE();
	virtual ~MSE();
};
} /* namespace Losses */
} /* namespace MRINN */

#endif /* MSE_H_ */
