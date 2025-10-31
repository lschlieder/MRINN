/*
 * Loss.h
 *
 *  Created on: Apr 3, 2018
 *      Author: lennart
 */

#ifndef LOSS_H_
#define LOSS_H_

#include "../VectorNd.h"
namespace MRINN {
namespace Losses{
class Loss {
public:
	virtual float get_loss(VectorNd<float> &output, VectorNd<float> &result); //returns the loss as return parameter. prediction should contain the derivative of the error as a pass through parameter
	Loss();
	virtual ~Loss();
};

} /* namespace Losses */
} /* namespace MRINN */

#endif /* LOSS_H_ */
