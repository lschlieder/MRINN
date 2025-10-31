/*
 * CrossEntropy.h
 *
 *  Created on: Apr 5, 2018
 *      Author: lennart
 */

#ifndef CROSSENTROPY_H_
#define CROSSENTROPY_H_

#include "Loss.h"

namespace MRINN {
namespace Losses {
class CrossEntropy: public Loss {
public:
	float get_loss(VectorNd<float> &output, VectorNd<float> &result); //returns the loss as return parameter. result should contain the derivative of the error as a pass through parameter
	CrossEntropy();
	virtual ~CrossEntropy();
};

}
} /* namespace MRINN */

#endif /* CROSSENTROPY_H_ */
