/*
 * BinaryCrossentropy.h
 *
 *  Created on: Apr 17, 2018
 *      Author: lennart
 */

#ifndef BINARYCROSSENTROPY_H_
#define BINARYCROSSENTROPY_H_

#include "Loss.h"
#include "../VectorNd.h"

namespace MRINN {
namespace Losses{
class BinaryCrossentropy: public Loss {
public:
	BinaryCrossentropy();
	virtual ~BinaryCrossentropy();

	float get_loss(VectorNd<float> &output, VectorNd<float> &result);
};

} /* namespace Losses*/
} /* namespace MRINN */

#endif /* BINARYCROSSENTROPY_H_ */
