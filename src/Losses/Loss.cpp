/*
 * Loss.cpp
 *
 *  Created on: Apr 3, 2018
 *      Author: lennart
 */

#include "Loss.h"
#include "mkl.h"

namespace MRINN {
namespace Losses{
Loss::Loss() {


}

Loss::~Loss() {
	// TODO Auto-generated destructor stub
}

/*
 * returns the loss as a number as the return parameter
 * prediction should contain the derivative of the error function after this
 * Overload this to implement different losses
 */
float Loss::get_loss(VectorNd<float> &output, VectorNd<float> &result){
	return 0;
}

}

} /* namespace MRINN */
