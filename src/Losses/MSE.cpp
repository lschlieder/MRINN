/*
 * MSE.cpp
 *
 *  Created on: Apr 3, 2018
 *      Author: lennart
 */

#include "MSE.h"
#include "mkl.h"
#include "iostream"

namespace MRINN {
namespace Losses{
MSE::MSE() {
	// TODO Auto-generated constructor stub

}

MSE::~MSE() {
	// TODO Auto-generated destructor stub
}
float MSE::get_loss(VectorNd<float> &output, VectorNd<float> &result) //returns the loss as return parameter. prediction should contain the derivative of the error as a pass through parameter
{
	cblas_saxpy(output.get_data().size(), -1.0, & output.get_data()[0], 1.0, &result.get_data()[0], 1.0);
	//std::cout << result.get_data()[0] << " " << result.getDimensionSize(1) <<std::endl;

	float squared_error= cblas_sdot(output.get_data().size(), &result(2,0,0), 1.0, &result(2,0,0), 1.0)/((float) result.getDimensionSize(0)* result.getDimensionSize(1));
	cblas_sscal(result.get_data().size(), 2/((float)result.getDimensionSize(1)*result.getDimensionSize(0)), &result.get_data()[0], 1);

	return squared_error;
}

}
} /* namespace MRINN */
