/*
 * Copyright 2019 <copyright holder> <email>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SPARSE_SOFTMAX_CROSSENTROPY_H
#define SPARSE_SOFTMAX_CROSSENTROPY_H

#include "Loss.h"
namespace MRINN{
namespace Losses{
/**
 * @todo write docs
 */
class Sparse_Softmax_Crossentropy : public Loss{
    public:
        
        Sparse_Softmax_Crossentropy();
        virtual ~Sparse_Softmax_Crossentropy();
        VectorNd<float> probabilities;
    	float get_loss(VectorNd<float> &output, VectorNd<float> &result); //returns the loss as return parameter. result should contain the derivative of the error as a pass throug parameter

};

}
}

#endif // SPARSE_SOFTMAX_CROSSENTROPY_H
