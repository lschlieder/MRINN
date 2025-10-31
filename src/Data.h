/*
 * Data.h
 *
 *  Created on: Mar 21, 2018
 *      Author: lennart
 */

#ifndef DATA_H_
#define DATA_H_

namespace MRINN{
    struct Data{
        //int s_inp= inp_size;
        //int s_out= out_size;
		std::vector<float> inp = std::vector<float>(0.0,0.0);
		std::vector<float> out = std::vector<float>(0.0,0.0);
	};
}




#endif /* DATA_H_ */
