/*
 * UpSampling2D.cpp
 *
 *  Created on: Apr 23, 2018
 *      Author: lennart
 */
#include <boost/serialization/export.hpp>
#include "UpSampling2D.h"

namespace MRINN {

UpSampling2D::UpSampling2D(std::vector<int> inp_size, std::vector<int> out_size, int pooling_size_x, int pooling_size_y):Layer(inp_size,out_size) {
	this->pooling_size_x = pooling_size_x;
	this->pooling_size_y = pooling_size_y;

}

UpSampling2D::~UpSampling2D() {
	// TODO Auto-generated destructor stub
}

void UpSampling2D::process(){
	for( int i_batch = 0; i_batch < input.getDimensionSize(0);i_batch++){
		const int batch_offset = i_batch*input.getDimensionSize(1);
		const int output_batch_offset = i_batch*output.getDimensionSize(2);
		for( int i_pics = 0; i_pics < inp_size[0];i_pics++){
			const int batch_pic_offset = batch_offset + i_pics * inp_size[1]*inp_size[2];
			const int output_batch_pic_offset = output_batch_offset + i_pics*out_size[1]*out_size[2];
			for( int iy_inp = 0; iy_inp < inp_size[1]; iy_inp++){
				const int iy_inp_offset = batch_pic_offset + iy_inp*inp_size[2];
				const int y_out = iy_inp*pooling_size_y;

				for( int ix_inp = 0; ix_inp < inp_size[2]; ix_inp++){
					const int x_out = ix_inp*pooling_size_x;

					const int inp_offset = iy_inp_offset + ix_inp;
					for( int iy_out = 0; iy_out < pooling_size_y; iy_out++){
						const int iy_out_offset = output_batch_pic_offset + (y_out+iy_out) * out_size[2];
						for( int ix_out = 0; ix_out < pooling_size_x; ix_out++){
							const int ix_out_offset = iy_out_offset + (x_out+ix_out);
							output.get_data()[ix_out_offset] = input.get_data()[inp_offset];
						}
					}
				}
			}
		}
	}

}

void UpSampling2D::backpropagate(){

}

} /* namespace MRINN */
//BOOST_CLASS_EXPORT(MRINN::UpSampling2D)
