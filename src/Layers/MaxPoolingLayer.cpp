/*
 * MaxPoolingLayer.cpp
 *
 *  Created on: Jan 23, 2018
 *      Author: lennart
 */
#include <boost/serialization/export.hpp>
#include "MaxPoolingLayer.h"
#include <stdexcept>
#include <map>
#include <iostream>
#include <cfloat>

namespace MRINN {

MaxPoolingLayer::MaxPoolingLayer(std::vector<int> inp_size, std::vector<int> out_size, int pooling_size_x, int pooling_size_y): Layer(inp_size, out_size) {
	// TODO Auto-generated constructor stub
	this->pooling_size_x = pooling_size_x;
	this->pooling_size_y = pooling_size_y;
	map = std::vector< std::map<int,int> >(0,std::map<int,int>());

	//std::cout << inp_num << " " << out_num
	if( inp_size[2]/pooling_size_x != out_size[2] || inp_size[1]/pooling_size_y != out_size[1] ){
		throw std::invalid_argument("Input size does not match to output size!");
	}
	if( inp_size[0] != out_size[0]){
		throw std::invalid_argument("number of input images is not equal to number of output images");
	}

}


void MaxPoolingLayer::process(){
	//map = std::map<int,int>();
	/*if( map.size()!= input.getDimensionSize(0)){
		map = std::vector< std::map<int,int> >(input.getDimensionSize(0),std::map<int,int>());
	}else{
		for( int i=0; i < map.size(); i++){
			map[i].clear();
		}
	}*/

	//std::cout << map.size() << std::endl;
	//std::cout << out_num << std::endl;
	//std::cout << output.getDimensionSize(0) << " " << output.getDimensionSize(1) << std::endl;
	//std::cout << input.getDimensionSize(0) << " " << input.getDimensionSize(1) << std::endl;

	const int pic_size = inp_size[1]*inp_size[2];
	//#pragma omp parallel for
	for( int i_batch = 0; i_batch < input.getDimensionSize(0); i_batch++){
		//const int batch_offset = i_batch * input.getDimensionSize(0);
		const int input_batch_offset = i_batch * input.getDimensionSize(1);
		const int output_batch_offset = i_batch * output.getDimensionSize(1);
		//#pragma omp parallel for
		for( int i_pic = 0; i_pic < out_size[0]; i_pic++){
			const int pic_offset = i_pic*pic_size;
			const int batchpic_offset = pic_offset + input_batch_offset;
			const int output_batchpic_offset = i_pic*out_size[1]*out_size[2] + output_batch_offset;


			for( int i_outy = 0; i_outy < out_size[1]; i_outy++){
				const int inp_y = i_outy * pooling_size_y;

				for( int i_outx = 0; i_outx < out_size[2]; i_outx++){
					const int inp_x = i_outx * pooling_size_x;
					//int inp_y = i_outy * pooling_size;
					float max_value = -DBL_MAX;
					int inp_num = 0;
					//std::cout << i_batch << " " << i_pic << " " << i_outy << " " << i_outx << " " << batchpic_offset  << std::endl;

					for( int i_pooly = 0; i_pooly < pooling_size_y; i_pooly++){
						const int pool_offset_y = (inp_y + i_pooly)*inp_size[2];
						const int picpool_offset_y = pool_offset_y + pic_offset +inp_x;
						const int batchpicpool_offset = pool_offset_y + batchpic_offset + inp_x ;
						for( int i_poolx = 0; i_poolx < pooling_size_x; i_poolx++){
							float inp_value = input.get_data()[batchpicpool_offset + i_poolx];
							if( inp_value > max_value){
								max_value = inp_value;
								inp_num = picpool_offset_y + i_poolx;
								//inp_num = batchpicpool

							}

						}
					}
					//map[i_batch][inp_num] = output_batchpic_offset + i_outy*( out_size[2]) + i_outx;
					//map[i_batch].insert(std::pair<int,int>(inp_num,output_batchpic_offset + i_outy*( out_size[2]) + i_outx));
					//map[i_batch][inp_num] =

					const int offset_y = i_outy*(out_size[2]) +output_batchpic_offset;
					//std::cout << offset_y << " " << i_outx << " " << output.get_data().size() << std::endl;
					output.get_data()[offset_y+ i_outx] = max_value;
					//std::cout << i_pic * (out_size[1]* out_size[2]) + i_outy * (out_size[2]) + i_outx << std::endl;
				}
			}
		}
	}
	//std::cout << output.getDimensionSize(0) << " " << output.getDimensionSize(1) << std::endl;
	//output.print();

}

void MaxPoolingLayer::backpropagate(){
	//error_out = std::vector<float>(inp_num,0.0);
	//error_out = VectorNd<float>(2,error_inp.getDimensionSize(0),inp_num);
	error_out.fill(0.0);
	//std::cout << "asfnw" << std::endl;
/*
	#pragma omp parallel for
	for( int i_batch = 0; i_batch < error_inp.getDimensionSize(0); i_batch++){
		const int batch_offset = i_batch* error_out.getDimensionSize(1);
		#pragma omp parallel for
		for( int i_error_out = 0; i_error_out < inp_num; i_error_out++){
			std::map<int,int>::iterator it  = map[i_batch].find(i_error_out);
			if( it != map[i_batch].end()){

				error_out.get_data()[batch_offset + i_error_out] = error_inp.get_data()[it->second];

			}


		}
	}*/

	const int pic_size = inp_size[1]*inp_size[2];
	//#pragma omp parallel for
	for( int i_batch = 0; i_batch < input.getDimensionSize(0); i_batch++){
		//const int batch_offset = i_batch * input.getDimensionSize(0);
		const int input_batch_offset = i_batch * input.getDimensionSize(1);
		const int output_batch_offset = i_batch * output.getDimensionSize(1);
		//#pragma omp parallel for
		for( int i_pic = 0; i_pic < out_size[0]; i_pic++){
			const int pic_offset = i_pic*pic_size;
			const int batchpic_offset = pic_offset + input_batch_offset;
			const int output_batchpic_offset = i_pic*out_size[1]*out_size[2] + output_batch_offset;


			for( int i_outy = 0; i_outy < out_size[1]; i_outy++){
				const int inp_y = i_outy * pooling_size_y;

				for( int i_outx = 0; i_outx < out_size[2]; i_outx++){
					const int inp_x = i_outx * pooling_size_x;
					//int inp_y = i_outy * pooling_size;
					float max_value = -DBL_MAX;
					int inp_num = 0;
					//std::cout << i_batch << " " << i_pic << " " << i_outy << " " << i_outx << " " << batchpic_offset  << std::endl;

					for( int i_pooly = 0; i_pooly < pooling_size_y; i_pooly++){
						const int pool_offset_y = (inp_y + i_pooly)*inp_size[2];
						const int picpool_offset_y = pool_offset_y + pic_offset +inp_x;
						const int batchpicpool_offset = pool_offset_y + batchpic_offset + inp_x ;
						for( int i_poolx = 0; i_poolx < pooling_size_x; i_poolx++){
							float inp_value = input.get_data()[batchpicpool_offset + i_poolx];
							if( inp_value > max_value){
								max_value = inp_value;
								inp_num = picpool_offset_y + i_poolx;
								//inp_num = batchpicpool

							}

						}
					}

					//map[i_batch].insert(std::pair<int,int>(inp_num,output_batchpic_offset + i_outy*( out_size[2]) + i_outx));
					//map[i_batch][inp_num] =

					//std::cout << input_batch_offset + inp_num<< " " << output_batchpic_offset + i_outy*( out_size[2]) + i_outx << std::endl;
					error_out.get_data()[input_batch_offset + inp_num] = error_inp.get_data()[output_batchpic_offset + i_outy*( out_size[2]) + i_outx];
					//std::cout << i_pic * (out_size[1]* out_size[2]) + i_outy * (out_size[2]) + i_outx << std::endl;
				}
			}
		}
	}

}

MaxPoolingLayer::~MaxPoolingLayer() {
	// TODO Auto-generated destructor stub
}

} /* namespace MRINN */
//BOOST_CLASS_EXPORT(MRINN::MaxPoolingLayer)
