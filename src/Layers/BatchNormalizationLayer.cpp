/*
 * BatchNormalizationLayer.cpp
 *
 *  Created on: May 14, 2018
 *      Author: lennart
 */

#include <boost/serialization/export.hpp>

#include "BatchNormalizationLayer.h"
#include "../VectorNd.h"

namespace MRINN {

BatchNormalizationLayer::BatchNormalizationLayer(std::vector<int> inp_size, std::vector<int> out_size, float eps):Layer(inp_size,out_size) {
	// TODO Auto-generated constructor stub
	this->eps = eps;
	this->batch_size = 1;

	params = std::vector<float>(inp_size[0]*2);
	gradient = VectorNd<float>(2,1,params.size());
	for( int i =0; i < params.size(); i++){
		//params[i]  = (rand()/(float)RAND_MAX)*0.1;
		params[i] = 1;
		if( i >= params.size()/2){
			params[i]=0;
		}
		//std::cout << params[i] << " " ;
		gradient.get_data()[i] = 0;
	}
	//std::cout << std::endl;

	if( inp_size[0] != out_size[0] || inp_size[1] != out_size[1] || inp_size[2] != out_size[2]){
		throw std::invalid_argument("Invalid argument!! Make sure that input size == output size");
	}

	create_normalization_primitive(1);
}

void BatchNormalizationLayer::create_normalization_primitive(int batch_number){
	//params = std::vector<float>(inp_size[0]*batch_number*2);
	size_t * inputSize = new size_t[4];
	inputSize[0] = inp_size[2];
	inputSize[1] = inp_size[1];
	inputSize[2] = inp_size[0];
	inputSize[3] = batch_number;

	size_t inputStride[4] = {1,inputSize[0],inputSize[0]*inputSize[1],inputSize[0]*inputSize[1]*inputSize[2]};

	//std::cout << inputSize[0] << " " << inputSize[1] << " " <<inputSize[2] << " " <<inputSize[3] << std::endl;
	dnnLayoutCreate_F32(&data_layout, 4, inputSize,inputStride);
	if( dnnPrimitiveAttributesCreate_F32(&attributes)!= E_SUCCESS){
		std::cout << "error" << std::endl;
	}
    dnnError_t err = dnnBatchNormalizationCreateForward_F32(&pBatchNormalization, attributes, data_layout, eps);
    if( err != E_SUCCESS){
    	throw std::invalid_argument("Could not create batch normalization forward primitive");
    }
    dnnError_t err_b = dnnBatchNormalizationCreateBackwardData_F32(&pBatchNormalizationData, attributes, data_layout, eps);
    if( err_b != E_SUCCESS){
    	throw std::invalid_argument("Could not create batch normalization backwards data primitive");
    }
    dnnError_t err_b_s = dnnBatchNormalizationCreateBackwardScaleShift_F32(&pBatchNormalizationScaleShift, attributes, data_layout, eps);
    if( err_b_s != E_SUCCESS){
    	throw std::invalid_argument("Could not create batch normalization backwards scale shift primitive");
    }


    dnnLayout_t lt_scaleshift, lt_workspace,lt_src, lt_dst;
    dnnLayoutCreateFromPrimitive_F32(&lt_scaleshift, pBatchNormalization, dnnResourceScaleShift);
    dnnLayoutCreateFromPrimitive_F32(&lt_workspace, pBatchNormalization, dnnResourceWorkspace);
    dnnLayoutCreateFromPrimitive_F32(&lt_src, pBatchNormalization, dnnResourceSrc);
    dnnLayoutCreateFromPrimitive_F32(&lt_dst, pBatchNormalization, dnnResourceDst);

    dnnLayout_t lt_scaleshift_bprop, lt_diffdst_bprop, lt_workspace_bprop,lt_src_bprop;
    dnnLayoutCreateFromPrimitive_F32(&lt_workspace_bprop, pBatchNormalizationScaleShift, dnnResourceWorkspace);
    dnnLayoutCreateFromPrimitive_F32(&lt_diffdst_bprop, pBatchNormalizationScaleShift, dnnResourceDiffDst);
    dnnError_t err_layout = dnnLayoutCreateFromPrimitive_F32(&lt_scaleshift_bprop, pBatchNormalizationScaleShift, dnnResourceDiffScaleShift);
    if( err_layout != E_SUCCESS){
    	std::cout << "error creating the layout" << std::endl;
		if( err_layout == E_MEMORY_ERROR){
			std::cout << "Memory Error" << std::endl;
		}
		if( err_layout == E_UNIMPLEMENTED){
			std::cout << "Unimplemented" << std::endl;
		}
		if( err_layout == E_UNSUPPORTED_DIMENSION){
			std::cout << "Unsupported dimension" << std::endl;
		}
		if( err_layout == E_INCORRECT_INPUT_PARAMETER){
			std::cout << "Incorrect input parameter" << std::endl;
		}

    }
    dnnLayoutCreateFromPrimitive_F32(&lt_src_bprop, pBatchNormalizationScaleShift, dnnResourceSrc);

    //std::cout << dnnLayoutGetMemorySize_F32(lt_scaleshift)/4 <<" " << params.size() << " " << gradient.get_data().size() <<std::endl;

    workspace= std::vector<float>(dnnLayoutGetMemorySize_F32(lt_workspace)/4,0.0);
    //std::cout << dnnLayoutGetMemorySize_F32(lt_workspace)/4 << " " << workspace.size() << std::endl;
    //std::cout << dnnLayoutGetMemorySize_F32(lt_src)/4 << " " << input.get_data().size()  << std::endl;
    //std::cout << dnnLayoutGetMemorySize_F32(lt_dst)/4 << " " << output.get_data().size() << std::endl;

    //std::cout << "bprop" << std::endl;
    //std::cout << dnnLayoutGetMemorySize_F32(lt_workspace_bprop)/4 << " " << workspace.size() << std::endl;
    //std::cout << dnnLayoutGetMemorySize_F32(lt_diffdst_bprop)/4 << " " << error_out.get_data().size() <<std::endl;
    //std::cout << dnnLayoutGetMemorySize_F32(lt_src_bprop)/4 << " " << input.get_data().size() << std::endl;
    //dnnLayoutGetMemorySize_F32(lt_scaleshift_bprop);

   // std::cout << dnnLayoutGetMemorySize_F32(lt_scaleshift_bprop)/4 << " "  << std::endl;

    batch_size = batch_number;


}

void BatchNormalizationLayer::set_var_parameters(std::vector<float> & pars){

	params.swap(pars);
	///std::cout << std::endl;
	//for( int i =0; i < params.size(); i++){
	//	std::cout << params[i] << " " ;
	//}
	//std::cout << std::endl;
}

std::vector<float> & BatchNormalizationLayer::get_var_parameters(){
	return params;
}

VectorNd<float> & BatchNormalizationLayer::get_parameter_gradient(VectorNd<float> & backprop_vec){
	resBatchNormalization[dnnResourceSrc] = &input.get_data()[0];
	resBatchNormalization[dnnResourceDiffDst] = &backprop_vec.get_data()[0];
	resBatchNormalization[dnnResourceWorkspace] = &workspace[0];

	resBatchNormalization[dnnResourceDiffScaleShift] = &gradient.get_data()[0];


	dnnError_t err = dnnExecute_F32(pBatchNormalizationScaleShift, (void**) resBatchNormalization);
	if( err != E_SUCCESS){
		std::cout <<"Error while executing gradient calculation for scale shift" << std::endl;
		if( err == E_MEMORY_ERROR){
			std::cout << "Memory Error" << std::endl;
		}
		if( err == E_UNIMPLEMENTED){
			std::cout << "Unimplemented" << std::endl;
		}
		if( err == E_UNSUPPORTED_DIMENSION){
			std::cout << "Unsupported dimension" << std::endl;
		}
		if( err == E_INCORRECT_INPUT_PARAMETER){
			std::cout << "Incorrect input parameter" << std::endl;
		}
	}
	return gradient;

}


void BatchNormalizationLayer::process(){
	if( input.getDimensionSize(0) != batch_size){
		create_normalization_primitive(input.getDimensionSize(0));
	}
	//input
	resBatchNormalization[dnnResourceSrc] = &input.get_data()[0];
	//params
	resBatchNormalization[dnnResourceWorkspace] = &workspace[0];
	resBatchNormalization[dnnResourceScaleShift] = &params[0];
	//output
	resBatchNormalization[dnnResourceDst] = &output.get_data()[0];
	dnnError_t err = dnnExecute_F32(pBatchNormalization, (void **) resBatchNormalization);
	if( err != E_SUCCESS){
		std::cout <<"Error while executing BatchNormalization forward propagation" << std::endl;
	}
}

void BatchNormalizationLayer::backpropagate(){
	if( error_inp.getDimensionSize(0) != batch_size){
		create_normalization_primitive(error_inp.getDimensionSize(0));
	}
	//input
	resBatchNormalization[dnnResourceDiffDst] = &error_inp.get_data()[0];
	//params
	resBatchNormalization[dnnResourceWorkspace] = &workspace[0];
	resBatchNormalization[dnnResourceScaleShift] = &params[0];
	//output
	resBatchNormalization[dnnResourceDiffSrc] = &error_out.get_data()[0];
	dnnError_t err = dnnExecute_F32(pBatchNormalizationData, (void**) resBatchNormalization);
	if( err != E_SUCCESS){
		std::cout <<"Error while executing BatchNormalization backwards propagation" << std::endl;
	}
}

BatchNormalizationLayer::~BatchNormalizationLayer() {
	// TODO Auto-generated destructor stub
}


} /* namespace MRINN */
//BOOST_CLASS_EXPORT(MRINN::BatchNormalizationLayer)
