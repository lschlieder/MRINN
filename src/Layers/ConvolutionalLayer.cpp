/*
 * ConvolutionalLayer.cpp

 *
 *  Created on: Dec 4, 2017
 *      Author: lennart
 */

#include <boost/serialization/export.hpp>
#include <cmath>
#include "ConvolutionalLayer.h"


namespace MRINN {


ConvolutionalLayer::ConvolutionalLayer(std::vector<int> inp_size, std::vector<int> out_size, int kernel_size, int kernel_num,  dnnBorder_t border_type, bool use_bias):Layer(inp_size, out_size) {
	// TODO Auto-generated constructor stub
	//this->kernel_size_x = kernel_size_x;
	this->kernel_size = kernel_size;
	this->kernel_number = kernel_num;
	this->border_method = border_type;
	this->use_bias = use_bias;
	if( out_size[0] != kernel_number){
		throw std::invalid_argument("output dimension does not check with kernel number. out_dim[0] must be == kernel_num");
	}

	if( inp_size.size() != out_size.size()){
		throw std::invalid_argument("input and output dimensions do not match. There should be output dimension == input dimension");
	}

	/*for( int i = inp_size.size() -1; i > 0; i--){
		if( inp_size[i] != out_size[i]){
			throw std::invalid_argument("one of the output dimension has the wrong size. Please check that newside = oldside ");
		}
	}*/

	dimension = inp_size.size()+1;
	attributes = NULL;
	batch_size = 0;

	int kernel_arr_size = pow(kernel_size,dimension-2)*(inp_size[0] * out_size[0]);
	kernels = std::vector<float>( kernel_arr_size,1.0);
	if( use_bias){
		bias = std::vector<float>(kernel_number,0.0);
	}
	params = std::vector<float>(kernel_arr_size + use_bias* kernel_number,0.0);
	for( unsigned int i_params = 0; i_params < params.size();i_params++){
		if( i_params < kernels.size()){
			//kernels[i_params] =(((float)rand()/RAND_MAX) -0.5) *2.0* sqrt(6.0/(inp_num+out_num)) ;
            kernels[i_params] =(((float)rand()/RAND_MAX) -0.5) /6.0 ;
            /*if( i_params == 0)
                kernels[i_params] = 1.0;
            else
                kernels[i_params] = 0.0;*/
			params[i_params] = kernels[i_params];
		}else{
			//bias[i_params-kernels.size()] = ((float)rand()/RAND_MAX -0.5)/10.0;
			//bias[i_params-kernels.size()] = 1.0;
			//std::cout << bias[i_params] << std::endl;
            bias[i_params - kernels.size()] = 0.0;
            //bias[i_params-kernels.size()] = 1.0;
			params[i_params] = bias[i_params-kernels.size()];
		}

	}

	gradient = VectorNd<float>(2,1,params.size());
   // int s_kernel = pow(kernel_size,dimension-2)* inp_size[0] * batch_number;
   // kernels = std::vector<float>(s_kernel);
	setup_convolution(1);


}



ConvolutionalLayer::~ConvolutionalLayer() {
	// TODO Auto-generated destructor stub
}

void ConvolutionalLayer::setup_convolution(int batch_number){

	/* fill input sizes with the new batch_number*/
	//size_t outputSize[dimension];
	//size_t inputSize[dimension];
	//size_t filterSize[dimension];

	size_t * outputSize = new size_t[dimension];
	size_t * inputSize = new size_t[dimension];
	size_t * filterSize = new size_t[dimension];
	size_t * biasSize = new size_t[dimension];

	//size_t outputStrides[dimension];
	//size_t inputStrides[dimension];
	//size_t filterStrides[dimension];

	size_t * outputStrides = new size_t[dimension];
	size_t * inputStrides = new size_t[dimension];
	size_t * filterStrides = new size_t[dimension];
	size_t * biasStrides = new size_t[dimension];

	outputStrides[0] = 1;
	inputStrides[0] = 1;
	filterStrides[0] = 1;
	//biasStrides[0] = 1;

	//size_t convolutionStride[dimension-2];
	size_t *convolutionStride = new size_t[dimension-2];
	int * inputOffset = new int[dimension-2];


	//std::cout << input_offset << std::endl;

	for( int i= 0; i < dimension-1; i++){
		outputSize[i] = out_size[dimension-2 -i];
		inputSize[i] = inp_size[dimension-2 -i];
		if( i < dimension-2 )
			filterSize[i] =  kernel_size;
		if( i == dimension-2)
			filterSize[i] = inp_size[0];
		if( i > 0){
			outputStrides[i] = outputStrides[i-1] * outputSize[i-1];
			inputStrides[i] = inputStrides[i-1] * inputSize[i-1];
			filterStrides[i] = filterStrides[i-1] * filterSize[i-1];
		}

		if( i < dimension-2){
			convolutionStride[i] = 1;
			int input_offset = - ( (outputSize[0]/2)) - filterSize[0]/2 + inputSize[0]/2;
			inputOffset[i] = input_offset;
		}
	}

	inputSize[dimension-1] = batch_number;
	outputSize[dimension-1] = batch_number;
	filterSize[dimension-1] = out_size[0];
	inputStrides[dimension-1] = inputStrides[dimension-2]*inputSize[dimension-2];
	outputStrides[dimension-1] = outputStrides[dimension-2]*outputSize[dimension-2];
	filterStrides[dimension-1] = filterStrides[dimension-2]*filterSize[dimension-2];

    biasSize[0] = outputSize[2] ;
    biasStrides[0] = outputStrides[2];
    biasStrides[0] = 1;
	/*
    size_t outputSize[4] = { 4, 4, 5, batch_number };
    size_t outputStrides[4] = { 1, 4, 4*4, 5 * 4*4 };

    size_t inputSize[4] = { 4, 4, 3, batch_number };
    size_t inputStrides[4] = { 1, 4, 4*4, 49*49*3 };

    size_t filterSize[4] = { 5, 5, 3, 5 };
    size_t filterStrides[4] = { 1, 5, 5*5, 3*5*5 };

    size_t convolutionStride[ 2] = { 1, 1 };
    int inputOffset[ 2 ] = { 0, 0 };
    */
/*
	for( int i=0; i < dimension; i++){
		std::cout << inputStrides[i] << " " ;
	}
	std::cout << std::endl;
	for( int i=0; i < dimension; i++){
		std::cout << outputStrides[i] << " " ;
	}
	std::cout << std::endl;
	for( int i=0; i < dimension; i++){
		std::cout << filterStrides[i] << " " ;
	}
	std::cout << std::endl;

	for( int i=0; i < dimension; i++){
		std::cout << inputSize[i] << " " ;
	}
	std::cout << std::endl;
	for( int i=0; i < dimension; i++){
		std::cout << outputSize[i] << " " ;
	}
	std::cout << std::endl;
	for( int i=0; i < dimension; i++){
		std::cout << filterSize[i] << " " ;
	}
	std::cout << std::endl;
*/
	//std::cout << convolutionStride[0] << " " << convolutionStride[1] << std::endl;
	//std::cout << inputOffset[0] << " " << inputOffset[0] << std::endl;





	//size_t outputStrides[dimension] = { 1, 45, 45 * 45, 45 * 45 * 16 };
    dnnLayout_t lt_conv1_input = NULL,
                lt_conv1_filt = NULL,
                lt_conv1_bias = NULL,
                lt_conv1_output = NULL,
    			lt_conv1_diff_src = NULL,
    			lt_conv1_diff_dst = NULL,
    			lt_conv1_diff_kernel = NULL,
    			lt_conv1_diff_bias = NULL;

	dnnLayout_t lt_user_input = NULL,
    			lt_user_kernels = NULL,
    			lt_user_bias = NULL,
    			lt_user_output = NULL,
    			lt_user_diff_src = NULL,
    			lt_user_diff_dst = NULL,
    			lt_user_diff_kernel = NULL,
    			lt_user_diff_bias = NULL;




	if( dnnPrimitiveAttributesCreate_F32(&attributes)!= E_SUCCESS){
		std::cout << "error" << std::endl;
	}
	dnnError_t err;
	if( use_bias ){
		err= dnnConvolutionCreateForwardBias_F32(&conv_prim, attributes,
	                    dnnAlgorithmConvolutionDirect, dimension, inputSize,
	                    outputSize, filterSize, convolutionStride, inputOffset,
	                    border_method);
	}else{
		err = dnnConvolutionCreateForward_F32(&conv_prim, attributes,
						dnnAlgorithmConvolutionDirect, dimension, inputSize,
						outputSize, filterSize, convolutionStride, inputOffset,
						border_method);
	}

	if( err != E_SUCCESS){
		switch (err){
		case E_INCORRECT_INPUT_PARAMETER:
				std::cout << "incorrect input parameter while creating the convolution" << std::endl;break;
		default:
			std::cout << "error while creating convolution" << std::endl;
		}

	}
	/* Create backwards convolution with respect to data*/
	dnnError_t err_back_data = dnnConvolutionCreateBackwardData_F32 (&conv_backwards_data, attributes,
	                dnnAlgorithmConvolutionDirect, dimension, inputSize,
	                outputSize, filterSize, convolutionStride, inputOffset,
	                dnnBorderZeros);
	if( err_back_data != E_SUCCESS){
		switch (err_back_data){
		case E_INCORRECT_INPUT_PARAMETER:
			std::cout << "incorrect input parameter while creating backwards convolution with respect to data" << std::endl;break;
		default: std::cout << " error while creating backwards convolution" << std::endl;
		}
	}
	/* Create gradient with respect to kernel*/
	dnnError_t err_back_kernel = dnnConvolutionCreateBackwardFilter_F32(&conv_backwards_kernel, attributes,
	                dnnAlgorithmConvolutionDirect, dimension, inputSize,
	                outputSize, filterSize, convolutionStride, inputOffset,
	                dnnBorderZeros);
	if( err_back_kernel != E_SUCCESS){
		switch (err_back_kernel){
		case E_INCORRECT_INPUT_PARAMETER:
			std::cout << "incorrect input parameter while creating backwards convolution with respect to kernel" << std::endl;break;
		default: std::cout << " error while creating backwards convolution" << std::endl;
		}
	}

	if( use_bias){
		dnnError_t err_back_bias = dnnConvolutionCreateBackwardBias_F32 (&conv_backwards_bias, attributes,
					   dnnAlgorithmConvolutionDirect, dimension, outputSize);

		if( err_back_bias != E_SUCCESS){
			switch (err_back_bias){
			case E_INCORRECT_INPUT_PARAMETER:
				std::cout << "incorrect input parameter while creating backwards convolution with respect to bias" << std::endl;break;
			default: std::cout << " error while creating backwards convolution" << std::endl;
			}
		}
	}
    dnnLayoutCreateFromPrimitive_F32(&lt_conv1_input, conv_prim, dnnResourceSrc);
    dnnLayoutCreateFromPrimitive_F32(&lt_conv1_filt, conv_prim, dnnResourceFilter);
    if( use_bias){
    	dnnLayoutCreateFromPrimitive_F32(&lt_conv1_bias, conv_prim, dnnResourceBias);
    }
    dnnLayoutCreateFromPrimitive_F32(&lt_conv1_output,conv_prim, dnnResourceDst);
    dnnLayoutCreateFromPrimitive_F32(&lt_conv1_diff_src, conv_backwards_data, dnnResourceDiffSrc);
    dnnLayoutCreateFromPrimitive_F32(&lt_conv1_diff_dst, conv_backwards_data, dnnResourceDiffDst);
    dnnLayoutCreateFromPrimitive_F32(&lt_conv1_diff_kernel, conv_backwards_kernel, dnnResourceDiffFilter);
    if( use_bias) dnnLayoutCreateFromPrimitive_F32(&lt_conv1_diff_bias, conv_backwards_bias, dnnResourceDiffBias);

    dnnLayoutCreate_F32(&lt_user_input, dimension,inputSize, inputStrides);
    dnnLayoutCreate_F32(&lt_user_kernels, dimension,filterSize,filterStrides);
    if( use_bias) dnnLayoutCreate_F32(&lt_user_bias, 1, biasSize, biasStrides );
    dnnLayoutCreate_F32(&lt_user_output, dimension,outputSize,outputStrides);

    dnnLayoutCreate_F32(&lt_user_diff_src, dimension, inputSize, inputStrides);
    dnnLayoutCreate_F32(&lt_user_diff_dst, dimension, outputSize, outputStrides);
    dnnLayoutCreate_F32(&lt_user_diff_kernel, dimension, filterSize,filterStrides);
    if( use_bias) dnnLayoutCreate_F32(&lt_user_diff_bias, 1, biasSize, biasStrides);

    dnnConversionCreate_F32(&conversion_user_to_input, lt_user_input, lt_conv1_input);
    dnnConversionCreate_F32(&conversion_output_to_user, lt_conv1_output,  lt_user_output);
    dnnConversionCreate_F32(&conversion_user_to_kernels, lt_user_kernels, lt_conv1_filt );
    if( use_bias) dnnConversionCreate_F32(&conversion_user_to_bias, lt_user_bias, lt_conv1_bias);

    dnnConversionCreate_F32(&conversion_errorsrc_to_user, lt_conv1_diff_src, lt_user_diff_src);
    dnnConversionCreate_F32(&conversion_user_to_errordst, lt_conv1_diff_dst, lt_user_diff_dst);
    dnnConversionCreate_F32(&conversion_errorkernels_to_user, lt_conv1_diff_kernel, lt_user_diff_kernel);
    if( use_bias) dnnConversionCreate_F32(&conversion_errorbias_to_user, lt_conv1_diff_bias, lt_user_diff_bias);


    dnnAllocateBuffer_F32((void**)&resConv1[dnnResourceDst], lt_conv1_output);
    dnnAllocateBuffer_F32((void**)&resConv1[dnnResourceSrc], lt_conv1_input);
    dnnAllocateBuffer_F32((void**)&resConv1[dnnResourceFilter], lt_conv1_filt);
    if( use_bias) dnnAllocateBuffer_F32((void**)&resConv1[dnnResourceBias], lt_conv1_bias);

    dnnAllocateBuffer_F32((void**)&resConv1[dnnResourceDiffDst], lt_conv1_diff_dst);
    dnnAllocateBuffer_F32((void**)&resConv1[dnnResourceDiffSrc], lt_conv1_diff_src);
    dnnAllocateBuffer_F32((void**)&resConv1[dnnResourceDiffFilter], lt_conv1_diff_kernel);
    if( use_bias) dnnAllocateBuffer_F32((void**)&resConv1[dnnResourceDiffBias], lt_conv1_diff_bias);


    //std::cout << dnnLayoutGetMemorySize_F32(lt_conv1_bias)/4 << " " << bias.size() << std::endl;

    //float* resConv1[dnnResourceNumber] = {0};
    /*std::cout << "batch size: " << batch_number << std::endl;
    std::cout << dnnLayoutGetMemorySize_F32(lt_conv1_input)/4<< " " << input.get_data().size() << std::endl;
    std::cout << dnnLayoutGetMemorySize_F32(lt_conv1_filt)/4<< " " << kernels.size() << std::endl;
    if( use_bias){
    	std::cout << dnnLayoutGetMemorySize_F32(lt_conv1_bias)/4<< " " << bias.size() << " " << dnnLayoutGetMemorySize_F32(lt_user_bias) << std::endl;
    }
    std::cout << dnnLayoutGetMemorySize_F32(lt_conv1_output)/4<<" " << output.get_data().size()<< std::endl;
    std::cout << dnnLayoutGetMemorySize_F32(lt_conv1_diff_dst)/4 << " " << error_inp.get_data().size() << std::endl;
    std::cout << dnnLayoutGetMemorySize_F32(lt_conv1_diff_src)/4 << " " << error_out.get_data().size() << std::endl;
    std::cout << dnnLayoutGetMemorySize_F32(lt_conv1_diff_kernel)/4 + (use_bias) * dnnLayoutGetMemorySize_F32(lt_conv1_diff_bias)/4 << " " << gradient.get_data().size() << std::endl;
*/
    //dnnAllocateBuffer_F32((void**)&resConv1[dnnResourceDst], lt_conv1_output);
    //dnnAllocateBuffer_F32((void**)&resConv1[dnnResourceSrc], lt_conv1_input);

    /*resConv1[dnnResourceSrc] = &(input.get_data()[0]);
    resConv1[dnnResourceFilter] = &kernels[0];
    if( use_bias)  resConv1[dnnResourceBias] = &bias[0];
    resConv1[dnnResourceDst]= &output.get_data()[0];*/

    //resConv1[dnnResourceDiffSrc] = &error_inp.get_data()[0];


    //resConv1[dnnResourceDiffDst] =


    batch_size = batch_number;




}

void ConvolutionalLayer::set_var_parameters(std::vector<float> & pars){
	std::copy(pars.begin(), pars.begin()+kernels.size(), kernels.begin());
	if( use_bias){
		std::copy(pars.begin()+kernels.size(), pars.end(), bias.begin());
	}
}

std::vector<float> & ConvolutionalLayer::get_var_parameters(){
	std::copy(kernels.begin(), kernels.end(), params.begin());
	if( use_bias){
	std::copy(bias.begin(), bias.end(), params.begin()+kernels.size());
	}
	return params;
}

VectorNd<float> & ConvolutionalLayer::get_parameter_gradient(VectorNd<float> & backprop_vec){
	/*resConv1[dnnResourceDiffFilter] = &gradient.get_data()[0];
	if(use_bias){
		resConv1[dnnResourceDiffBias] = &gradient.get_data()[kernels.size()];
	}

	resConv1[dnnResourceSrc] = &input.get_data()[0];
	resConv1[dnnResourceDiffDst] = &backprop_vec.get_data()[0];*/

	dnnConversionExecute_F32(conversion_user_to_input, &input.get_data()[0], resConv1[dnnResourceSrc] );
	dnnConversionExecute_F32(conversion_user_to_errordst, &backprop_vec.get_data()[0], resConv1[dnnResourceDiffDst]);
	dnnError_t err = dnnExecute_F32(conv_backwards_kernel, (void**) resConv1);
	if( err != E_SUCCESS){
		std::cout << "Error while executing convolution backwards with respect to kernels" << std::endl;
	}
	dnnConversionExecute_F32(conversion_errorkernels_to_user, resConv1[dnnResourceDiffFilter], &gradient.get_data()[0]);

	//resConv1[dnnResourceFilter] = &kernels[0];
	dnnConversionExecute_F32(conversion_user_to_kernels, &kernels[0], resConv1[dnnResourceFilter]);
	if( use_bias){
		dnnError_t err = dnnExecute_F32(conv_backwards_bias, (void**) resConv1);
		if( err != E_SUCCESS){
			std::cout << "Error while executing convolution backwards with respect to bias" << std::endl;
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
		dnnConversionExecute_F32(conversion_errorbias_to_user, resConv1[dnnResourceDiffBias],&gradient.get_data()[kernels.size()]);
	}


	return gradient;
}

void ConvolutionalLayer::process(){
	/*if( c_method == cutoff){
		for( int i_size = 0; i_size < inp_size[0]; i_size++){
			std::vector<int> pos(0,0);
			recursive_convolute(pos);
		}
	}*/
	if( input.getDimensionSize(0) != batch_size){
		setup_convolution(input.getDimensionSize(0));
	}
   // resConv1[dnnResourceFilter] = &kernels[0];
   // resConv1[dnnResourceBias] = &bias[0];

	/*resConv1[dnnResourceSrc] = &(input.get_data()[0]);
	resConv1[dnnResourceFilter] = &kernels[0];
	if( use_bias){
		resConv1[dnnResourceBias] = &bias[0];
	}
	resConv1[dnnResourceDst]= &output.get_data()[0];*/
	//std::cout << std::endl;
    dnnError_t converr1 = dnnConversionExecute_F32(conversion_user_to_input,&input.get_data()[0], resConv1[dnnResourceSrc] );
    dnnError_t converr2 = dnnConversionExecute_F32(conversion_user_to_kernels, &kernels[0], resConv1[dnnResourceFilter]);
    if( use_bias) dnnConversionExecute_F32(conversion_user_to_bias, &bias[0], resConv1[dnnResourceBias]);
	dnnError_t err = dnnExecute_F32(conv_prim, (void**) resConv1);
	if( err != E_SUCCESS){
		std::cout << "Error while forward propagation in convolutional layer" << std::endl;
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
	dnnError_t converr3 = dnnConversionExecute_F32(conversion_output_to_user, resConv1[dnnResourceDst], &output.get_data()[0]);
	if( converr1 != E_SUCCESS){
		std::cout << "error1" << std::endl;
		if( converr1  == E_MEMORY_ERROR){
			std::cout << "Memory Error" << std::endl;
		}
		if( converr1  == E_UNIMPLEMENTED){
			std::cout << "Unimplemented" << std::endl;
		}
		if( converr1  == E_UNSUPPORTED_DIMENSION){
			std::cout << "Unsupported dimension" << std::endl;
		}
		if( converr1  == E_INCORRECT_INPUT_PARAMETER){
			std::cout << "Incorrect input parameter" << std::endl;
		}
	}
	if( converr2 != E_SUCCESS){
		std::cout << "error2" << std::endl;
	}
	if( converr3 != E_SUCCESS){
		std::cout << "error3" << std::endl;
	}
}


void ConvolutionalLayer::backpropagate(){
	//std::cout << "error_out size0: " << error_out.getDimensionSize(0) << " " << error_out.get_data().size()<<  std::endl;
	if( error_inp.getDimensionSize(0) != batch_size){

		setup_convolution(error_inp.getDimensionSize(0));
	}
	dnnConversionExecute_F32(conversion_user_to_errordst, &error_inp.get_data()[0], resConv1[dnnResourceDiffDst]);
    dnnConversionExecute_F32(conversion_user_to_kernels, &kernels[0], resConv1[dnnResourceFilter]);
    if( use_bias) dnnConversionExecute_F32(conversion_user_to_bias, &bias[0], resConv1[dnnResourceBias]);

	//resConv1[dnnResourceDiffDst] = &error_inp.get_data()[0];
	//resConv1[dnnResourceDiffSrc]= &error_out.get_data()[0];

	dnnError_t err = dnnExecute_F32(conv_backwards_data, (void**) resConv1);
	if( err != E_SUCCESS){
		std::cout << "Error while executing convolution backwards with respect to data" << std::endl;
	}
	dnnConversionExecute_F32(conversion_errorsrc_to_user, resConv1[dnnResourceDiffSrc], &error_out.get_data()[0]);
}

} /* namespace MRINN */
//BOOST_CLASS_EXPORT(MRINN::ConvolutionalLayer)
