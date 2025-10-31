//============================================================================
// Name        : MRI_NeuralNetworkSuite.cpp
// Author      : Lennart Schlieder
// Version     :
// Copyright   : 
// Description : Mainfile MRI_NNS
//============================================================================

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/export.hpp>
#include <iostream>
#include "Network.h"
#include "Layers/Layer.h"


#include "Layers/DenseLayer.h"
BOOST_CLASS_EXPORT(MRINN::DenseLayer)
#include "Layers/TanhLayer.h"
BOOST_CLASS_EXPORT(MRINN::TanhLayer)
#include "Layers/ReluLayer.h"
BOOST_CLASS_EXPORT(MRINN::ReluLayer)
#include "Layers/Convolution2DLayer.h"
BOOST_CLASS_EXPORT(MRINN::Convolution2DLayer)
#include "Layers/ConvolutionalLayer.h"
BOOST_CLASS_EXPORT(MRINN::ConvolutionalLayer)
#include "Layers/SoftmaxLayer.h"
BOOST_CLASS_EXPORT(MRINN::SoftmaxLayer)
#include "Layers/MaxPoolingLayer.h"
BOOST_CLASS_EXPORT(MRINN::MaxPoolingLayer)
#include "Layers/DropoutLayer.h"
BOOST_CLASS_EXPORT(MRINN::DropoutLayer)
#include "Layers/BatchNormalizationLayer.h"
BOOST_CLASS_EXPORT(MRINN::BatchNormalizationLayer)
#include "Layers/convolution2dcudnn.h"
BOOST_CLASS_EXPORT(MRINN::Convolution2DCudnn)
//#include "ScewedDistanceLayer.h"
//#include "ExpLayer.h"


#include <mkl.h>
//#include <cblas.h>
#include <stdlib.h>
#include <cmath>
#include <fstream>
#include "VectorNd.h"
//#include <time.h>
#include <ctime>
#include "Datasets.h"
#include "Optimizers/Optimizer.h"
#include "Optimizers/SGD.h"
#include "Optimizers/AdaGrad.h"
#include "Optimizers/AdaDelta.h"
#include "Optimizers/Adam.h"
#include "Optimizers/AdaMax.h"
#include "omp.h"

#include "Losses/MSE.h"
#include "Losses/CrossEntropy.h"
#include "Losses/BinaryCrossentropy.h"

#include "CImg.h"
#include <cudnn.h>
#include <algorithm>


#include "Helper_Functions.cpp"
using namespace std;
using namespace MRINN;
using namespace cimg_library;

std::vector<int> v(int dims,  int* size){
	std::vector<int> vec(dims);
	for(int i = 0; i < dims;i++){
		vec[i] = size[i];
	}
	return vec;
}

void loadFile(std::vector<unsigned char>& buffer, const std::string& filename) //designed for loading files from hard disk in an std::vector
{
  std::ifstream file(filename.c_str(), std::ios::in|std::ios::binary|std::ios::ate);

  //get filesize
  std::streamsize size = 0;
  if(file.seekg(0, std::ios::end).good()) size = file.tellg();
  if(file.seekg(0, std::ios::beg).good()) size -= file.tellg();

  //read contents of the file into the vector
  if(size > 0)
  {
    buffer.resize((size_t)size);
    file.read((char*)(&buffer[0]), size);
  }
  else buffer.clear();
}


int main() {
	//std::vector<int> v {10,20};
	srand (time(NULL));
	//srand (time(1337));
	//srand();
	omp_set_num_threads(4);
	Network * net = new Network();
	//Network * net_loaded = new Network();
	//std::ifstream ifsload1("/home/lennart/Dokumente/Logfiles_denoising/GPU_17_layers_20_epochs_100_batchsize.dat");
    std::ifstream ifsload1("/home/lennart/projects/MRINN/Debug/Logfiles_denoising/saved_net_6Layers_20_epochs_100_batchsize_2.dat");
	boost::archive::binary_iarchive ia1(ifsload1);
	ia1 >> net;
	std::cout << "net loaded" << std::endl;
	ifsload1.close();
    for (int i_layers = 0; i_layers < net->Layers.size();i_layers++){
        std::cout << net->Layers[i_layers]->get_var_parameters().size() << std::endl;
    }
    std::cin.ignore();
    
    
    
/*
	std::vector<int> test_sinp(3,8); test_sinp[0] = 8;
	std::vector<int> test_sout(3,8); test_sout[0] = 8;
	BatchNormalizationLayer bnl = BatchNormalizationLayer(test_sinp, test_sout, 0.001);
	VectorNd<float>inptest(2,3,8*8*8);
	inptest.fill(1.0);
	for( int i =0; i < inptest.get_data().size(); i++){
		inptest.get_data()[i] = rand()/(float)RAND_MAX;
		inptest.get_data()[i] = i/8;
	}
	std::cout << "input: " << std::endl;
	inptest.print();
	VectorNd<float> outtest = bnl.calc_output(inptest);
	std::cout << "output: " << std::endl;
	outtest.print();

	VectorNd<float>errortest(2,3,8*8*8);
	errortest.fill(0.0);
	for( int i =0; i < errortest.get_data().size(); i++){
		errortest.get_data()[i] = i%8;
		errortest.get_data()[i] = i/32;
	}
	errortest.print();
	std::cout << std::endl;
	VectorNd<float> erroutest = bnl.calc_error(errortest);
	erroutest.print();


	VectorNd<float> grad_out = bnl.get_parameter_gradient(errortest);
	grad_out.print();
	std::cin.ignore();
*/
/*
	std::vector<int> test_sinp(3,4); test_sinp[0] = 2;
	std::vector<int> test_sout(3,2); test_sout[0] = 2;
	MaxPoolingLayer pooling_test = MaxPoolingLayer(test_sinp, test_sout, 2,2);

	VectorNd<float> test_input(2,2,32);
	test_input.fill(0.0);
	test_input(2,0,0) = 2.0; test_input(2,0,12) = 3.0; test_input(2,0,2) = 0.5; test_input(2,0,15) = 0.9; test_input(2,0,3) = 1.9;
	test_input(2,0,30) = 2.0; test_input(2,0,20) = 3.0; test_input(2,0,25) = 0.5; test_input(2,0,29) = 0.9; test_input(2,0,19) = 1.9;

	test_input(2,1,0) = 1.9; test_input(2,1,12) = 2.0; test_input(2,1,2) = 0.4; test_input(2,1,15) = 0.8; test_input(2,1,2) = 1.8;
	test_input(2,1,30) = 1.9; test_input(2,1,20) = 2.0; test_input(2,1,25) = 0.4; test_input(2,1,29) = 0.8; test_input(2,1,19) = 1.8;
	test_input.print();
	//std::cout << test_input.getDimensionSize(0) << " " << test_input.getDimensionSize(1) << std::endl;

	VectorNd<float> test_output  = pooling_test.calc_output(test_input);
	//std::cout << test_output.getDimensionSize(0) << " " << test_output.getDimensionSize(1) << std::endl;
	test_output.print();

	VectorNd<float> test_error_inp(2,2,8);
	test_error_inp.fill(1.0);
	//std::cout << " asdf2" << std::endl;
	VectorNd<float> test_error_out = pooling_test.calc_error(test_error_inp);
	for( int i =0; i < test_error_out.get_data().size();i++){
		if( i%4 == 0){
			std::cout << std::endl;
		}
		std::cout << test_error_out.get_data()[i] << " " ;

	}
	std::cin.ignore();*/
/*
	//std::cout << pooling_test.get_var_parameter_number()  << std::endl;
	/*VectorNd<float> testinp(2,3,4);
	testinp(2,0,0) = 0.5; testinp(2,0,1) = 0.5; testinp(2,0,2) = 0.5; testinp(2,0,3) = 0.5;
	testinp(2,1,0) = 0.2; testinp(2,1,1) = 0.2; testinp(2,1,2) = 0.2; testinp(2,1,3) = 0.2;
	testinp(2,2,0) = 0.2; testinp(2,2,1) = 0.2; testinp(2,2,2) = 0.2; testinp(2,2,3) = 0.2;
	DenseLayer test_layer = DenseLayer(std::vector<int>(1,4), std::vector<int>(1,2));
	VectorNd<float> out_test = test_layer.calc_output(testinp);
	std::cout << out_test(2,0,0) << " " << out_test(2,1,0)<< " " << out_test(2,2,0)<< std::endl;
	std::cout << out_test(2,0,1) << " " << out_test(2,1,1)<< " " << out_test(2,2,1)<< std::endl;
	std::cin.ignore();*/

/*
	// convolution test below
    cudnnHandle_t cudnn;
    std::cout << cudnnCreate(&cudnn) << std::endl;
    
    vector<int> test_inp(3,4); test_inp[0] = 2;
	vector<int> test_out(3,4); test_out[0] = 2;

    Convolution2DCudnn convo_test(test_inp, test_out,3,2, true);
    ConvolutionalLayer conv_comp(test_inp,test_out, 3, 2, dnnBorderZeros,true);

    
    convo_test.set_cudnn_handle(cudnn);;
    VectorNd<float>inpt(2,1,4*4*2);
	inpt.fill(1.0);
	for( int i =0; i < inpt.get_data().size(); i++){
		inpt.get_data()[i] = i%16;
	}
    std::cout << "input:";
	inpt.print();
    std::cin.ignore();
    VectorNd<float> out_test2 = convo_test.calc_output(inpt);
    VectorNd<float> out_test_comp = conv_comp.calc_output(inpt);
    std::cout << "output new:" << std::endl;
	out_test2.print();
    std::cout << std::endl << "output comparison:" << std::endl;
    out_test_comp.print();
    cin.ignore();
    
    VectorNd<float>errort(2,2,4*4*2);
	errort.fill(0.0);
	for( int i =0; i < errort.get_data().size(); i++){
		errort.get_data()[i] = 1.0;
	}
    

	//vector<float> params(2*3*3,1.0);
	//std::vector<float> newparams = *convo_test.get_var_parameters();
	//for( int i=0; i < newparams.size();i++){
	//	std::cout << newparams[i] << " ";
	//}
	//std::cout << std::endl;

	//convo_test.set_var_parameters(newparams);
	std::cout << "error inp";
	errort.print();
	VectorNd<float> erroroutt = convo_test.calc_error(errort);
	//VectorNd<float> errorout_test2 = convo_test.calc_error(errort);
    VectorNd<float> errorout_comp = conv_comp.calc_error(errort);
	std::cout << " errorout:";
	erroroutt.print();
	//std::cout << std::endl << " errorout_new " << std::endl;
	//errorout_test2.print();
    std::cout << std::endl << " errorout comp:" << std::endl;
    errorout_comp.print();
    cin.ignore();
    
    
	//VectorNd<float> gradient = convo_test.get_parameter_gradient(errort);
	//VectorNd<float> gradient_2 = convo_nd_test.get_parameter_gradient(errort);
    VectorNd<float> gradient = convo_test.get_parameter_gradient(errort);
    VectorNd<float> gradient_comp = conv_comp.get_parameter_gradient(errort);
    std::cout << std::endl << " gradient: " << std::endl;
    gradient.print();
    std::cout << std::endl << " gradient_comp:" << std::endl;
    gradient_comp.print();
    cin.ignore();*/
	//std::cout << " gradient:";
	//gradient.print();
    
/*
	vector<int> test_inp(3,4); test_inp[0] = 2;
	vector<int> test_out(3,4); test_out[0] = 2;
	Convolution2DLayer convo_test = Convolution2DLayer(test_inp, test_out,3,2, Convolution2DLayer::zero_padded);
	ConvolutionalLayer convo_nd_test = ConvolutionalLayer(test_inp, test_out, 3,2, dnnBorderZeros,true);
	//vector<float> inpt(16,1.0);
	VectorNd<float>inpt(2,3,4*4*2);
	inpt.fill(1.0);
	for( int i =0; i < inpt.get_data().size(); i++){
		inpt.get_data()[i] = i%16;
	}
	std::cout << "input:";
	inpt.print();
	//VectorNd<float> outt = convo_test.calc_output(inpt);
	VectorNd<float> out_test2 = convo_nd_test.calc_output(inpt);
    std::cout << "input again:" ;
    inpt.print();
	//std::cout << "output:";
	//outt.print();
	std::cout << "output new:" << std::endl;
	out_test2.print();
	std::cout << std::endl;
	std::cin.ignore();

	VectorNd<float>errort(2,3,4*4*2);
	errort.fill(0.0);
	for( int i =0; i < errort.get_data().size(); i++){
		errort.get_data()[i] = 1.0;
	}
    

	//vector<float> params(2*3*3,1.0);
	//std::vector<float> newparams = *convo_test.get_var_parameters();
	//for( int i=0; i < newparams.size();i++){
	//	std::cout << newparams[i] << " ";
	//}
	//std::cout << std::endl;

	//convo_test.set_var_parameters(newparams);
	std::cout << "error inp";
	errort.print();
	//VectorNd<float> erroroutt = convo_test.calc_error(errort);
	VectorNd<float> errorout_test2 = convo_nd_test.calc_error(errort);
	//std::cout << " errorout:";
	//erroroutt.print();
	std::cout << std::endl << " errorout_new " << std::endl;
	errorout_test2.print();
	VectorNd<float> gradient = convo_test.get_parameter_gradient(errort);
	VectorNd<float> gradient_2 = convo_nd_test.get_parameter_gradient(errort);

	std::cout << " gradient:";
	gradient.print();
	std::cout << " gradient_new" << std::endl;
	gradient_2.print();
	std::cout << " params new :" << std::endl;

	std::vector<float>  params = *convo_nd_test.get_var_parameters();
	for( int i =0; i < params.size();i++){
		std::cout << params[i]<< " ";
		params[i] = 0.5;
	}
	std::cout << std::endl;
	convo_nd_test.set_var_parameters(params);
	params = *convo_nd_test.get_var_parameters();
	for( int i =0; i < params.size();i++){
		//std::cout << params[i]<< " ";
		//params[i] = 0.5;
	}

	std::cout << std::endl;

	std::cin.ignore();
*/
	//convolution test end

	/*vector<int> test_size(3,4); test_size[0] = 1;
	vector<int> test_size_out(3,2); test_size_out[0] = 1;
	MaxPoolingLayer pool = MaxPoolingLayer(test_size, test_size_out,2);
	VectorNd<float> testinp(2,1,16); testinp.fill(0.5); testinp(2,0,2) = 1.0; testinp(2,0,8) = 3.0; testinp(2,0,14) = 4;
	//std::cout <<" test" << std::endl;
	testinp.print();
	VectorNd<float> testout = pool.calc_output(testinp);
	testout.print();

	/*vector<int> inp_relu(3,4); inp_relu[0] = 2;
	ReluLayer relu = ReluLayer(inp_relu,inp_relu);
	VectorNd<float> inp_vec_relu(2,2,32);
	for( int i =0; i < inp_vec_relu.get_data().size();i++){
		inp_vec_relu.get_data()[i] = (rand()/(float)RAND_MAX)-0.5;
	}
	inp_vec_relu.print();
	VectorNd<float> relu_out = relu.calc_output(inp_vec_relu);
	relu_out.print();
	VectorNd<float> error_in(2,2,32);
	error_in.fill(0.5);
	VectorNd<float> error_out = relu.calc_error(error_in);
	error_out.print();

	std::cin.ignore();*/
	/*
	DenseLayer test_dense = DenseLayer(std::vector<int>(1,2), std::vector<int>(1,2), true);
	VectorNd<float> test_input(2,2,2);
	test_input.fill(1.0);
	for( int i=0; i < test_input.get_data().size(); i++){
		test_input.get_data()[i] = i+1;
	}
	VectorNd<float> test_out = test_dense.calc_output(test_input);
	test_out.print();

	VectorNd<float> error_test_input(2,2,2);
	error_test_input.fill(1.0);
	for( int i=0; i < error_test_input.get_data().size(); i++){
		error_test_input.get_data()[i] = i+1;
	}
	VectorNd<float> error_test_out = test_dense.calc_error(error_test_input);
	VectorNd<float> gradient = test_dense.get_parameter_gradient(error_test_input);
	error_test_out.print();
	gradient.print();
	std::cin.ignore();*/







	cout << " initialize network " << endl;
	vector<int> inp_size1(3,50); inp_size1[0] = 2; // 748
	vector<int> out_size1(3,50); out_size1[0] = 64;
	vector<int> out_size2(3,14); out_size2[0] = 31;
	vector<int> out_size3(3,14); out_size3[0] = 32;
	vector<int> out_size4(3,7); out_size4[0] = 32;
	vector<int> out_size5(3,7); out_size5[0] = 2;
	vector<int> out_size6(3,3); out_size6[0] = 2;

	/*vector<int> mnist_size1(3,28); mnist_size1[0] = 1;
	vector<int> mnist_size2(3,28); mnist_size2[0] = 32;
	vector<int> mnist_size3(3,14); mnist_size3[0] = 32;
	vector<int> mnist_size4(3,14); mnist_size4[0] = 32;
	vector<int> mnist_size5(3,7); mnist_size5[0] = 32;
	vector<int> mnist_size6(3,7); mnist_size6[0] = 2;
	vector<int> mnist_size7(3,3); mnist_size7[0] = 2;*/
	//vector<int> out_size4(3,4); out_size4[0] = 64;
	/*vector<int> out_size1(3,28); out_size1[0] = 10;// 7840
	vector<int> out_size2(3,14); out_size2[0] = 10;// 1960
	vector<int> out_size3(3,14); out_size3[0] = 40; // 7480
	vector<int> out_size4(3,7); out_size4[0] = 40; // 1960*/




    //net->add_layer(new BatchNormalizationLayer( inp_size1, inp_size1, 0.00000001)); 
    //cudnnHandle_t cudnn_context;
    //cudnnCreate(&cudnn_context);
    //Convolution2DCudnn * convo_first = new Convolution2DCudnn(inp_size1, out_size1,3,64,true);
    //Convolution2DCudnn * convo_second = new Convolution2DCudnn(out_size1, out_size1,3,64,true);
    /*Convolution2DCudnn * convo_third = new Convolution2DCudnn(out_size1, out_size1,3,64,true);
    Convolution2DCudnn * convo_fourth = new Convolution2DCudnn(out_size1, out_size1,3,64,true);
    Convolution2DCudnn * convo_fifth = new Convolution2DCudnn(out_size1, out_size1,3,64,true);
    Convolution2DCudnn * convo_sixth = new Convolution2DCudnn(out_size1, out_size1,3,64,true);
    Convolution2DCudnn * convo_seven = new Convolution2DCudnn(out_size1, out_size1,3,64,true);
    Convolution2DCudnn * convo_eight = new Convolution2DCudnn(out_size1, out_size1,3,64,true);
    Convolution2DCudnn * convo_nine = new Convolution2DCudnn(out_size1, out_size1,3,64,true);
    Convolution2DCudnn * convo_ten = new Convolution2DCudnn(out_size1, out_size1,3,64,true);
    Convolution2DCudnn * convo_eleven = new Convolution2DCudnn(out_size1, out_size1,3,64,true);
    Convolution2DCudnn * convo_twelve = new Convolution2DCudnn(out_size1, out_size1,3,64,true);
    Convolution2DCudnn * convo_thirteen = new Convolution2DCudnn(out_size1, out_size1,3,64,true);
    Convolution2DCudnn * convo_fourteen = new Convolution2DCudnn(out_size1, out_size1,3,64,true);
    Convolution2DCudnn * convo_fifteen = new Convolution2DCudnn(out_size1, out_size1,3,64,true);
    Convolution2DCudnn * convo_sixteen = new Convolution2DCudnn(out_size1, out_size1,3,64,true);*/

    //Convolution2DCudnn * convo_first = new Convolution2DCudnn(inp_size1, out_size1,3,64,true);
    //Convolution2DCudnn * convo_first = new Convolution2DCudnn(inp_size1, out_size1,3,64,true);
    //Convolution2DCudnn * convo_first = new Convolution2DCudnn(inp_size1, out_size1,3,64,true);
    //Convolution2DCudnn * convo_first = new Convolution2DCudnn(inp_size1, out_size1,3,64,true);

    //Convolution2DCudnn * convo_output = new Convolution2DCudnn(out_size1, inp_size1,3,2,true);
    /*convo_first->set_cudnn_handle(cudnn_context);
    convo_second->set_cudnn_handle(cudnn_context);
    convo_third->set_cudnn_handle(cudnn_context);
    convo_fourth->set_cudnn_handle(cudnn_context);
    convo_fifth->set_cudnn_handle(cudnn_context);
    convo_sixth->set_cudnn_handle(cudnn_context);
    convo_output->set_cudnn_handle(cudnn_context);*/

    
    //net->add_layer(convo_first);
	//net->add_layer(new ReluLayer(out_size1,out_size1));
    
    //net->add_layer(convo_second);
	//net->add_layer(new ReluLayer(out_size1,out_size1));
    
   /* net->add_layer(convo_third);
	net->add_layer(new ReluLayer(out_size1,out_size1));
    
    net->add_layer(convo_fourth);
	net->add_layer(new ReluLayer(out_size1,out_size1));
    
    net->add_layer(convo_fifth);
	net->add_layer(new ReluLayer(out_size1,out_size1));
    
    net->add_layer(convo_sixth);
	net->add_layer(new ReluLayer(out_size1,out_size1));
    
    net->add_layer(convo_seven);
	net->add_layer(new ReluLayer(out_size1,out_size1));
    
    net->add_layer(convo_eight);
	net->add_layer(new ReluLayer(out_size1,out_size1));
    
    net->add_layer(convo_nine);
	net->add_layer(new ReluLayer(out_size1,out_size1));
    
    net->add_layer(convo_ten);
	net->add_layer(new ReluLayer(out_size1,out_size1));
    
    net->add_layer(convo_eleven);
	net->add_layer(new ReluLayer(out_size1,out_size1));
    
    net->add_layer(convo_twelve);
	net->add_layer(new ReluLayer(out_size1,out_size1));
    
    net->add_layer(convo_thirteen);
	net->add_layer(new ReluLayer(out_size1,out_size1));
    
    net->add_layer(convo_fourteen);
	net->add_layer(new ReluLayer(out_size1,out_size1));
    
    net->add_layer(convo_fifteen);
	net->add_layer(new ReluLayer(out_size1,out_size1));
    
    net->add_layer(convo_sixteen);
	net->add_layer(new ReluLayer(out_size1,out_size1));*/
    /*net->add_layer(convo_sixth);
	net->add_layer(new ReluLayer(out_size1,out_size1));*/
    
    //net->add_layer(convo_output);
    
    /*
	//net->add_layer(new BatchNormalizationLayer( out_size1, out_size1,0.001));
	net->add_layer(new ConvolutionalLayer(inp_size1,out_size1, 3, 64, dnnBorderZeros,true)); //1
	//net->add_layer(new BatchNormalizationLayer( out_size1, out_size1,0.00000001));
	net->add_layer(new ReluLayer(out_size1,out_size1));
	net->add_layer(new ConvolutionalLayer(out_size1,out_size1, 3, 64, dnnBorderZeros,true)); //2
	//net->add_layer(new BatchNormalizationLayer( out_size1, out_size1,0.00000001));
	net->add_layer(new ReluLayer(out_size1,out_size1));
	net->add_layer(new ConvolutionalLayer(out_size1,out_size1, 3, 64, dnnBorderZeros,true)); //3
	//net->add_layer(new BatchNormalizationLayer( out_size1, out_size1,0.00000001));
	net->add_layer(new ReluLayer(out_size1,out_size1));
	net->add_layer(new ConvolutionalLayer(out_size1,out_size1, 3, 64, dnnBorderZeros,true)); //4
	//net->add_layer(new BatchNormalizationLayer( out_size1, out_size1,0.00000001));
	net->add_layer(new ReluLayer(out_size1,out_size1));
    net->add_layer(new ConvolutionalLayer(out_size1,out_size1, 3, 64, dnnBorderZeros,true)); //5
	//net->add_layer(new BatchNormalizationLayer( out_size1, out_size1,0.00000001));
	net->add_layer(new ReluLayer(out_size1,out_size1));
	net->add_layer(new ConvolutionalLayer(out_size1,out_size1, 3, 64, dnnBorderZeros,true));//6
	//net->add_layer(new BatchNormalizationLayer( out_size1, out_size1,0.00000001));
	net->add_layer(new ReluLayer(out_size1,out_size1));
	net->add_layer(new ConvolutionalLayer(out_size1,out_size1, 3, 64, dnnBorderZeros,true));//7
	//net->add_layer(new BatchNormalizationLayer( out_size1, out_size1,0.00000001));
	net->add_layer(new ReluLayer(out_size1,out_size1));
	net->add_layer(new ConvolutionalLayer(out_size1,out_size1, 3, 64, dnnBorderZeros,true));//8
	//net->add_layer(new BatchNormalizationLayer( out_size1, out_size1,0.00000001));
	net->add_layer(new ReluLayer(out_size1,out_size1));
	net->add_layer(new ConvolutionalLayer(out_size1,out_size1, 3, 64, dnnBorderZeros,true));//9
	//net->add_layer(new BatchNormalizationLayer( out_size1, out_size1,0.00000001));
	net->add_layer(new ReluLayer(out_size1,out_size1));
	net->add_layer(new ConvolutionalLayer(out_size1,out_size1, 3, 64, dnnBorderZeros,true));//10
	//net->add_layer(new BatchNormalizationLayer( out_size1, out_size1,0.00000001));
	net->add_layer(new ReluLayer(out_size1,out_size1));
	net->add_layer(new ConvolutionalLayer(out_size1,out_size1, 3, 64, dnnBorderZeros,true));//11
	//net->add_layer(new BatchNormalizationLayer( out_size1, out_size1,0.00000001));
	net->add_layer(new ReluLayer(out_size1,out_size1));
	net->add_layer(new ConvolutionalLayer(out_size1,out_size1, 3, 64, dnnBorderZeros,true));//12
	//net->add_layer(new BatchNormalizationLayer( out_size1, out_size1,0.00000001));
	net->add_layer(new ReluLayer(out_size1,out_size1));
	net->add_layer(new ConvolutionalLayer(out_size1,out_size1, 3, 64, dnnBorderZeros,true)); //13
	//net->add_layer(new BatchNormalizationLayer( out_size1, out_size1,0.00000001));
	net->add_layer(new ReluLayer(out_size1,out_size1));
	net->add_layer(new ConvolutionalLayer(out_size1,out_size1, 3, 64, dnnBorderZeros,true));//14
	//net->add_layer(new BatchNormalizationLayer( out_size1, out_size1,0.00000001));
	net->add_layer(new ReluLayer(out_size1,out_size1));
	net->add_layer(new ConvolutionalLayer(out_size1,out_size1, 3, 64, dnnBorderZeros,true));//15
	//net->add_layer(new BatchNormalizationLayer( out_size1, out_size1,0.00000001));
	net->add_layer(new ReluLayer(out_size1,out_size1));
	net->add_layer(new ConvolutionalLayer(out_size1,out_size1, 3, 64, dnnBorderZeros,true));//16
	//net->add_layer(new BatchNormalizationLayer( out_size1, out_size1,0.00000001));
	net->add_layer(new ReluLayer(out_size1,out_size1));
	//net->add_layer(new ConvolutionalLayer(out_size1,out_size1, 3, 64, dnnBorderZeros,true));
	//net->add_layer(new BatchNormalizationLayer( out_size1, out_size1,0.00000001));
	//net->add_layer(new ReluLayer(out_size1,out_size1));
	//net->add_layer(new ConvolutionalLayer(out_size1,out_size1, 3, 64, dnnBorderZeros,true));
	//net->add_layer(new BatchNormalizationLayer( out_size1, out_size1,0.00000001));
	//net->add_layer(new ReluLayer(out_size1,out_size1));
	
	//net->add_layer(new ConvolutionalLayer(out_size1,out_size1, 3, 64, dnnBorderZeros,true));
	//net->add_layer(new BatchNormalizationLayer( out_size1, out_size1,0.00000001));
	//net->add_layer(new ReluLayer(out_size1,out_size1));

	//net->add_layer(new ConvolutionalLayer(out_size1,out_size1, 3, 32, dnnBorderZeros,false));
	//net->add_layer(new BatchNormalizationLayer( out_size1, out_size1,0.00000001));
	//net->add_layer(new ReluLayer(out_size1,out_size1));

	//net->add_layer(new BatchNormalizationLayer( out_size1, out_size1,0.001));
*/


	//--->
	//net->add_layer(new ConvolutionalLayer(out_size1,inp_size1, 3, 2, dnnBorderZeros,true));
	//--->
	//std::cout << "test2" << std::endl;
	//net->add_layer(new ReluLayer(inp_size1,inp_size1));


/*
	net->add_layer(new Convolution2DLayer(inp_size1, out_size1,3,32,Convolution2DLayer::zero_padded));
	//net->add_layer(new BatchNormalizationLayer( out_size1, out_size1,0.001));
	net->add_layer(new ReluLayer(out_size1,out_size1));

	net->add_layer(new Convolution2DLayer(out_size1, out_size1,3,32,Convolution2DLayer::zero_padded));
	net->add_layer(new ReluLayer(out_size1,out_size1));

	net->add_layer(new Convolution2DLayer(out_size1, out_size1,3,32,Convolution2DLayer::zero_padded));
	net->add_layer(new ReluLayer(out_size1,out_size1));

	net->add_layer(new Convolution2DLayer(out_size1, out_size1,3,32,Convolution2DLayer::zero_padded));
	net->add_layer(new ReluLayer(out_size1,out_size1));

	net->add_layer(new Convolution2DLayer(out_size1, inp_size1,3,1,Convolution2DLayer::zero_padded));
	net->add_layer(new ReluLayer(inp_size1,inp_size1));



*/
/*
	net->add_layer( new ConvolutionalLayer(mnist_size1, mnist_size2, 3,32, dnnBorderZeros,false));
	net->add_layer(new ReluLayer(mnist_size2, mnist_size2));
	net->add_layer( new MaxPoolingLayer( mnist_size2, mnist_size3, 2,2));

	net->add_layer( new ConvolutionalLayer( mnist_size3, mnist_size4, 3,32, dnnBorderZeros,false));
	net->add_layer( new ReluLayer( mnist_size4, mnist_size4));
	net->add_layer( new MaxPoolingLayer( mnist_size4, mnist_size5, 2,2));
*/
	/*net->add_layer( new ConvolutionalLayer( out_size4, out_size3, 3,32, dnnBorderZeros,false));
	net->add_layer( new ReluLayer( out_size3, out_size3));
	net->add_layer( new MaxPoolingLayer( out_size3, out_size4, 2,2));*/

	//net->add_layer(new DenseLayer(mnist_size5, std::vector<int>(1,100), false));
	//net->add_layer(new DenseLayer(std::v6ctor<int>(1,300), std::vector<int>(1,300), true));
	//net->add_layer(new DenseLayer(std::vector<int>(1,300), std::vector<int>(1,300), true));
	//net->add_layer(new DenseLayer(std::vector<int>(1,300), std::vector<int>(1,300), true));

	//net->add_layer(new DenseLayer(std::vector<int>(1,100), std::vector<int>(1,100), true));
	//net->add_layer(new DenseLayer(std::vector<int>(1,1000), std::vector<int>(1,100), false));
	//net->add_layer(new DenseLayer(std::vector<int>(1,100), std::vector<int>(1,100), false));
	//net->add_layer(new DenseLayer(std::vector<int>(1,100), std::vector<int>(1,100), false));
	//net->add_layer(new DenseLayer(std::vector<int>(1,100), std::vector<int>(1,100), false));
	//net->add_layer(new DenseLayer(std::vector<int>(1,100), std::vector<int>(1,100), false));
	//net->add_layer(new DenseLayer(std::vector<int>(1,100), std::vector<int>(1,100), false));
   // net->add_layer(new DenseLayer(std::vector<int>(1,100), std::vector<int>(1,100), false));
	//net->add_layer(new DenseLayer(std::vector<int>(1,100), std::vector<int>(1,100), false));
	//net->add_layer(new DenseLayer(std::vector<int>(1,100), std::vector<int>(1,10), false));
	//net->add_layer(new SoftmaxLayer(std::vector<int>(1,10), std::vector<int>(1,10)));






	cout << "finished initializing network" << endl;
	cout << "start loading data..." << std::endl;
	//vector<Data> noisy_mnist;
	//vector<Data> noisy_mnist_test;
	//noisy_mnist = Datasets::MNIST::load_noisy_mnist_train(noisy_mnist,60000,0.1);
	//noisy_mnist_test= Datasets::MNIST::load_noisy_mnist_test(noisy_mnist_test,10000,0.1);

	//vector<Data> noisy_cifar10;// = Datasets::CIFAR10::load_noisy_CIFAR10_train(noisy_cifar10, 40000, 0.3);
	//vector<Data> noisy_cifar10_test;// = Datasets::CIFAR10::load_CIFAR10_test(noisy_cifar10_test,10000,0.3);
	//noisy_cifar10 = Datasets::CIFAR10::load_noisy_CIFAR10_train(noisy_cifar10, 40000, 0.3);
	//noisy_cifar10_test = Datasets::CIFAR10::load_noisy_CIFAR10_test(noisy_cifar10_test,10000,0.3);

	vector<Data*> *noisy_knees = new std::vector<Data*>(0);
	vector<Data*> *noisy_knees_test = new std::vector<Data*>(0);
	Datasets::FullySampledKnees::load_noisy_knees_train_single_sigma(noisy_knees,50000,0.13);
    std::cout << "loading test data" << std::endl;
    Datasets::FullySampledKnees::load_noisy_knees_test_single_sigma(noisy_knees_test,10000, 0.13);
   // Datasets::FullySampledKnees::
	std::cout << noisy_knees->size() << " " << (*noisy_knees)[0]->inp.size() << " " << (*noisy_knees)[0]->out.size() << std::endl;
	//noisy_knees_test = Datasets::FullySampledKnees::load_noisy_knees_train(noisy_knees_test,100,0.1);
	std::cout << "finished loading data!" << std::endl;
    
	//vector<Data> mnist;
	//vector<Data> mnist_test;
	//mnist = Datasets::MNIST::load_mnist_train(mnist,60000);
	//mnist_test = Datasets::MNIST::load_mnist_test(mnist_test,10000);
	//std::cout << "finished loaded data!" << std::endl;

/*
	//std::cout << noisy_knees.size() << " " << noisy_knees[0].inp.size() << " " << noisy_knees[0].out.size() << std::endl;
	CImg<float> noisy_img_test(&noisy_knees[101].inp[0],50,50,1,1);
	//std::cout << "test3" << std::endl;
	CImg<float> noise_test(&noisy_knees[101].out[0],50,50,1,1);
	//std::cout << "test4" << std::endl;
	CImg<float> denoised_img_test = noisy_img_test;// - (noise_test-0.5);
	//std::cout << "test1" << std::endl;
	for(int i =0; i < noisy_img_test.size();i++){
		denoised_img_test.data()[i] = max(min(noisy_img_test.data()[i] -( noise_test.data()[i]-0.5),1.0),0.0);
		//std::cout << noisy

		//std::cout << noisy_img_test.data()[i] << " " << noise_test.data()[i] << " " << noisy_img_test.data()[i] +abs( noise_test.data()[i]-0.5) << std::endl;
	}

	//std::cout << "test2" << std::endl;
	VectorNd<float> inp(2,1,2500);
	std::copy( noisy_knees[101].inp.begin(), noisy_knees[101].inp.end(), inp.get_data().begin());
	VectorNd<float> denoised_image = net->calc_output(inp);

	CImg<float> denoised_img(&denoised_image.get_data()[0],50,50,1,1);

	noisy_img_test*=255;
	noise_test*=255;
	denoised_img_test*=255;
	denoised_img *= 255;
	//denoised_img*=200;
	noisy_img_test.save("noisy_img_test.bmp");
	noise_test.save("noise_test.bmp");
	denoised_img_test.save("denoised_test.bmp");
	denoised_img.save("denoised_with_net.bmp");
*/




	//net->set_data(noisy_cifar10);
	//net->set_data(mnist);
	std::cout << "setting data" << std::endl;
	net->set_data(*noisy_knees);
    net->set_validation_data(*noisy_knees_test);
	std::cout << "finished setting data!" << std::endl;

	training_opts opts;
	opts.batch_size = 100;
	opts.epochs = 5;
    opts.log_loss = true;
    opts.log_filename = "Logfiles_denoising/Logfile_cpu_17layers_20_epochs_100_batchsize.txt";
    opts.validation_log_filename = "Logfiles_denoising/Validation_Logfile_cpu_17layers_20_epochs_100_batchsize.txt";
    opts.use_validation_dataset= true;
	Optimizers::SGD sgd = Optimizers::SGD(0.004,0.9,false);
	//Optimizers::AdaDelta optim = Optimizers::AdaDelta(0.01, 0.95f);
	Optimizers::AdaGrad adagrad = Optimizers::AdaGrad(0.01f);
	Optimizers::Adam *adam = new Optimizers::Adam(0.001,0.9,0.999);
	Optimizers::AdaMax adamax = Optimizers::AdaMax(0.002,0.9,0.999);

	opts.optimizer = adam;
    //opts.optimizer = &sgd;
    //opts.optimizer = &adagrad

	Losses::MSE *mse = new Losses::MSE();
	Losses::CrossEntropy crossent = Losses::CrossEntropy();
	opts.loss = mse;



	time_t tstart,tend;
	std::cout << "starting_training"<<std::endl;
	time(&tstart);

	//net->train_network(opts);
	time(&tend);
	float diff = difftime(tend,tstart);
	std::cout << "learning finished in: " << diff << " s"<< std::endl;
    
    
	//std::ofstream ofs("Logfiles_denoising/CPU_17_layers_20_epochs_100_batchsize.dat");
	//boost::archive::binary_oarchive oa(ofs);
	//oa << net;
	//ofs.close();
	//std::cout << "saved network!" << std::endl;
    
	for( int i =0; i < 100;i++){

        //std::cout << "test" << std::endl;
		VectorNd<float> inp(2,1,5000);
        VectorNd<float> actual_noise_compl(2,1,5000);
        //std::cout << "test" << std::endl;
		std::copy((*noisy_knees_test)[i]->inp.begin(), (*noisy_knees_test)[i]->inp.end(), &inp.get_data()[0]);
        std::copy((*noisy_knees_test)[i]->out.begin(), (*noisy_knees_test)[i]->out.end(), actual_noise_compl.get_data().begin());
        //std::cout << "test" << std::endl;
        //std::cout << "knees_inp: " << noisy_knees[i].inp[0] << " " << noisy_knees[i].inp[1] << std::endl;
        //std::cout << "copied input:" << inp.get_data()[0] << " " << inp.get_data()[1] << std::endl;
		VectorNd<float> denoiser_output = net->calc_output(inp);
        //std::cout << "knees_inp: " << noisy_knees[i].inp[0] << " " << noisy_knees[i].inp[1] << std::endl;
        //std::cout << "copied input:" << inp.get_data()[0] << " " << inp.get_data()[1] << std::endl;
        VectorNd<float> out(2,1,2500);
        VectorNd<float> in(2,1,2500);
        VectorNd<float> denoised_img_vec(2,1,2500);
        VectorNd<float> actual_noise(2,1,2500);
        VectorNd<float> perfect_image(2,1,2500);

        //std::cout << max<float>(inp) << " " << min<float>(inp) << std::endl;
        for(int j =0; j < 2500;j++){
                double real = denoiser_output.get_data()[j];
                double imag = denoiser_output.get_data()[j+2500];
                out.get_data()[j] = sqrt(real*real+imag*imag);
                double real_inp = (*noisy_knees_test)[i]->inp[j];
                double imag_inp = (*noisy_knees_test)[i]->inp[j+2500];
                //double real_inp = inp.get_data()[j];
                //double imag_inp = inp.get_data()[j+2500];
                //std::cout << real_inp << " " << imag_inp << std::endl;
                //std::cin.ignore();
                in.get_data()[j] = sqrt(real_inp*real_inp + imag_inp*imag_inp);
                //std::cout << in.get_data()[i] << std::endl;
                
                double real_denoised = (*noisy_knees_test)[i]->inp[j] - denoiser_output.get_data()[j];
                double imag_denoised = (*noisy_knees_test)[i]->inp[j+2500] - denoiser_output.get_data()[j+2500];
                denoised_img_vec.get_data()[j] = sqrt(real_denoised*real_denoised + imag_denoised*imag_denoised);
                
                double real_act_noise = actual_noise_compl.get_data()[j];
                double imag_act_noise = actual_noise_compl.get_data()[j+2500];
                actual_noise.get_data()[j] = sqrt(real_act_noise*real_act_noise + imag_act_noise * imag_act_noise);
                
                double real_perfect_image = (*noisy_knees_test)[i]->inp[j] - actual_noise_compl.get_data()[j];
                double imag_perfect_image = (*noisy_knees_test)[i]->inp[j+2500] - actual_noise_compl.get_data()[j+2500];
                perfect_image.get_data()[j] = sqrt(real_perfect_image*real_perfect_image + imag_perfect_image*imag_perfect_image);
            }
            
        float max_out = max<float>(out);
        float min_out = min<float>(out);
        float max_in = max<float>(in);
        float min_in = min<float>(in);
        float max_denoised_img = max<float>(denoised_img_vec);
        float min_denoised_img = min<float>(denoised_img_vec);
        float max_actual_noise = max<float>(actual_noise);
        float min_actual_noise = min<float>(actual_noise);
        float max_perfect_image = max<float>(perfect_image);
        float min_perfect_image = min<float>(perfect_image);
        //std::cout << max_in << " " << min_in << std::endl;
        for( int j =0; j <2500;j++){
            out.get_data()[j] = (out.get_data()[j]/(max_out )) * 255;
            
            in.get_data()[j] = (in.get_data()[j]/(max_in))*255 ;
            
           // std::cout << in.get_data()[j] << std::endl;
            if( in.get_data()[j] > 255 ){
                std::cout << "value bigger than 255" << std::endl;
            }
            denoised_img_vec.get_data()[j] = (denoised_img_vec.get_data()[j]/(max_denoised_img)) * 255;
            actual_noise.get_data()[j] = (actual_noise.get_data()[j]/(max_actual_noise))*255;
            perfect_image.get_data()[j] = (perfect_image.get_data()[j]/(max_perfect_image))*255;
        }
		CImg<float> denoiser_output_img(&out.get_data()[0],50,50,1);
		CImg<float> denoised_img(&denoised_img_vec.get_data()[0],50,50,1);
		CImg<float> denoised_img_real(&perfect_image.get_data()[0],50,50,1);
        CImg<float> noisy_img(&in.get_data()[0],50,50,1);
		CImg<float> noise(&actual_noise.get_data()[0],50,50,1);
        

        
		/*for( int i = 0; i < denoised_img.size(); i++){
			denoised_img._data[i] = min(max(noisy_img._data[i] - (denoiser_output_img._data[i]-(float)0.5),(float)0.0),(float)1.0);
			denoised_img_real._data[i] = min(max(noisy_img._data[i] - (noise._data[i]-(float)0.5),(float)0.0),(float)1.0);
		}*/
		

		//denoised_img = noisy_img - denoiser_output;

		/*noisy_img*=255;
		noise*=255;
		denoised_img*=255;
		denoised_img_real*=255;
		denoiser_output_img*=255;
*/
		//ss1 << "images/noisy_img_"<< i << ".bmp";
		//ss2 << "images/noise_"<<i<<".bmp";
		//ss3 << "images/denoised_img_"<<i<<".bmp";

        noisy_img.save("/home/lennart/Dokumente/Logfiles_denoising/images_knee_final_6_gpu/noisy_img.bmp",i);
		noise.save("/home/lennart/Dokumente/Logfiles_denoising/images_knee_final_6_gpu/noise_.bmp",i);
		denoiser_output_img.save("/home/lennart/Dokumente/Logfiles_denoising/images_knee_final_6_gpu/denoiser_output_.bmp",i);
		denoised_img.save("/home/lennart/Dokumente/Logfiles_denoising/images_knee_final_6_gpu/denoised_image_.bmp",i);
		denoised_img_real.save("/home/lennart/Dokumente/Logfiles_denoising/images_knee_final_6_gpu/denoised_img_real_.bmp",i);
		
	}


	delete net;
/*
	Network * net_loaded = new Network();
	std::ifstream ifsload("saved_net.dat");
	boost::archive::binary_iarchive ia(ifsload);
	ia >> net_loaded;
	std::cout << "net loaded" << std::endl;
	ifsload.close();*/
/*
	int correct_counter = 0;
	//std::cout << test_data.inp.getDimensionSize(0) << "  " << test_data.inp.getDimensionSize(1) << std::endl;
	//VectorNd<float> out = net->calc_output(test_data_inp);
	//std::cout << "test" << std::endl;
	for( int i = 0; i < 10000;i++){
		VectorNd<float> inp(2,1,784);

		std::copy( mnist_test[i].inp.begin(), mnist_test[i].inp.end(), inp.get_data().begin());
		VectorNd<float> out = net->calc_output(inp);
		float jmax_out = 0.0; float jmax_r = 0.0;
		int max_num_out = 0; int max_num_r = 0;
		for( int j =0; j < 10; j++){
			if( out(2,0,j) > jmax_out){
				jmax_out = out(2,0,j);
				max_num_out = j;
			}

			if( mnist_test[i].out[j]> jmax_r){
				jmax_r = mnist_test[i].out[j];
				max_num_r = j;
			}
		}
		if( max_num_r == max_num_out){
			correct_counter++;
		}
		//std:cout << i << std::endl;

		if( i % 100 == 0&& i !=0){
			//std::cout << correct_counter/(float)i << std::endl;
		}

	}
	std::cout <<" Test accuarcy is: " <<  correct_counter/(float)10000 << std::endl;
	/*cout << " calculating output for one vec" << endl;

	vector<float> out = net->calc_and_write_output(inp);
	cout <<  out[0] << endl;
	for( int i = 0; i < inp.size();i++){
		inp[i] = (float)rand()/RAND_MAX;
	}
	inp[0] = 0.5;

	out = net->calc_and_write_output(inp);
	cout << out[0] << endl;

	for( int i= 0; i < 100; i++){
		vector<float> inp = vector<float>(1,i*0.01);
		vector<float> out = net->calc_output(inp);
		std::cout << i*0.01 << " " << out[0] << std::endl;
	}

	net->print_layers();
	 */





	//test_data_inp.get_data().clear();
	//.get_data().clear();
	//cout << "Output: " << out[0] <<  endl; // prints
	return 0;
}
