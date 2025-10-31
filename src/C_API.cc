#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/export.hpp>
#include "Network.h"
BOOST_CLASS_EXPORT(MRINN::Network)
#include "Layers/Layer.h"
BOOST_CLASS_EXPORT(MRINN::Layer)
#include <fstream>
#include <iostream>

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
#include "Layers/convolution2dcudnn.h"
BOOST_CLASS_EXPORT(MRINN::Convolution2DCudnn)
//#include "Layers/BatchNormalizationLayer.h"
//BOOST_CLASS_EXPORT(MRINN::BatchNormalizationLayer)

#include "VectorNd.h"


using namespace MRINN;
using namespace std;

Network * net;
int _size_x, _size_y;

extern "C" bool load_net(int x_size, int y_size,char *filename);
bool load_net(int x_size, int y_size,char *filename){
    try{
    //Network * net_loaded = new Network();
    std::cout << "trying to load: " << filename << std::endl;
	std::ifstream ifsload1(filename);
    std::cout << "test2" << std::endl;
	boost::archive::binary_iarchive ia1(ifsload1);
    
    Network * loaded_net;
	ia1 >> loaded_net;
	std::cout << "net loaded" << std::endl;
    std::cout << "net size: " << loaded_net->Layers.size() << " " << x_size<< " " << y_size<< std::endl;
    
    _size_x = x_size;
    _size_y = y_size;
   	vector<int> inp_size1(3,0); inp_size1[0] = 2; inp_size1[1] = y_size; inp_size1[2] = x_size; // 748
	vector<int> out_size1(3,50); out_size1[0] = 64; out_size1[1] = y_size; out_size1[2] = x_size;
    std::cout << y_size << " " << x_size << std::endl;
    std::cout << "setting up network" << std::endl;
    net = new Network();
    
   // net->add_layer(new ConvolutionalLayer(inp_size1, inp_size1, 3, 2, dnnBorderZeros, true));
   // std::vector<float> params(9*4,0.0);
    //for( int i=0; i < 9*4;i++){
    //    params[i] = 0.0;
    //}
   /* params[4] = 1.0;
    params[13] = 0.0;
    params[22] = 0.0;
    params[31] = 1.0;
    net->Layers[0]->set_var_parameters(params);*/
    /*net->add_layer(new ConvolutionalLayer(inp_size1, out_size1, 3, 64, dnnBorderZeros, true));
    net->add_layer(new ReluLayer(out_size1,out_size1));
    net->add_layer(new ConvolutionalLayer(out_size1,out_size1, 3, 64, dnnBorderZeros,true));
    net->add_layer(new ReluLayer(out_size1,out_size1));
    net->add_layer(new ConvolutionalLayer(out_size1,out_size1, 3, 64, dnnBorderZeros,true));
    net->add_layer(new ReluLayer(out_size1,out_size1));
    net->add_layer(new ConvolutionalLayer(out_size1,out_size1, 3, 64, dnnBorderZeros,true));
    net->add_layer(new ReluLayer(out_size1,out_size1));
    net->add_layer(new ConvolutionalLayer(out_size1,inp_size1, 3, 2, dnnBorderZeros,true));
    */
    Convolution2DCudnn * convo_first = new Convolution2DCudnn(inp_size1, out_size1,3,64,true);
    Convolution2DCudnn * convo_second = new Convolution2DCudnn(out_size1, out_size1,3,64,true);
    Convolution2DCudnn * convo_third = new Convolution2DCudnn(out_size1, out_size1,3,64,true);
    Convolution2DCudnn * convo_fourth = new Convolution2DCudnn(out_size1, out_size1,3,64,true);
    Convolution2DCudnn * convo_fifth = new Convolution2DCudnn(out_size1, out_size1,3,64,true);
    Convolution2DCudnn * convo_output = new Convolution2DCudnn(out_size1, inp_size1,3,2,true);
    
    net->add_layer(convo_first);
	net->add_layer(new ReluLayer(out_size1,out_size1));
    
    net->add_layer(convo_second);
	net->add_layer(new ReluLayer(out_size1,out_size1));
    
    net->add_layer(convo_third);
	net->add_layer(new ReluLayer(out_size1,out_size1));
    
    net->add_layer(convo_fourth);
	net->add_layer(new ReluLayer(out_size1,out_size1));
    
    net->add_layer(convo_fifth);
	net->add_layer(new ReluLayer(out_size1,out_size1));
    
    net->add_layer(convo_output);
    
    
    std::cout << "copying parameters" << std::endl;
    
    for( int i_layer = 0; i_layer < loaded_net->Layers.size();i_layer+=1){
            std::vector<float> params;
            std::vector<float> params_net;
            params = (loaded_net->Layers[i_layer]->get_var_parameters());
            params_net = (net->Layers[i_layer]->get_var_parameters());
            if (params.size() != 0&& params_net.size() != 0)
            {
                std::cout << params.size() << " " << params_net.size() << std::endl;
                for( int j = 0; j < params.size();j++){
                //   std::cout << params[j];
                }
                std::cout << std::endl;
                net->Layers[i_layer]->set_var_parameters(params);
            }
            //net->Layers[i_layer]->set_var_parameters(params_net);
            //std::cout << "test" << std::endl;
    }
    //net->Layers[0].set_var_parameters(loaded_net->Layers[0].get_var_parameters())
    
	//net->add_layer(new BatchNormalizationLayer( out_size1, out_size1,0.00000001));
	ifsload1.close(); 
    std::cout << " Sucessfully loaded network: " <<filename <<  std::endl;
    }catch(...){
        std::cout << "net loading failed" << std::endl;
        return false;
    }
    return true;
    
   // std::cout << "test" << std::endl;
    //return true;
}

extern "C" float * process_input( float * inp, float * output);
float * process_input( float * inp, float * output){
    VectorNd<float> v_inp(2,1,2*_size_x*_size_y);
    std::copy(inp, &inp[_size_x*_size_y*2], &v_inp.get_data()[0]);
    //std::cout << "calculation output" << std::endl;
    //v_inp.print();
    VectorNd<float> out = net->calc_output(v_inp);
    //std::cout << "calculation output ended" << std::endl;
    //std::cout << 2*_size_x*_size_y << " " << out.get_data().size() << std::endl;
    //float * output =(float *) malloc(sizeof(float)*2*_size_x*_size_y);
    std::copy(out.get_data().begin(), out.get_data().end(), output);
    //std::cout << "returning output" << std::endl;
    //out.print();
    return output;
    /*VectorNd<float> inp_dat(2,batches,size);
    std::copy(inp, &inp[0]+size*batches, &inp_dat.get_data()[0]);
    VectorNd<float> out = net->calc_output(inp_dat);
    float * output_arr = new float[out.get_data().size()];
    std::copy(out.get_data().begin(), out.get_data().end(), output_arr);
    return output_arr;
    */
    //return NULL;
}
