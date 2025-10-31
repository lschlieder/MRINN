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

int main() {
	//std::vector<int> v {10,20};
	srand (time(NULL));
	//srand (time(1337));
	//srand();
	omp_set_num_threads(4);
	Network * net = new Network();
	//Network * net_loaded = new Network();
	std::ifstream ifsload1("Logfiles_denoising/saved_net_probably6Layers_20_epochs_100_batchsize.dat");
	boost::archive::binary_iarchive ia1(ifsload1);
	ia1 >> net;
	std::cout << "net loaded" << std::endl;
	ifsload1.close();
    for (int i_layers = 0; i_layers < net->Layers.size();i_layers++){
        std::cout << net->Layers[i_layers]->get_var_parameters().size() << std::endl;
    }
    
    Network * new_net = new Network();
    vector<int> inp_size1(3,50); inp_size1[0] = 2; // 748
	vector<int> out_size1(3,50); out_size1[0] = 64;
    new_net->add_layer(new ConvolutionalLayer(inp_size1,out_size1, 3, 64, dnnBorderZeros,true)); //1
	//net->add_layer(new BatchNormalizationLayer( out_size1, out_size1,0.00000001));
	new_net->add_layer(new ReluLayer(out_size1,out_size1));
    new_net->add_layer(new ConvolutionalLayer(out_size1,out_size1, 3, 64, dnnBorderZeros,true)); //2
	//net->add_layer(new BatchNormalizationLayer( out_size1, out_size1,0.00000001));
	new_net->add_layer(new ReluLayer(out_size1,out_size1));
    new_net->add_layer(new ConvolutionalLayer(out_size1,out_size1, 3, 64, dnnBorderZeros,true)); //3
	//net->add_layer(new BatchNormalizationLayer( out_size1, out_size1,0.00000001));
	new_net->add_layer(new ReluLayer(out_size1,out_size1));
    new_net->add_layer(new ConvolutionalLayer(out_size1,out_size1, 3, 64, dnnBorderZeros,true)); //4
	//net->add_layer(new BatchNormalizationLayer( out_size1, out_size1,0.00000001));
	new_net->add_layer(new ReluLayer(out_size1,out_size1));
    new_net->add_layer(new ConvolutionalLayer(out_size1,out_size1, 3, 64, dnnBorderZeros,true)); //5
	//net->add_layer(new BatchNormalizationLayer( out_size1, out_size1,0.00000001));
	new_net->add_layer(new ReluLayer(out_size1,out_size1));
    new_net->add_layer(new ConvolutionalLayer(out_size1,inp_size1, 3, 2, dnnBorderZeros,true)); //6
	//net->add_layer(new BatchNormalizationLayer( out_size1, out_size1,0.00000001));
	//net->add_layer(new ReluLayer(out_size1,out_size1));
    for (int i_layers = 0; i_layers < net->Layers.size();i_layers++){
        std::vector<float> params;
        std::vector<float> new_params;
        new_params =  net->Layers[i_layers]->get_var_parameters();
        params = new_net->Layers[i_layers]->get_var_parameters();
        std::cout << params.size() << " " << new_params.size() << std::endl;
        if( new_params.size() != 0 && params.size() !=0){
            int bias_size = net->Layers[i_layers]->get_out_size()[0];
            std::cout << bias_size << std::endl;
            std::cin.ignore();
            std::reverse_copy(new_params.begin(),new_params.end()-bias_size, params.begin());
            std::copy(new_params.end()-bias_size,new_params.end(), params.end()-bias_size);
            new_net->Layers[i_layers]->set_var_parameters(params);
        }
    }
    
    vector<Data*> *noisy_knees = new std::vector<Data*>(0);
	vector<Data*> *noisy_knees_test = new std::vector<Data*>(0);
	Datasets::FullySampledKnees::load_noisy_knees_train_single_sigma(noisy_knees,50000,0.13);
    std::cout << "loading test data" << std::endl;
    Datasets::FullySampledKnees::load_noisy_knees_test_single_sigma(noisy_knees_test,10000, 0.13);
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
		VectorNd<float> denoiser_output = new_net->calc_output(inp);
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

        noisy_img.save("Logfiles_denoising/images_knees_CPU_TEST/noisy_img.bmp",i);
		noise.save("Logfiles_denoising/images_knees_CPU_TEST/noise_.bmp",i);
		denoiser_output_img.save("Logfiles_denoising/images_knees_CPU_TEST/denoiser_output_.bmp",i);
		denoised_img.save("Logfiles_denoising/images_knees_CPU_TEST/denoised_image_.bmp",i);
		denoised_img_real.save("Logfiles_denoising/images_knees_CPU_TEST/denoised_img_real_.bmp",i);
		
	}

    std::ofstream ofs("Logfiles_denoising/saved_net_cpu_6_layers_20_epochs_100.dat");
	boost::archive::binary_oarchive oa(ofs);
	oa << new_net;
	ofs.close();
    std::cout << "saved network!" << std::endl;
    
    
    
}
