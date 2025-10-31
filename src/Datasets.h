/*
 * Datasets.h
 *
 *  Created on: Mar 21, 2018
 *      Author: lennart
 */

#ifndef DATASETS_H_
#define DATASETS_H_

#include "Data.h"
#include <iostream>
#include <algorithm>
#include <random>

namespace MRINN{
	namespace Datasets{
		namespace FullySampledKnees{
			/*
			 * Returns n images from the CIFAR-10 dataset with the noise as output
			 */
			std::vector<Data> & load_noisy_knees_train(std::vector<Data> & train_data, int n, double to_sigma){
				if( n < 0 )
					n = 0;
				if( n> 76075)
					n = 76075;

				//char * memblock;
				try{

					std::ifstream file1 ("image_slices_reconstructed/knee_patches_5_50_2_splitcomplex.bin", std::ios::in|std::ios::binary|std::ios::ate);

					std::streampos size1 = file1.tellg();
					//std::cout << size1 << " " << n*8*50*50 << std::endl;



					size_t filesize= size1;

					//memblock = new char[n*2*(50*50)];
                    //std::cout << size1 << " " << filesize << " " << sizeof(float) << " " << sizeof(unsigned char) << std::endl;
                    std::vector<float> inp_data(size1/sizeof(float));
                    file1.seekg (0, std::ios::beg);
					file1.read(reinterpret_cast<char*>(&inp_data[0]), size1);
                    
					//std::cout << "test6" << std::endl;

					file1.close();


					std::default_random_engine generator;





					train_data.resize(n);
					//std::cout << train_data.size() << std::endl;
					//std::cout << "test" << std::endl;
					for( int i = 0; i < n;i++){
					//	std::cout << i << std::endl;
						std::vector<float> inp= std::vector<float>(2500*2,0.0);
						std::vector<float> out = std::vector<float>(2500*2,0.0);

						float sigma = (rand()/(float)RAND_MAX)*to_sigma;

						std::normal_distribution<double> distribution(0.0,sigma);
						//std::cout << n*2500 << " " << n*(50*50) << std::endl;
                            //double sum_ = 0.0;
							for( int j = 0; j < 5000; j+=2)
							{

								double gaussian_rand_real = distribution(generator);
                                double gaussian_rand_imag = distribution(generator);
								//inp[j]  = max(min(((float)  *(unsigned char*)(&memblock[i*2500+j])/255.0)+gaussian_rand,1.0),0.0);
                                //inp[j] = (float) *(unsigned char*)(&memblock[i*2500*2+j])
                                
                                inp[j] = (inp_data[i*2500*2+j]) + gaussian_rand_real;
                                inp[j+1] = (inp_data[i*2500*2+(j+1)]) + gaussian_rand_imag;
                                
                                //sum += inp[j] + inp[j];
                              
								std::cout << inp_data[i*2500+j];

								//inp[j] = max(min(((float)  *(unsigned char*)(&memblock_images[16 + i*784+j])/255.0) + gaussian_rand,1.0),0.0);
								//std::cout << inp[j] << " " << gaussian_rand << " " << ((float)  *(unsigned char*)(&memblock_images[16 + i*784+j])/255.0) << std::endl;
								//out[j] = 0.5+gaussian_rand;
								//out[j] = 0.5 + inp[j] - ((float)  *(unsigned char*)(&memblock[i*2500*2+j])/255.0);
                                out[j] = inp[j] - inp_data[i*2500+j];
                                out[j+1] = inp[j+1] - inp_data[i*2500+j+1];
							}
							//std:cout << std::endl;
							//std::cin.ignore();

						//std::cout << i << " " << train_data.size()<< std::endl;
						//train_data[i].inp.swap(inp);// = inp;
						//train_data[i].out.swap(out);// = out;
						train_data[i].inp = inp;
						train_data[i].out = out;

						}
					std::random_shuffle(train_data.begin(), train_data.end());
					//delete [] memblock;
					//train_data.resize(n);
				} catch (const std::exception& e) {
						std::cout << "could not load knee images" << std::endl;
						std::cout << e.what() << std::endl;
				}
				//std::cout << train_data.size() << " " << train_data[0].inp.size() << " " << train_data[0].out.size() <<  std::endl;
				return train_data;
			}
			
						/*
			 * Returns n images from the CIFAR-10 dataset with the noise as output
			 */
			void load_noisy_knees_train_single_sigma(std::vector<Data*> * train_data, int n, double sigma){
				if( n < 0 )
					n = 0;
				if( n> 76075)
					n = 76075;

				try{

					std::ifstream file1 ("image_slices_reconstructed/knee_patches_5_50_2_splitcomplex.bin", std::ios::in|std::ios::binary|std::ios::ate);

					std::streampos size1 = file1.tellg();
					//std::cout << size1 << " " << n*8*50*50 << std::endl;



					size_t filesize= size1;

					//memblock = new char [n*2*(50*50)];

					file1.seekg (0, std::ios::beg);
                    //std::cout << size1 << " " << sizeof(float) << " " <<size1/sizeof(float) << std::endl;
                    std::vector<float> inp_data(size1/sizeof(float));
                    
                    file1.seekg (0, std::ios::beg);
					file1.read(reinterpret_cast<char*>(&inp_data[0]), size1);
					//std::cout << "test6" << std::endl;

					file1.close();

                    
					std::default_random_engine generator;





					//train_data.resize(n);
                    std::cout << "test" << std::endl;
                    (*train_data).clear();
                    std::cout << "test2" << std::endl;
					//std::cout << train_data.size() << std::endl;
					//std::cout << "test" << std::endl;
                    //float sigma = (rand()/(float)RAND_MAX)*sigma;
                    std::cout << "this is sigma:" << sigma << std::endl;
					for( int i = 0; i < n;i++){
					//	std::cout << i << std::endl;
						std::vector<float> inp= std::vector<float>(2500*2,0.0);
						std::vector<float> out = std::vector<float>(2500*2,0.0);

					

						std::normal_distribution<double> distribution(0.0,sigma);
						//std::cout << n*2500 << " " << n*(50*50) << std::endl;
                            double sum  = 0;
							for( int j = 0; j < 2500; j++)
							{


								//inp[j]  = max(min(((float)  *(unsigned char*)(&memblock[i*2500+j])/255.0)+gaussian_rand,1.0),0.0);
                                
                                inp[j] = inp_data[i*2500*2+j*2];// + gaussian_rand_real;
                                inp[j+2500] = inp_data[i*2500*2+(j*2+1)];// + gaussian_rand_imag;
                                sum += inp[j];
                                sum += inp[j+2500];
                                //if( i ==1 ){
                                //std::cout << inp_data[i*2500*2+j] << " " << inp_data[i*2500*2+j+1] <<std::endl;
                                //std::cin.ignore();
                                //}
                                //sum += inp[j] + inp[j+1];
								//std::cout << (float)  *(unsigned char*)(&memblock[i*2500+j]);

								//inp[j] = max(min(((float)  *(unsigned char*)(&memblock_images[16 + i*784+j])/255.0) + gaussian_rand,1.0),0.0);
								//std::cout << inp[j] << " " << gaussian_rand << " " << ((float)  *(unsigned char*)(&memblock_images[16 + i*784+j])/255.0) << std::endl;
								//out[j] = 0.5+gaussian_rand;
								//out[j] = inp[j] - inp_data[i*2*2500+j];
                                //out[j+1] = inp[j+1] - inp_data[i*2*2500+j+1];
                                //out[j] = gaussian_rand_real;
                                //out[j+1] = gaussian_rand_imag;
                                
                                
							}
							double mean = sum/5000;
                            double sum_stdev = 0.0;
                            for( int j = 0; j < 2500*2; j++)
							{
                                sum_stdev += (inp[j] - mean)*(inp[j]-mean);
                                //sum_stdev += (inp[j+1] - mean) *(inp[j+1]+mean);
                                inp[j] = inp[j]-mean;
                                
                            }
                            double stdev = sqrt(sum_stdev/(4999));
                            double adjust = 1/sqrt(5000); // calculate the adjusted standard derivation
                            //std::cout << stdev << " " << adjust << " " << mean << std::endl;
                            if( stdev < adjust) 
                                stdev = adjust;
                            
                            
                            for( int j = 0; j < 2500*2; j++){
                                double gaussian_rand_real = distribution(generator);
                                //double gaussian_rand_imag = distribution(generator);
                                //std::cout << mean << " " << inp[j]/stdev << " " << gaussian_rand_real << std::endl;
                                inp[j] = inp[j]/stdev + gaussian_rand_real;
                                
                                out[j] = gaussian_rand_real;
                            }
							//std:cout << std::endl;
							//std::cin.ignore();

						//std::cout << i << " " << train_data.size()<< std::endl;
						//train_data[i].inp.swap(inp);// = inp;
						//train_data[i].out.swap(out);// = out;
						Data *dat = new Data();
                        dat->inp = inp;
                        dat->out = out;
						//train_data[i].inp = inp;
						//train_data[i].out = out;
                        train_data->push_back(dat);

						}
					std::random_shuffle(train_data->begin(), train_data->end());
					//delete [] memblock;
					train_data->resize(n);
				} catch (const std::exception& e) {
						std::cout << "could not load knee images" << std::endl;
						std::cout << e.what() << std::endl;
				}
				//std::cout << train_data.size() << " " << train_data[0].inp.size() << " " << train_data[0].out.size() <<  std::endl;
				//return train_data;
			}
			
			void load_noisy_knees_test_single_sigma(std::vector<Data*> * train_data, int n, double sigma){
				if( n < 0 )
					n = 0;
				if( n> 76075)
					n = 76075;

				try{

					std::ifstream file1 ("image_slices_reconstructed/validation_knee_patches_5_50_2_splitcomplex.bin", std::ios::in|std::ios::binary|std::ios::ate);

					std::streampos size1 = file1.tellg();
					//std::cout << size1 << " " << n*8*50*50 << std::endl;



					size_t filesize= size1;

					//memblock = new char [n*2*(50*50)];

					file1.seekg (0, std::ios::beg);
                    //std::cout << size1 << " " << sizeof(float) << " " <<size1/sizeof(float) << std::endl;
                    std::vector<float> inp_data(size1/sizeof(float));
                    std::cout << "test" << std::endl;
                    file1.seekg (0, std::ios::beg);
					file1.read(reinterpret_cast<char*>(&inp_data[0]), size1);
					//std::cout << "test6" << std::endl;

					file1.close();


					std::default_random_engine generator;





					//train_data.resize(n);
                    (*train_data).clear();
					//std::cout << train_data.size() << std::endl;
					//std::cout << "test" << std::endl;
                    //float sigma = (rand()/(float)RAND_MAX)*sigma;
                    std::cout << "this is sigma:" << sigma << std::endl;
					for( int i = 0; i < n;i++){
					//	std::cout << i << std::endl;
						std::vector<float> inp= std::vector<float>(2500*2,0.0);
						std::vector<float> out = std::vector<float>(2500*2,0.0);

					

						std::normal_distribution<double> distribution(0.0,sigma);
						//std::cout << n*2500 << " " << n*(50*50) << std::endl;
                            double sum  = 0;
							for( int j = 0; j < 2500; j++)
							{


								//inp[j]  = max(min(((float)  *(unsigned char*)(&memblock[i*2500+j])/255.0)+gaussian_rand,1.0),0.0);
                                
                                inp[j] = inp_data[i*2500*2+j*2];// + gaussian_rand_real;
                                inp[j+2500] = inp_data[i*2500*2+(j*2+1)];// + gaussian_rand_imag;
                                sum += inp[j];
                                sum += inp[j+2500];
                                //if( i ==1 ){
                                //std::cout << inp_data[i*2500*2+j] << " " << inp_data[i*2500*2+j+1] <<std::endl;
                                //std::cin.ignore();
                                //}
                                //sum += inp[j] + inp[j+1];
								//std::cout << (float)  *(unsigned char*)(&memblock[i*2500+j]);

								//inp[j] = max(min(((float)  *(unsigned char*)(&memblock_images[16 + i*784+j])/255.0) + gaussian_rand,1.0),0.0);
								//std::cout << inp[j] << " " << gaussian_rand << " " << ((float)  *(unsigned char*)(&memblock_images[16 + i*784+j])/255.0) << std::endl;
								//out[j] = 0.5+gaussian_rand;
								//out[j] = inp[j] - inp_data[i*2*2500+j];
                                //out[j+1] = inp[j+1] - inp_data[i*2*2500+j+1];
                                //out[j] = gaussian_rand_real;
                                //out[j+1] = gaussian_rand_imag;
                                
                                
							}
							double mean = sum/5000;
                            double sum_stdev = 0.0;
                            for( int j = 0; j < 2500*2; j++)
							{
                                sum_stdev += (inp[j] - mean)*(inp[j]-mean);
                                //sum_stdev += (inp[j+1] - mean) *(inp[j+1]+mean);
                                inp[j] = inp[j]-mean;
                                
                            }
                            double stdev = sqrt(sum_stdev/(4999));
                            double adjust = 1/sqrt(5000); // calculate the adjusted standard derivation
                            //std::cout << stdev << " " << adjust << " " << mean << std::endl;
                            if( stdev < adjust) 
                                stdev = adjust;
                            
                            
                            for( int j = 0; j < 2500*2; j++){
                                double gaussian_rand_real = distribution(generator);
                                //double gaussian_rand_imag = distribution(generator);
                                //std::cout << mean << " " << inp[j]/stdev << " " << gaussian_rand_real << std::endl;
                                inp[j] = inp[j]/stdev + gaussian_rand_real;
                                
                                out[j] = gaussian_rand_real;
                            }
							//std:cout << std::endl;
							//std::cin.ignore();

						//std::cout << i << " " << train_data.size()<< std::endl;
						//train_data[i].inp.swap(inp);// = inp;
						//train_data[i].out.swap(out);// = out;
						Data *dat = new Data();
                        dat->inp = inp;
                        dat->out = out;
						//train_data[i].inp = inp;
						//train_data[i].out = out;
                        train_data->push_back(dat);

						}
					//std::random_shuffle(train_data->begin(), train_data->end());
					//delete [] memblock;
					(*train_data).resize(n);
				} catch (const std::exception& e) {
						std::cout << "could not load knee images" << std::endl;
						std::cout << e.what() << std::endl;
				}
				//std::cout << train_data.size() << " " << train_data[0].inp.size() << " " << train_data[0].out.size() <<  std::endl;
				//return train_data;
			}
		}/*end namespace FullySampledKnees*/
		namespace CIFAR10{
			/*
			 * Returns n images from the CIFAR-10 dataset with the noise as output
			 */
			std::vector<Data> & load_noisy_CIFAR10_train(std::vector<Data> & train_data, int n, double to_sigma){
				if( n < 0 )
					n = 0;
				if( n> 40000)
					n = 40000;

				char * memblock;
				std::ifstream file1 ("cifar-10-batches-bin/data_batch_1.bin", std::ios::in|std::ios::binary|std::ios::ate);
				std::streampos size1 = file1.tellg();
				std::ifstream file2 ("cifar-10-batches-bin/data_batch_2.bin", std::ios::in|std::ios::binary|std::ios::ate);
				std::streampos size2 = file2.tellg();
				std::ifstream file3 ("cifar-10-batches-bin/data_batch_3.bin", std::ios::in|std::ios::binary|std::ios::ate);
				std::streampos size3 = file3.tellg();
				std::ifstream file4 ("cifar-10-batches-bin/data_batch_4.bin", std::ios::in|std::ios::binary|std::ios::ate);
				std::streampos size4 = file4.tellg();
				std::ifstream file5 ("cifar-10-batches-bin/data_batch_5.bin", std::ios::in|std::ios::binary|std::ios::ate);
				std::streampos size5 = file5.tellg();


				std::cout << "test" << std::endl;
				size_t filesize= size1 + size2 + size3 + size4 + size5;
				std::cout << filesize << " " << size1 << " " << size2 << " " << size3 << " " << size4 << " " << size5 << std::endl;

				memblock = new char [filesize];
				//std::cout << "test1" << std::endl;
				file1.seekg (0, std::ios::beg);
				file1.read(memblock, size1);
				//std::cout << "test2" << std::endl;
				file2.seekg(0, std::ios::beg);
				file2.read(&memblock[size1],size2);
				//std::cout << "test3" << std::endl;
				file3.seekg(0,std::ios::beg);
				file3.read(&memblock[size1+size2],size3);
				//std::cout << "test4" << std::endl;
				file4.seekg(0,std::ios::beg);
				file4.read(&memblock[size1+size2+size3], size4);
				//std::cout << "test5" << std::endl;
				file5.seekg(0,std::ios::beg);
				file5.read(&memblock[size1+size2+size3+size4], size5);
				//std::cout << "test6" << std::endl;

				file1.close();
				file2.close();
				file3.close();
				file4.close();
				file5.close();

				std::default_random_engine generator;





				train_data.resize(40000);
				for( int i = 0; i < 40000;i++){
					//std::cout << i << std::endl;
					std::vector<float> inp= std::vector<float>(3072,0.0);
					std::vector<float> out = std::vector<float>(3072,0.0);

					float sigma = (rand()/(float)RAND_MAX)*to_sigma;

					std::normal_distribution<double> distribution(0.0,sigma);

						for( int j = 1; j <= 3072; j++)
						{
							double gaussian_rand = distribution(generator);
							inp[j]  = max(min(((float)  *(unsigned char*)(&memblock[i*3073+j])/255.0)+gaussian_rand,1.0),0.0);
							//inp[j] = max(min(((float)  *(unsigned char*)(&memblock_images[16 + i*784+j])/255.0) + gaussian_rand,1.0),0.0);
							//std::cout << inp[j] << " " << gaussian_rand << " " << ((float)  *(unsigned char*)(&memblock_images[16 + i*784+j])/255.0) << std::endl;
							//out[j] = 0.5+gaussian_rand;
							out[j] = 0.5 + inp[j] - ((float)  *(unsigned char*)(&memblock[i*3073+j])/255.0);

						}


						//std::cout << i << " " << train_data.size()<< std::endl;
					train_data[i].inp = inp;
					train_data[i].out = out;


				}
				std::random_shuffle(train_data.begin(), train_data.end());
				train_data.resize(n);
				return train_data;
			}
			/*
			 * Returns n images from the CIFAR-10 test dataset with the noise as output
			 */
			std::vector<Data> & load_noisy_CIFAR10_test(std::vector<Data> & train_data, int n, double to_sigma){
				if( n < 0 )
					n = 0;
				if( n> 10000)
					n = 10000;

				char * memblock;
				std::ifstream file1 ("cifar-10-batches-bin/test_batch.bin", std::ios::in|std::ios::binary|std::ios::ate);
				std::streampos size1 = file1.tellg();

				memblock = new char [size1];
				file1.seekg (0, std::ios::beg);
				file1.read(memblock, size1);


				file1.close();


				std::default_random_engine generator;





				train_data.resize(10000);
				for( int i = 0; i < 10000;i++){
					std::vector<float> inp= std::vector<float>(3072,0.0);
					std::vector<float> out = std::vector<float>(3072,0.0);

					float sigma = (rand()/(float)RAND_MAX)*to_sigma;

					std::normal_distribution<double> distribution(0.0,sigma);

						for( int j = 1; j <= 3072; j++)
						{
							double gaussian_rand = distribution(generator);
							inp[j]  = max(min(((float)  *(unsigned char*)(&memblock[i*3073+j])/255.0)+gaussian_rand,1.0),0.0);
							//inp[j] = max(min(((float)  *(unsigned char*)(&memblock_images[16 + i*784+j])/255.0) + gaussian_rand,1.0),0.0);
							//std::cout << inp[j] << " " << gaussian_rand << " " << ((float)  *(unsigned char*)(&memblock_images[16 + i*784+j])/255.0) << std::endl;
							//out[j] = 0.5+gaussian_rand;
							out[j] = 0.5 + inp[j] - ((float)  *(unsigned char*)(&memblock[i*3073+j])/255.0);

						}


						//std::cout << i << " " << train_data.size()<< std::endl;
					train_data[i].inp = inp;
					train_data[i].out = out;


				}
				std::random_shuffle(train_data.begin(), train_data.end());
				train_data.resize(n);
				return train_data;
			}
			/*
			 * Returns n images from the CIFAR-10 dataset with labels
			 */
			void load_CIFAR10_train( std::vector<Data*> * train_data, int n){
				if( n < 0 )
					n = 0;
				if( n> 50000)
					n = 50000;

				char * memblock;
				std::ifstream file1 ("cifar-10-batches-bin/data_batch_1.bin", std::ios::in|std::ios::binary|std::ios::ate);
				std::streampos size1 = file1.tellg();
				std::ifstream file2 ("cifar-10-batches-bin/data_batch_2.bin", std::ios::in|std::ios::binary|std::ios::ate);
				std::streampos size2 = file2.tellg();
				std::ifstream file3 ("cifar-10-batches-bin/data_batch_3.bin", std::ios::in|std::ios::binary|std::ios::ate);
				std::streampos size3 = file3.tellg();
				std::ifstream file4 ("cifar-10-batches-bin/data_batch_4.bin", std::ios::in|std::ios::binary|std::ios::ate);
				std::streampos size4 = file4.tellg();
				std::ifstream file5 ("cifar-10-batches-bin/data_batch_5.bin", std::ios::in|std::ios::binary|std::ios::ate);
				std::streampos size5 = file5.tellg();
                //std::cout << size1 << " " << size2 << " " << size3 << " " << size4  << " " << size5 << std::endl;


				memblock = new char [size1+ size2 + size3 +size4+ size5];

				file1.seekg (0, std::ios::beg);
				file1.read(memblock, size1);
				file2.seekg(0, std::ios::beg);
				file2.read(&memblock[size1],size2);
				file3.seekg(0,std::ios::beg);
				file3.read(&memblock[size1+size2],size3);
				file4.seekg(0,std::ios::beg);
				file4.read(&memblock[size1+size2+size3], size4);
				file5.seekg(0,std::ios::beg);
				file5.read(&memblock[size1+size2+size3+size4], size5);

				file1.close();
				file2.close();
				file3.close();
				file4.close();
				file5.close();

                //std::cout << "before resize" << std::endl;


                
				//train_data->resize(50000);
                train_data->clear();
                //std::cout << "after resize" << std::endl;
				for( int i = 0; i < 50000;i++){
					std::vector<float> inp= std::vector<float>(3072,0.0);
					std::vector<float> out = std::vector<float>(10,0.0);
					out[1] = ((int)  *(unsigned char*)(&memblock[i*3073]) == 1) ;
					out[2] = ((int)  *(unsigned char*)(&memblock[i*3073]) == 2) ;
					out[3] = ((int)  *(unsigned char*)(&memblock[i*3073]) == 3) ;
					out[4]  = ((int)  *(unsigned char*)(&memblock[i*3073]) == 4) ;
					out[5]  = ((int)  *(unsigned char*)(&memblock[i*3073]) == 5) ;
					out[6]  = ((int)  *(unsigned char*)(&memblock[i*3073]) == 6) ;
					out[7]  = ((int)  *(unsigned char*)(&memblock[i*3073]) == 7) ;
					out[8]  = ((int)  *(unsigned char*)(&memblock[i*3073]) == 8) ;
					out[9]  = ((int)  *(unsigned char*)(&memblock[i*3073]) == 9) ;
					out[0]  = ((int) *(unsigned char*)(&memblock[i*3073]) == 0) ;
                    
					/*out[1] = 1.0 ;
					out[2] = 1.0 ;
					out[3] = 1.0 ;
					out[4]  = 1.0 ;
					out[5]  = 1.0 ;
					out[6]  = 1.0 ;
					out[7]  = 1.0 ;
					out[8]  = 1.0 ;
					out[9]  = 1.0;
					out[0]  = 1.0 ;*/
						for( int j = 1; j <= 3072; j++)
						{
							inp[j-1]  = ((float)  *(unsigned char*)(&memblock[i*3073+j])/255.0);
                            //inp[j-1] =1.0;

						}

                    //std::cout << i << std::endl;
                    //std::cin.ignore();
						//std::cout << i << " " << train_data.size()<< std::endl;
                    Data *dat = new Data();
                    dat->inp = inp;
                    dat->out = out;
                    train_data->push_back(dat);
					//(*train_data)[i].inp = inp;
					//(*train_data)[i].out = out;


				}
				//std:cout << "after loading" << std::endl;
				std::random_shuffle((*train_data).begin(), (*train_data).end());
				(*train_data).resize(n);
				//return train_data;
			}

			/*
			 * Returns n images from the CIFAR-10 test dataset with labels
			 */
			void load_CIFAR10_test(std::vector<Data*> * train_data, int n){
				if( n < 0 )
					n = 0;
				if( n> 10000)
					n = 10000;

				char * memblock;
				std::ifstream file1 ("cifar-10-batches-bin/test_batch.bin", std::ios::in|std::ios::binary|std::ios::ate);
				std::streampos size1 = file1.tellg();



				memblock = new char [size1];
				file1.seekg (0, std::ios::beg);
				file1.read(memblock, size1);
				file1.close();

				//(*train_data).resize(10000);
                (*train_data).clear();
				for( int i = 0; i < 10000;i++){
					std::vector<float> inp= std::vector<float>(3072,0.0);
					std::vector<float> out = std::vector<float>(10,0.0);
					out[1] = ((int)  *(unsigned char*)(&memblock[i*3073]) == 1) ;
					out[2] = ((int)  *(unsigned char*)(&memblock[i*3073]) == 2) ;
					out[3] = ((int)  *(unsigned char*)(&memblock[i*3073]) == 3) ;
					out[4]  = ((int)  *(unsigned char*)(&memblock[i*3073]) == 4) ;
					out[5]  = ((int)  *(unsigned char*)(&memblock[i*3073]) == 5) ;
					out[6]  = ((int)  *(unsigned char*)(&memblock[i*3073]) == 6) ;
					out[7]  = ((int)  *(unsigned char*)(&memblock[i*3073]) == 7) ;
					out[8]  = ((int)  *(unsigned char*)(&memblock[i*3073]) == 8) ;
					out[9]  = ((int)  *(unsigned char*)(&memblock[i*3073]) == 9) ;
					out[0]  = ((int) *(unsigned char*)(&memblock[i*3073]) == 0) ;

						for( int j = 1; j <= 3072; j++)
						{
							inp[j-1]  = ((float)  *(unsigned char*)(&memblock[i*3073+j])/255.0);
						}


						//std::cout << i << " " << train_data.size()<< std::endl;
                    Data *dat = new Data();
                    dat->inp = inp;
                    dat->out = out;
                    train_data->push_back(dat);
					//(*train_data)[i].inp = inp;
					//(*train_data)[i].out = out;


				}
				std::random_shuffle((*train_data).begin(), (*train_data).end());
				(*train_data).resize(n);
				//return train_data;
			}
		}
		namespace MNIST {
			/**
			 * Returns n noisy MNIST images with the corresponding residual noise from the train dataset
			 */
			std::vector<Data> & load_noisy_mnist_train(std::vector<Data> & train_data, int n, double sigma){
			    std::streampos size;
				char * memblock;
				std::ifstream file ("train-labels.idx1-ubyte", std::ios::in|std::ios::binary|std::ios::ate);
				size = file.tellg();
				memblock = new char [size];
				file.seekg (0, std::ios::beg);
				file.read (memblock, size);
				file.close();

				std::streampos size2;
				char * memblock_images;
				std::ifstream imagefile("train-images.idx3-ubyte", std::ios::in|std::ios::binary|std::ios::ate);

				size2 = imagefile.tellg();
				memblock_images = new char [size2];
				imagefile.seekg(0,std::ios::beg );
				imagefile.read(memblock_images,size2);
				imagefile.close();
                
				train_data.resize(60000);

				 std::default_random_engine generator;
				 std::normal_distribution<double> distribution(0.0,sigma);


				for( int i = 0; i < n;i++){
					std::vector<float> inp= std::vector<float>(784,0.0);
					std::vector<float> out = std::vector<float>(784,0.0);

					for( int j = 0; j < 784; j++)
					{
						double gaussian_rand = distribution(generator);
						inp[j] = max(min(((float)  *(unsigned char*)(&memblock_images[16 + i*784+j])/255.0) + gaussian_rand,1.0),0.0);
						//std::cout << inp[j] << " " << gaussian_rand << " " << ((float)  *(unsigned char*)(&memblock_images[16 + i*784+j])/255.0) << std::endl;
						//out[j] = 0.5+gaussian_rand;
						out[j] = 0.5 + inp[j] - ((float)  *(unsigned char*)(&memblock_images[16 + i*784+j])/255.0);
					}


					train_data[i].inp = inp;
					train_data[i].out = out;

				}
				std::random_shuffle(train_data.begin(), train_data.end());
				train_data.resize(n);
				return train_data;

			}

			/*
			 * Returns n noisy MNIST images with the corresponding residual noise from the test dataset
				*/
			std::vector<Data> & load_noisy_mnist_test(std::vector<Data> & train_data, int n, double sigma){
				if( n < 0 )
					n = 0;
				if( n> 10000)
					n = 10000;
				std::streampos size;
				char * memblock;
				std::ifstream file ("t10k-labels.idx1-ubyte", std::ios::in|std::ios::binary|std::ios::ate);
				size = file.tellg();
				memblock = new char [size];
				file.seekg (0, std::ios::beg);
				file.read (memblock, size);
				file.close();
					std::streampos size2;
				char * memblock_images;
				std::ifstream imagefile("t10k-images.idx3-ubyte", std::ios::in|std::ios::binary|std::ios::ate);
					size2 = imagefile.tellg();
				memblock_images = new char [size2];
				imagefile.seekg(0,std::ios::beg );
				imagefile.read(memblock_images,size2);
				imagefile.close();
				train_data.resize(10000);
					 std::default_random_engine generator;
				 std::normal_distribution<double> distribution(0.0,sigma);

				for( int i = 0; i < 10000;i++){
					std::vector<float> inp= std::vector<float>(784,0.0);
					std::vector<float> out = std::vector<float>(784,0.0);
					for( int j = 0; j < 784; j++)
					{
						double gaussian_rand = distribution(generator);
						inp[j] = max(min(((float)  *(unsigned char*)(&memblock_images[16 + i*784+j])/255.0) + gaussian_rand,1.0),0.0);
						//out[j] = 0.5+gaussian_rand;
						out[j] = 0.5 + inp[j] - ((float)  *(unsigned char*)(&memblock_images[16 + i*784+j])/255.0);
					}

					train_data[i].inp = inp;
					train_data[i].out = out;
					}
				std::random_shuffle(train_data.begin(), train_data.end());
				train_data.resize(n);
				return train_data;

			}


			/*
			 * Returns n entries from the randomly shuffeled MNIST training dataset with labels.
			 */
			void load_mnist_train(std::vector<Data*> * train_data, int n){
				if( n < 0 )
					n = 0;
				if( n> 60000)
					n = 60000;
			    std::streampos size;
				char * memblock;
				std::ifstream file ("train-labels.idx1-ubyte", std::ios::in|std::ios::binary|std::ios::ate);
				size = file.tellg();
				memblock = new char [size];
				file.seekg (0, std::ios::beg);
				file.read (memblock, size);
				file.close();

				std::streampos size2;
				char * memblock_images;
				std::ifstream imagefile("train-images.idx3-ubyte", std::ios::in|std::ios::binary|std::ios::ate);

				size2 = imagefile.tellg();
				memblock_images = new char [size2];
				imagefile.seekg(0,std::ios::beg );
				imagefile.read(memblock_images,size2);
				imagefile.close();
				//(*train_data).resize(0);
                (*train_data).clear();


				for( int i = 0; i < n;i++){
					//std::vector<float> inp= *(new std::vector<float>(784,0.0));
					//std::vector<float> out = *(new std::vector<float>(10,0.0));
                    std::vector<float> inp=  std::vector<float>(784,0.0);
					std::vector<float> out =  std::vector<float>(10,0.0);
                    out[1] = ((int)  *(unsigned char*)(&memblock[8+i]) == 1) ;
					out[2] = ((int)  *(unsigned char*)(&memblock[8+i]) == 2) ;
					out[3] = ((int)  *(unsigned char*)(&memblock[8+i]) == 3) ;
					out[4]  = ((int)  *(unsigned char*)(&memblock[8+i]) == 4) ;
					out[5]  = ((int)  *(unsigned char*)(&memblock[8+i]) == 5) ;
					out[6]  = ((int)  *(unsigned char*)(&memblock[8+i]) == 6) ;
					out[7]  = ((int)  *(unsigned char*)(&memblock[8+i]) == 7) ;
					out[8]  = ((int)  *(unsigned char*)(&memblock[8+i]) == 8) ;
					out[9]  = ((int)  *(unsigned char*)(&memblock[8+i]) == 9) ;
					out[0]  = ((int) *(unsigned char*)(&memblock[8+i]) == 0) ;

						for( int j = 0; j < 784; j++)
						{
							//std::cout << "after" << std::endl;
						  inp[j]  = ((float)  *(unsigned char*)(&memblock_images[16 + i*784+j])/255.0);
						}


						//std::cout << i << " " << train_data.size()<< std::endl;
                     Data * dat = new Data();
                     dat->inp= inp;
                     dat->out= out;
                     (*train_data).push_back(dat);
					//(*train_data)[i].inp = inp;
					//(*train_data)[i].out = out;


				}
				std::cout << "test" << std::endl;
				std::random_shuffle((*train_data).begin(), (*train_data).end());
				(*train_data).resize(n);
				//return train_data;
			}

			/*
			 * Returns n entries from the randomly shuffeled MNIST test dataset with labels.
			 */
			void load_mnist_test(std::vector<Data*> * test_data, int n){
				if( n < 0 )
					n = 0;
				if( n> 10000)
					n = 10000;
				std::streampos size;
				char * memblock;

				std::ifstream file ("t10k-labels.idx1-ubyte", std::ios::in|std::ios::binary|std::ios::ate);
				size = file.tellg();
				memblock = new char [size];
				file.seekg (0, std::ios::beg);
				file.read (memblock, size);
				file.close();
				std::streampos size2;
				char * memblock_images;
				std::ifstream imagefile("t10k-images.idx3-ubyte", std::ios::in|std::ios::binary|std::ios::ate);

				size2 = imagefile.tellg();
				memblock_images = new char [size2];
				imagefile.seekg(0,std::ios::beg );
				imagefile.read(memblock_images,size2);
				imagefile.close();
				//test_data->resize(n);
                test_data->clear();
				for( int i = 0; i < n;i++){
					std::vector<float> inp= std::vector<float>(784,0.0);
					std::vector<float> out = std::vector<float>(10,0.0);
					out[1] = ((int)  *(unsigned char*)(&memblock[8+i]) == 1) ;
					out[2] = ((int)  *(unsigned char*)(&memblock[8+i]) == 2) ;
					out[3] = ((int)  *(unsigned char*)(&memblock[8+i]) == 3) ;
					out[4]  = ((int)  *(unsigned char*)(&memblock[8+i]) == 4) ;
					out[5]  = ((int)  *(unsigned char*)(&memblock[8+i]) == 5) ;
					out[6]  = ((int)  *(unsigned char*)(&memblock[8+i]) == 6) ;
					out[7]  = ((int)  *(unsigned char*)(&memblock[8+i]) == 7) ;
					out[8]  = ((int)  *(unsigned char*)(&memblock[8+i]) == 8) ;
					out[9]  = ((int)  *(unsigned char*)(&memblock[8+i]) == 9) ;
					out[0]  = ((int) *(unsigned char*)(&memblock[8+i]) == 0) ;
					for( int j = 0; j < 784; j++)
					{
					  inp[j]  = ((float)  *(unsigned char*)(&memblock_images[16 + i*784+j])/255.0);
					}
                    Data * dat = new Data();
                     dat->inp= inp;
                     dat->out= out;
					(*test_data).push_back(dat);
				}
				std::random_shuffle(test_data->begin(), test_data->end());
				test_data->resize(n);
				//return test_data;
			}
		}
	}
}




#endif /* DATASETS_H_ */
