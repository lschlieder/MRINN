 /*
 * Network.cpp
 *
 *  Created on: Nov 15, 2017
 *      Author: lennart
 */

#include "Network.h"
#include <iostream>
#include <mkl.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>    // std::random_shuffle
#include <time.h>
#include <chrono>
#include <iostream>
#include <fstream>

namespace MRINN {

Network::Network() {
	parameters = 0;
}



/**
 * set the input and output dimensions of the data
 */
void Network::set_data_dimensions(std::vector<int> inp_size, std::vector<int> out_size){
	int inp_num = 1;
	int out_num = 1;
	for( int i =0; i < inp_size.size();i++){
		inp_num = inp_num*inp_size[i];
	}
	for ( int i = 0; i < out_size.size(); i++){
		out_num = out_num*out_size[i];
	}

	this->dims.inp_num = inp_num;
	this->dims.out_num = out_num;
	this->dims.inp_dim = inp_size.size();
	this->dims.out_dim = out_size.size();
	this->dims.inp_size = inp_size;
	this->dims.out_size = out_size;
}

/**
 * set the input and output dimensions of the data
 * dims: the input_dimensions of the data
 */
void Network::set_data_dimensions(data_dimensions dims){
	this->dims = dims;
}

/**
 * get the data dimsions for the network
 */
data_dimensions Network::get_data_dimensions(){
	return this->dims;
}

/**
 * set the data to train the network
 */
void Network::set_data(std::vector<Data*> data){
	this->Training_Data = data;
}

/**
 * Set the validation data for the network
 */
void Network::set_validation_data(std::vector<Data*> data){
    this->Validation_Data = data;
}

/**
 * Add a layer to the neural network
 * The latest layer to be added is the output layer
 */
void Network::add_layer(Layer * layer){
    std::cout << "set layer" << std::endl;
	Layers.push_back(layer);
	set_data_dimensions(Layers[0]->get_inp_size(),Layers[Layers.size()-1]->get_out_size()); // update the dimensions of the input and output
    //std::cout << "after set" << std::endl;
    //const std::vector<float> params = layer->get_var_parameters();
    //std::cout << params.size() << std::endl;
    //std::cout << layer->get_var_parameter_number() << std::endl;
	parameters+= layer->get_var_parameter_number();

}

/**
 * calculate the output that the networks gives out for a given input
 * inp: the input data
 */

void Network::print_layers(){
	for( unsigned int i_layers = 0; i_layers < Layers.size(); i_layers++){
		Layers[i_layers]->print_weights();
	}
}
VectorNd<float> Network::calc_and_write_output(VectorNd<float> inp){

	VectorNd<float> layer_inp = inp;
	std::cout << "input:" << std::endl;
	for( int i_data = 0; i_data < layer_inp.get_data().size();i_data++){
		std::cout << layer_inp.get_data()[i_data] << " ";
	}
	std::cout << std::endl << std::endl;


	for( unsigned int i_layers = 0; i_layers < Layers.size(); i_layers++){
		std::cout << "Layer "<< i_layers << ":" << std::endl;
		Layers[i_layers]->print_weights();
		layer_inp = Layers[i_layers]->calc_output(layer_inp);
		std::cout << "input:" << std::endl;
		for( int i_data = 0; i_data < layer_inp.get_data().size();i_data++){
			std::cout << layer_inp.get_data()[i_data] << " ";
		}
		std::cout << std::endl << std::endl;
	}

	return layer_inp;

}
VectorNd<float> & Network::calc_output(VectorNd<float> & inp){

	//VectorNd<float> layer_inp = inp;
	for(  unsigned int i_layers = 0; i_layers < Layers.size(); i_layers++){
		//std::cout << inp.getDimensionSize(0) << std::endl;
		//layer_inp = Layers[i_layers]->calc_output(layer_inp);
		//std::cout << i_layers<< std::endl;
		//std::cout << inp.getDimensionSize(1) << std::endl;
		//auto cstart = std::chrono::high_resolution_clock::now();
        
         /*inp.print();
         for( int i =0; i< inp.get_data().size();i++){
            if (inp.get_data()[i] > 1e8)
                std::cout << "big value found" << std::endl;
         }
         std::cout << "above from layer: " << i_layers << std::endl;
         std::cin.ignore();*/

		inp = Layers[i_layers]->calc_output(inp);
        //inp.print();
        //std::cin.ignore();
		//auto cend = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float> elapsed_seconds = cend-cstart;
		//std::cout << "forward duration: " << elapsed_seconds.count()*1000 << " layer:"<< i_layers<< std::endl;

		//Layers[i_layers]->print_weights();
		//layer_inp = inp;


	}
	return inp;

}

VectorNd<float> & Network::calc_error(VectorNd<float> & error_inp, std::vector<float> & accumulated_gradient){

	//VectorNd<float> layer_inp(error_inp);

	/*for( int i_inp = 0; i_inp < layer_inp.size(); i_inp++){
		//std::cout << layer_inp.size() << " " << accumulated_error[Layers.size()].size()<< std::endl;
		accumulated_error.at(Layers.size()).at(i_inp) = accumulated_error.at(Layers.size()).at(i_inp) + layer_inp.at(i_inp);
	}*/
    //error_inp.print();
    //std::cin.ignore();
    
	int offset = 0;
	for( int i_layers = Layers.size()-1; i_layers >= 0; i_layers--){

		VectorNd<float> grad = Layers[i_layers]->get_parameter_gradient(error_inp);
        
        /*for(int i = 0; i < grad.get_data().size();i++){
            std::cout << grad.get_data()[i] << " " ;
        }
        std::cout << std::endl;
        std::cin.ignore();*/
		//VectorNd<float> grad(Layers[i_layers]->get_parameter_gradient(error_inp));
		//std::cout << " test4 " << i_layers << " " << grad.get_data().size() << " " << accumulated_gradient.size()<<" " << offset << " " << grad.getDimensionSize(1) << std::endl;
		if (grad.getDimensionSize(0) != 0){
            std::copy( grad.get_data().begin(), grad.get_data().end(),accumulated_gradient.begin()+offset);
            offset += grad.getDimensionSize(1);
        }

        //std::cout << grad.getDimensionSize(0) << " " << grad.getDimensionSize(1) << std::endl;
        

		//auto gradend = std::chrono::high_resolution_clock::now();
		//std::cout << grad.getDimensionSize(1) << " " << accumulated_gradient[i_layers].size()<< std::endl;
		//std::cout << grad.<< std::endl;
		//auto cstart = std::chrono::high_resolution_clock::now();

		error_inp = Layers[i_layers]->calc_error(error_inp);

        /*for(int i = 0; i < error_inp.get_data().size();i++){
            std::cout << error_inp.get_data()[i] << " " ;
        }
        std::cout << std::endl;
        std::cin.ignore();*/
		//auto cend = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float> elapsed_seconds = cend-cstart;
		//std::chrono::duration<float>elapsed_seconds2 = gradend-gradstart;
		//std::cout << "Error duration: " << elapsed_seconds.count()*1000 << " Gradient duration: " << elapsed_seconds2.count()*1000 <<" layer: " << i_layers << std::endl;


		//std::vector<float>temp_vec(784,1.0);
		//std::vector<float>temp_vec(grad.getDimensionSize(0),1.0);
		//std::cout << "grad:" ;
		///grad.print();
		//std::cout << temp_vec.size() << " " << grad.getDimensionSize(1) << " " << grad.getDimensionSize(0) << std::endl;
		/*for( int i_param = 0; i_param < grad.getDimensionSize(1);i_param++){
			accumulated_gradient[i_layers][i_param] =cblas_dasum(grad.getDimensionSize(0),&grad(2,0,i_param), grad.getDimensionSize(1) );
		}*/
		//cblas_dgemv(CblasRowMajor, CblasTrans, 1, 784, 1.0, &grad[0], 784, &temp_vec[0], 1.0, 0.0,&accumulated_gradient[i_layers][0], 1.0);

		//cblas_dgemv(CblasRowMajor, CblasTrans, grad.getDimensionSize(0), grad.getDimensionSize(1), 1.0, &grad.get_data()[0], grad.getDimensionSize(1), &temp_vec[0], 1.0, 0.0,&accumulated_gradient[i_layers][0], 1.0);




		/*std::cout << "accumulated_grad";
		for( int i =0; i < accumulated_gradient[i_layers].size();i++){
			std::cout << accumulated_gradient[i_layers][i] << " ";
		}
		std::cout << std::endl;*/
		//accumulated_gradient.
		//cblas_daxpy(grad.get_data().size(), 1.0, &grad.get_data()[0],1.0, &accumulated_gradient[i_layers][0], 1.0);
		/*for( int i_inp = 0; i_inp < layer_inp.size(); i_inp++){
			accumulated_error.at(i_layers).at(i_inp) = accumulated_error.at(i_layers).at(i_inp) + layer_inp.at(i_inp);
			//std::cout << i_layers << " " << i_inp<< " " << layer_inp.size()<< " " << accumulated_error[i_layers].size() <<  std::endl;
		}*/
	}
	return error_inp;
}


int get_accuracy(int size, VectorNd<float> & output, VectorNd<float> & labels ){
    int count = 0;
    for(int val_i = 0; val_i < size; val_i++){

        int max_res = 0;
        int max_labels = 0;
        float max_val_res = output(2,val_i,0);
        float max_val_labels = labels(2,val_i,0);
        //std::cout << validation_result.get_data().size() << " " << val_output.get_data().size() << std::endl;
        //std::cout << output.getDimensionSize(1) << std::endl;
        for( int output_i = 0; output_i < output.getDimensionSize(1); output_i++){
                
            if( output(2,val_i,output_i) > max_val_res ) {
                max_res = output_i;
                max_val_res = output(2,val_i,output_i);
            }
            if( labels(2,val_i,output_i) > max_val_labels){
                max_labels = output_i;
                max_val_labels = labels(2,val_i,output_i);
            }
                    
        }
                //std::cout << std::endl;
                
        if ( max_res == max_labels ) {
            count += 1;
        }
    
                //std::cout << (acc_loss) << std::endl;
    }
    return count;
}

/*
 * Trains the network with the given training options ( training algorithm, batchsize and epochs)
 */

void Network::train_network(training_opts opts){

	int i_count = 0;
    int i_batch_count_epochs  = 0;
	VectorNd<float> result;
	std::vector<float> parameter_vector(parameters);
    auto chrono_alltime_start = std::chrono::system_clock::now();
    double running_time = 0.0;

	float squared_error;

	opts.optimizer->set_size(parameters);
    
    ofstream logfile;
    ofstream validation_logfile;
    if( opts.log_loss == true){
        logfile.open(opts.log_filename);
        logfile << "#logfile Tensorflow run: "<< std::endl;
            logfile<< "#epochs: "<<opts.epochs <<", batch_size: "<< opts.batch_size << std::endl << 
                    "#num epoch is_validation_set   i   running_actuall_num   loss   average_loss all_time single_time    epoch_time running_training_counter count/i*batch_size percent"
                    << std::endl;
        //logfile << "#Logfile Training run. Number of Layers " << Layers.size() << " number of Training Data: " << Training_Data.size() << " Batch Size: " << opts.batch_size << " epochs: " << opts.epochs << std::endl;
        if( opts.use_validation_dataset){
            validation_logfile.open(opts.validation_log_filename);
            validation_logfile << "#logfile Tensorflow run: cpu mse validation file\n#epoch    count   percentage" << std::endl;
    
        }
    }
    
        
	//float * accumulated_gradient = new float[parameters];
	std::vector<float> accumulated_gradient(parameters,0.0);
	std::vector<std::vector<float> > temp_parameter(Layers.size(),std::vector<float>(0));
	for( int i_layer = 0; i_layer < Layers.size(); i_layer++){
		temp_parameter[i_layer] = std::vector<float>(Layers[i_layer]->get_var_parameter_number(),0.0);
		//accumulated_gradient[i_layer].fill(0.0);
	}
	for( int epochs = 0; epochs < opts.epochs; epochs++){
		std::cout << std::endl << "epoch: " << epochs << std::endl;
		int i_data = 0;
		VectorNd<float> input(2,opts.batch_size, dims.inp_num);
		VectorNd<float> output(2,opts.batch_size, dims.out_num);
		VectorNd<float> error_vec(2,opts.batch_size, dims.out_num);
	//	std::cout << opts.batch_size << " " << dims.inp_num << " " << dims.out_num << " " << dims.inp_dim << " " << dims.out_dim<< std::endl;


		for(int i_batch = 0; i_batch < opts.batch_size; i_batch++){
			//std::cout << Training_Data[i_batch+i_data].inp.size() << " " << Training_Data[i_batch+i_data].out.size() << " " << input.get_data().size() << std::endl;
			std::copy( Training_Data[i_batch+i_data]->inp.begin(), Training_Data[i_batch+i_data]->inp.end(),& input(2,i_batch,0));
			std::copy( Training_Data[i_batch+i_data]->out.begin(), Training_Data[i_batch+i_data]->out.end(),& output(2,i_batch,0));
		}

		std::random_shuffle(Training_Data.begin(), Training_Data.end());

		int batch_count = 0;
		squared_error = 0;
        int epoch_acc_count = 0;
        auto chrono_epoch_start = std::chrono::system_clock::now();
		while( i_data+ opts.batch_size <= Training_Data.size()){
            
//std::cout << "test:" << i_batch + i_data << std::endl;
			//int i_batch = 0;
			//std::vector<std::vector<float> > accumulated_error = std::vector< std::vector<float> >(Layers.size()+1,std::vector<float>(0,0.0));
			//std::vector< std::vector<float> > accumulated_gradient = std::vector<std::vector<float> >(Layers.size(), std::vector<float>(0,0.0));

			//std::vector<float> squared_error = std::vector<float>(dims.out_num,0.0);

			//VectorNd<std::vector<float> >  accumulated_gradient(2,opts.batch_size, Layers.size());
			std::fill(accumulated_gradient.begin(), accumulated_gradient.end(),0.0);
			/*std::vector< std::vector<float> > accumulated_gradient(Layers.size(),std::vector<float>(0,0.0));
			for( int i_layer = 0; i_layer < Layers.size(); i_layer++){
				accumulated_gradient[i_layer] = std::vector<float>(Layers[i_layer]->get_var_parameter_number(),0.0);
				//accumulated_gradient[i_layer].fill(0.0);
			}*/

			//accumulated_error.at(Layers.size()) = std::vector<float>(Layers[Layers.size()-1]->get_output_num(),0.0);

			clock_t start, end,end_actuall;
			//auto chrono_start, chrono_end;

			//VectorNd<float> input(this->dims.inp_num*opts.batch_size,0.0);
			VectorNd<float> input(2,opts.batch_size, dims.inp_num);
			VectorNd<float> output(2,opts.batch_size, dims.out_num);
			VectorNd<float> error_vec(2,opts.batch_size, dims.out_num);
			for(int i_batch = 0; (i_batch < opts.batch_size) && i_batch+i_data < Training_Data.size(); i_batch++){
				std::copy( Training_Data[i_batch+i_data]->inp.begin(), Training_Data[i_batch+i_data]->inp.end(),& input(2,i_batch,0));
				std::copy( Training_Data[i_batch+i_data]->out.begin(), Training_Data[i_batch+i_data]->out.end(),& output(2,i_batch,0));
			}
			
			
            
            //if( i_batch+i_data < Training_Data.size()) 
            //    break;
            //std::cout << i_batch +i_data << " " << i_batch << std::endl;
			/*std::cout << i_batch+i_data << " " << Training_Data[i_batch+i_data].inp.size() << std::endl;
			for( int i = 0; i < Training_Data[i_batch+i_data].inp.size(); i++){
				std::cout << Training_Data[i_batch+i_data].inp[i] << " " ;
			}
			std::cout << std::endl;*/


			//std::cout << "test1" << std::endl;
			//std::cout << "input: ";
			//input.print();

			start= clock();
			auto chrono_start = std::chrono::system_clock::now();
			//std::cout << "input :" << std::endl;
			//input.print();
          
			result = calc_output(input);
           
            
            double o_sum = 0.0;
            for( int out_i = 0; out_i < output.get_data().size(); out_i++){
                //o_sum+= output.get_data()[out_i]; 
                o_sum+= (output.get_data()[out_i]-result.get_data()[out_i] ) *(output.get_data()[out_i]-result.get_data()[out_i] ) ;
            }
            o_sum = o_sum / output.get_data().size();
            //std::cout <<"mse: " <<  o_sum << std::endl;
            //result.print();
			//std::cin.ignore();
			//cblas_saxpy(result.get_data().size(), -1.0, & output.get_data()[0], 1.0, &result.get_data()[0], 1.0);
            
            //epoch_acc_count += get_accuracy(opts.batch_size, result, output);
            epoch_acc_count = 0;
			//squared_error= cblas_sdot(result.get_data().size(), &result(2,0,0), 1.0, &result(2,0,0), 1.0)/((float) opts.batch_size);
            double loss = 0.0;
            
            loss = opts.loss->get_loss(output,result);
           
			squared_error += loss;
			//std::cout << "loss:" << std::endl;
			//std::cout << squared_error << std::endl;
			auto chrono_end_forward = std::chrono::system_clock::now();

			calc_error(result,accumulated_gradient);
            
			/*std::cout << " gradient: " << std::endl;
			for( int i = 0; i < accumulated_gradient.size();i++){
				std::cout << accumulated_gradient[i] << " ";
			}
			std::cout << std::endl;
			std::cin.ignore();*/

			end = clock();
			auto chrono_end = std::chrono::system_clock::now();
			i_count = i_count + opts.batch_size;

            

            batch_count++;
            i_batch_count_epochs++;

			if(  i_count > 0){
                
				std::cout << "\r";
				std::chrono::duration<float> elapsed_seconds = chrono_end-chrono_start;
				std::cout << "\r data: " << opts.batch_size + i_data<< "/" <<Training_Data.size()<< " " << squared_error/batch_count;
				std::cout <<"time: " << 1000* (end-start)/((float) CLOCKS_PER_SEC) <<  " one cycle time: " << 1000* (end-start)/((float)opts.batch_size * CLOCKS_PER_SEC)  << " Difftime: "<< 1000*(elapsed_seconds.count())/opts.batch_size;
				//std::cout << std::endl;
				i_count = 0;
                //std::string test; 
                //std::cin >> test; 
			}
			clock_t startt,endt;
			startt = clock();
			//std::cout << std::endl;
			//std::vector<float>::iterator parameters_ptr = parameter_vector.begin();
			float * parameters_ptr = &parameter_vector[0];
           
			for( int i_layer = Layers.size(); i_layer > 0; i_layer--){ //goes from layers.size to 1;

				std::vector<float> ptr = Layers[i_layer-1]->get_var_parameters();
				//std::cout <<  i_layer << " " << Layers.size() << " ";

				if(ptr.size() != 0 ){
					//std::cout << (*ptr).size() << std::endl;

					std::copy((ptr).begin(),(ptr).end(), parameters_ptr);
					parameters_ptr +=(ptr).size();
					//std::cout << ptr->size() << " " << parameters_ptr << " " << std::endl;
					//params = * ptr;
					//(*ptr) = opts.optimizer->get_updated_parameters(*ptr, accumulated_gradient[i_layer-1]);
					//cblas_saxpy((*ptr).size(), -0.01/opts.batch_size, &accumulated_gradient[i_layer-1][0], 1.0, &((*ptr)[0]), 1.0);
					//Layers[i_layer-1]->set_var_parameters(params);
					//Layers[i_layer-1]->set_var_parameters(*ptr);
				}else{
					//std::cout << "0" << std::endl;
				}
				//std::cin.ignore();


			}
			//#pragma omp parallel for
            //int bigcount = 0;
			for( int i_grad = 0; i_grad < accumulated_gradient.size(); i_grad++){

				accumulated_gradient[i_grad] = accumulated_gradient[i_grad];///opts.batch_size;
			}
			//std::cout << bigcount << std::endl;
			//std::cin.ignore();*/
			parameter_vector = opts.optimizer->get_updated_parameters(parameter_vector,accumulated_gradient);

                

			int offset = 0;

			for( int i_layer = Layers.size(); i_layer > 0; i_layer--){

				std::copy(parameter_vector.begin() + offset, parameter_vector.begin()+offset +temp_parameter[i_layer-1].size(), temp_parameter[i_layer-1].begin() );
				//std::cout << temp_parameter[i_layer-1].size() << " " << offset +temp_parameter[i_layer-1].size() << " " << parameter_vector.size() << std::endl;
				//std::cin.ignore();
				offset += temp_parameter[i_layer-1].size();
               /* for( int j_test=0; j_test < temp_parameter[i_layer-1].size();j_test++){
                    //if(temp_parameter[i_layer-1][j_test] < 0)
                        std::cout << temp_parameter[i_layer-1][j_test] << " ";
                }
                std::cout<<std::endl <<i_layer <<  std::endl;
                std::cin.ignore();*/
				Layers[i_layer-1]->set_var_parameters(temp_parameter[i_layer-1]);
			}
           /* for( int j_test=0; j_test < parameter_vector.size();j_test++){
                    if(parameter_vector[j_test] < 0)
                        std::cout << parameter_vector[j_test] << " ";
            }
            std::cout << std::endl;
            std::cin.ignore();*/


			endt = clock();
			auto chrono_end_actuall = std::chrono::system_clock::now();
			std::chrono::duration<float> elapsed_seconds_backward_end = chrono_end_actuall-chrono_end;
			std::chrono::duration<float> elapsed_seconds_start_forward = chrono_end_forward-chrono_start;
			std::chrono::duration<float> elapsed_seconds_forward_backward = chrono_end-chrono_end_forward;
            std::chrono::duration<float> elapsed_seconds_start_end = chrono_end_actuall - chrono_start;
			//std::cout << std::endl;
			//std::cout << 1000*(elapsed_seconds_backward_end.count())/opts.batch_size << " " << 1000*(elapsed_seconds_forward_backward.count())/opts.batch_size << " " << 1000*(elapsed_seconds_start_forward.count())/opts.batch_size <<std::endl;
			i_data = i_data + opts.batch_size;
			//std::cout << "test1" << std::endl;
            running_time += elapsed_seconds_start_end.count();
            if( opts.log_loss){
                //logfile << opts.batch_size + i_data+Training_Data.size()*epochs<<"    "<< loss << std::endl;
                
                //#num epoch is_validation_set   i   running_actuall_num   loss   average_loss all_time single_time    epoch_time running_training_counter count/i*batch_size percent 
                auto chrono_alltime_stop = std::chrono::system_clock::now();
                std::chrono::duration<float> alltime_elapsed = chrono_alltime_stop - chrono_alltime_start;
                
                std::chrono::duration<float> epoch_ellapsed = chrono_alltime_stop - chrono_epoch_start;
                logfile << batch_count << " " << epochs << " " << 0 << " " << i_batch_count_epochs << " " << i_data+Training_Data.size()*epochs << " " << loss << " " << squared_error/batch_count << " " << alltime_elapsed.count()<< " " << elapsed_seconds_start_end.count() << " " << epoch_ellapsed.count() << " " << running_time << " " << epoch_acc_count << " " << (float)epoch_acc_count/(float)i_data<< std::endl;
            }
			//std::cout << "test3" << std::endl;
		}
		if(opts.use_validation_dataset){
            VectorNd<float> validation_result;
            double acc_loss= 0.0;
            int count = 0;
            for(int val_i = 0; val_i < Validation_Data.size(); val_i++){
                VectorNd<float> val_input(2,1, dims.inp_num);
                VectorNd<float> val_output(2,1, dims.out_num);
                // VectorNd<float> error_vec(2,1, dims.out_num);
			 
				std::copy( Validation_Data[val_i]->inp.begin(), Validation_Data[val_i]->inp.end(),& val_input(2,0,0));
				std::copy( Validation_Data[val_i]->out.begin(), Validation_Data[val_i]->out.end(),& val_output(2,0,0));
                validation_result = calc_output(val_input);
                //bool correct = ;
                /*int max_res = 0;
                int max_labels = 0;
                float max_val_res = validation_result.get_data()[0];
                float max_val_labels = val_output.get_data()[0];
                //std::cout << validation_result.get_data().size() << " " << val_output.get_data().size() << std::endl;
                for( int output_i = 0; output_i < validation_result.get_data().size(); output_i++){
                    //std::cout << validation_result.get_data()[output_i] << " " <<val_output.get_data()[output_i] << std::endl;
                    
                    if( validation_result.get_data()[output_i] > max_val_res ) {
                        max_res = output_i;
                        max_val_res = validation_result.get_data()[output_i];
                    }
                    if( val_output.get_data()[output_i] > max_val_labels){
                        max_labels = output_i;
                        max_val_labels = val_output.get_data()[output_i];
                    }
                    
                }
                //std::cout << std::endl;
                
                //std::cout << val_i << " " << max_res << " " << max_labels << std::endl;
                if ( max_res == max_labels ) {
                    count += 1;
                }*/
                double loss = 0.0;
                
                loss = opts.loss->get_loss(val_output,validation_result);
                acc_loss += loss;
                //std::cout << (acc_loss) << std::endl;
            }
            float accuracy = (float)count/(float)Validation_Data.size();
           
            validation_logfile<< epochs << " " << count << " " << accuracy << "  " << acc_loss/Validation_Data.size() << std::endl;
        }

	}
	if( opts.log_loss){
        logfile.close();
        if(opts.use_validation_dataset){
            validation_logfile.close();
        }
    }


}

/**
 * create Network with the given data dimensions
 */
Network::Network(std::vector<int> inp_size, std::vector<int> out_size ){
	set_data_dimensions(inp_size,out_size);

}

Network::~Network() {
	// TODO Auto-generated destructor stub
	for( int i = 0; i < Layers.size();i++){
		delete Layers[i];
	}
}

} /* namespace MRINN */
