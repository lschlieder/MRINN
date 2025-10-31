/*
 * Network.h
 *
 *  Created on: Nov 15, 2017
 *      Author: lennart
 */

#ifndef NETWORK_H_
#define NETWORK_H_

#include "Layers/Layer.h"
#include "Data.h"
#include "Optimizers/Optimizer.h"
#include "Losses/Loss.h"
#include <iostream>

#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>

namespace MRINN {




struct data_dimensions{
	friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
    	ar & inp_num;
    	ar & out_num;
    	ar & inp_dim;
    	ar & out_dim;
    	ar & inp_size;
    	ar & out_size;

    }
	int inp_num;
	int out_num;
	int inp_dim;
	int out_dim;
	std::vector<int> inp_size;
	std::vector<int> out_size;
};

/*struct training_algorithm{
	algorithm t_algo;
	step_size alpha;
	momentum




};*/

struct training_opts{
	Optimizers::Optimizer * optimizer;
	Losses::Loss * loss;
	int batch_size;
	int epochs;
    bool log_loss;
    std::string log_filename = "logfile.txt"; 
    std::string validation_log_filename = "validation_logfile.txt";
    bool use_validation_dataset = false;
};

class Network {
public:
	friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
    	ar & Layers;
    	ar & dims;
    	ar & parameters;

    }
	void set_data_dimensions( std::vector<int> inp_size, std::vector<int> out_size );
	void set_data_dimensions(data_dimensions dims);

	void print_layers();

	void set_data(std::vector<Data*> data);
    void set_validation_data( std::vector<Data*> data);
    
	data_dimensions get_data_dimensions();

	void add_layer(Layer * layer);
	void train_network(training_opts opts);
	VectorNd<float> & calc_output(VectorNd<float> & inp);
	VectorNd<float> calc_and_write_output(VectorNd<float> inp);
	VectorNd<float> & calc_error(VectorNd<float> & error_inp, std::vector<float> & acccumulated_gradient);
    
    void set_logfile(std::string filename);

	Network();
	Network(std::vector<int> inp_size, std::vector<int> out_size );


	virtual ~Network();
	std::vector<Layer*> Layers;
protected:

	std::vector<Data*> Training_Data;
    std::vector<Data*> Validation_Data;
    

	//std::vector<std::vector<float> > accumulated_error;

	data_dimensions dims;

	int parameters;

};

} /* namespace MRINN */

#endif /* NETWORK_H_ */
