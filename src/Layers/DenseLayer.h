/*
 * DenseLayer.h
 *
 *  Created on: Nov 20, 2017
 *      Author: lennart
 */

#ifndef DENSELAYER_H_
#define DENSELAYER_H_

#include "Layer.h"
#include <vector>
#include "../VectorNd.h"
//#include <boost/serialization/export.hpp>
//#include <boost/archive/binary_iarchive.hpp>
//#include <boost/archive/binary_oarchive.hpp>

namespace MRINN{
class DenseLayer;
}/* end namespace MRINN*/


namespace boost { namespace serialization {
template<class Archive>
inline void load_construct_data( Archive & ar, MRINN::DenseLayer * t, const unsigned int file_version);
template<class Archive>
inline void save_construct_data(Archive & ar, const MRINN::DenseLayer * t, const unsigned int file_version);
}} /*end namespace boost::serialization*/

namespace MRINN {

class DenseLayer: public Layer {
public:

	friend class boost::serialization::access;
	template<class Archive> friend void boost::serialization::save_construct_data(Archive & ar, const MRINN::DenseLayer * t, const unsigned int file_version);
	template<class Archive> friend void boost::serialization::load_construct_data(Archive & ar, MRINN::DenseLayer * t, const unsigned int file_version);

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
    	//ar & boost::serialization::base_object<Layer>(*this);
    	ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Layer);
    	ar & use_bias;
    	ar & weight_matrix;
    }

	void calculate_jacobian();
	void set_var_parameters(std::vector<float> & pars);
	std::vector<float> & get_var_parameters();

	VectorNd<float> & get_parameter_gradient(VectorNd<float> & backprop_vec);
	DenseLayer( std::vector<int> inp_size, std::vector<int> out_size, bool use_bias = true);

	void print_weights();

	virtual ~DenseLayer();

protected:
	//std::vector<std::vector<float> > weights;
	//std::vector<float> weight_matrix;
	bool use_bias;
	VectorNd<float> weight_matrix;



	//vector<float> parameters;
	//std::vector<float> bias;
	void process();
	void backpropagate();

};

} /* namespace MRINN */

/*
 * Overload the load operator from boost::serialization to execute the non emtpy constructor of this Layer
 */
namespace boost { namespace serialization {

template<class Archive>
inline void save_construct_data(
    Archive & ar, const MRINN::DenseLayer * t, const unsigned int file_version
){
    // save data required to construct instance
    ar << t->inp_size;
    ar << t->out_size;
    ar << t->use_bias;
}
template<class Archive>
inline void load_construct_data(
    Archive & ar, MRINN::DenseLayer * t, const unsigned int file_version
){
    // retrieve data from archive required to construct new instance
	std::vector<int> inp_size, out_size;
    bool use_bias;
    ar >> inp_size;
    ar >> out_size;
    ar >> use_bias;
    // invoke inplace constructor to initialize instance of my_class
    ::new(t)MRINN::DenseLayer( inp_size, out_size, use_bias);
}
}}

#endif /* DENSELAYER_H_ */
