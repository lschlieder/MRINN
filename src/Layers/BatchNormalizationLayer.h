/*
 * BatchNormalizationLayer.h
 *
 *  Created on: May 14, 2018
 *      Author: lennart
 */

#ifndef BATCHNORMALIZATIONLAYER_H_
#define BATCHNORMALIZATIONLAYER_H_

#include "Layer.h"
#include "mkl_dnn.h"

namespace MRINN{
class BatchNormalizationLayer;
}/* end namespace MRINN*/


namespace boost { namespace serialization {
template<class Archive>
inline void load_construct_data( Archive & ar, MRINN::BatchNormalizationLayer * t, const unsigned int file_version);
template<class Archive>
inline void save_construct_data(Archive & ar, const MRINN::BatchNormalizationLayer* t, const unsigned int file_version);

}} /*end namespace boost::serialization*/

namespace MRINN {

class BatchNormalizationLayer: public Layer {
public:
	friend class boost::serialization::access;

	template<class Archive> friend void boost::serialization::save_construct_data(Archive & ar, const MRINN::BatchNormalizationLayer* t, const unsigned int file_version);
	template<class Archive> friend void boost::serialization::load_construct_data(Archive & ar, MRINN::BatchNormalizationLayer * t, const unsigned int file_version);

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
    	//ar & boost::serialization::base_object<Layer>(*this);
    	ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Layer);
    	ar & batch_size;
    	ar & params;
    	ar & workspace;
    	ar & eps;
    }

	void set_var_parameters(std::vector<float> &pars);
	std::vector<float> & get_var_parameters();
	VectorNd<float> & get_parameter_gradient(VectorNd<float> & backprop_vec);


	BatchNormalizationLayer(std::vector<int> inp_size, std::vector<int> out_size, float eps);
	virtual ~BatchNormalizationLayer();

private:
	int batch_size;
	float* resBatchNormalization[dnnResourceNumber] = {0};
	dnnPrimitive_t pBatchNormalization;
	dnnPrimitive_t pBatchNormalizationData;
	dnnPrimitive_t pBatchNormalizationScaleShift;
	dnnPrimitiveAttributes_t attributes;
	dnnLayout_t data_layout;

	std::vector<float> workspace;
	std::vector<float> params;

	void create_normalization_primitive(int batch_number);
	void process();
	void backpropagate();

	float eps;
};

} /* namespace MRINN */

/*
 * Overload the load operator from boost::serialization to execute the non emtpy constructor of this Layer
 */
namespace boost { namespace serialization {

template<class Archive>
inline void save_construct_data(
    Archive & ar, const MRINN::BatchNormalizationLayer * t, const unsigned int file_version){
    // save data required to construct instance
    ar << t->inp_size;
    ar << t->out_size;
    ar << t->eps;

}
template<class Archive>
inline void load_construct_data(
    Archive & ar, MRINN::BatchNormalizationLayer * t, const unsigned int file_version
){
    // retrieve data from archive required to construct new instance
	std::vector<int> inp_size, out_size;
    float eps;
   // dnnBorder_t border_method;
    ar >> inp_size;
    ar >> out_size;
    ar >> eps;


    // invoke inplace constructor to initialize instance of my_class
    ::new(t)MRINN::BatchNormalizationLayer( inp_size, out_size,eps);
}
}}

#endif /* BATCHNORMALIZATIONLAYER_H_ */
