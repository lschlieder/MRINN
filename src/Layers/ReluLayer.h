/*
 * ReluLayer.h
 *
 *  Created on: Nov 20, 2017
 *      Author: lennart
 */

#ifndef RELULAYER_H_
#define RELULAYER_H_

#include "Layer.h"

namespace MRINN{
class ReluLayer;
}/* end namespace MRINN*/


namespace boost { namespace serialization {
template<class Archive>
inline void load_construct_data(
    Archive & ar, MRINN::ReluLayer * t, const unsigned int file_version);
template<class Archive>
inline void save_construct_data(Archive & ar, const MRINN::ReluLayer* t, const unsigned int file_version);
}} /*end namespace boost::serialization*/

namespace MRINN {

class ReluLayer: public Layer {
public:
	friend class boost::serialization::access;

	template<class Archive> friend void boost::serialization::save_construct_data(Archive & ar, const MRINN::ReluLayer* t, const unsigned int file_version);
	template<class Archive> friend void boost::serialization::load_construct_data(Archive & ar, MRINN::ReluLayer * t, const unsigned int file_version);


    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
    	//ar & boost::serialization::base_object<Layer>(*this);
    	ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Layer);
    }
	//void calculate_jacobian();
	ReluLayer( std::vector<int> inp_size, std::vector<int> out_size);
	virtual ~ReluLayer();

protected:
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
    Archive & ar, const MRINN::ReluLayer * t, const unsigned int file_version
){
    // save data required to construct instance
    ar << t->inp_size;
    ar << t->out_size;
}
template<class Archive>
inline void load_construct_data(
    Archive & ar, MRINN::ReluLayer * t, const unsigned int file_version
){
    // retrieve data from archive required to construct new instance
	std::vector<int> inp_size, out_size;
	ar >> inp_size;
	ar >> out_size;
    // invoke inplace constructor to initialize instance of my_class
    ::new(t)MRINN::ReluLayer(inp_size,out_size);
}
}}

#endif /* RELULAYER_H_ */
