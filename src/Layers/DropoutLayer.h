/*
 * DropoutLayer.h
 *
 *  Created on: May 9, 2018
 *      Author: lennart
 */

#ifndef DROPOUTLAYER_H_
#define DROPOUTLAYER_H_

#include "Layer.h"
#include <map>
//#include <boost/serialization/access.hpp>

namespace MRINN{
class DropoutLayer;
}/* end namespace MRINN*/


namespace boost { namespace serialization {
template<class Archive>
inline void load_construct_data( Archive & ar, MRINN::DropoutLayer * t, const unsigned int file_version);
template<class Archive>
inline void save_construct_data(Archive & ar, const MRINN::DropoutLayer* t, const unsigned int file_version);
}} /*end namespace boost::serialization*/

namespace MRINN{

class DropoutLayer: public Layer {
public:
	/* befriend stuff for serialization*/
	friend class boost::serialization::access;

	template<class Archive> friend void boost::serialization::save_construct_data(Archive & ar, const MRINN::DropoutLayer* t, const unsigned int file_version);
	template<class Archive> friend void boost::serialization::load_construct_data(Archive & ar, MRINN::DropoutLayer * t, const unsigned int file_version);


    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
    	//ar & boost::serialization::base_object<Layer>(*this);
    	ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Layer);
    	ar & rate;
    	ar & connection_map;
    }
	DropoutLayer(std::vector<int> inp_size, std::vector<int> out_size, float rate);
	virtual ~DropoutLayer();

protected:
	float rate;
	std::vector< int > connection_map;
	void process();
	void backpropagate();
};

}/* end namespace MRINN*/

/*
 * Overload the load operator from boost::serialization to execute the non emtpy constructor of this Layer
 */
namespace boost { namespace serialization {

template<class Archive>
inline void save_construct_data(
    Archive & ar, const MRINN::DropoutLayer * t, const unsigned int file_version
){
    // save data required to construct instance
    ar << t->inp_size;
    ar << t->out_size;
    ar << t->rate;
}
template<class Archive>
inline void load_construct_data(
    Archive & ar, MRINN::DropoutLayer * t, const unsigned int file_version
){
    // retrieve data from archive required to construct new instance
	std::vector<int> inp_size, out_size;
	float rate;
	ar >> inp_size;
	ar >> out_size;
	ar >> rate;
    // invoke inplace constructor to initialize instance of my_class
    ::new(t)MRINN::DropoutLayer(inp_size,out_size,rate);
}
}}


#endif /* DROPOUTLAYER_H_ */
