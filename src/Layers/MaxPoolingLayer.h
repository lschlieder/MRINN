/*
 * MaxPoolingLayer.h
 *
 *  Created on: Jan 23, 2018
 *      Author: lennart
 */

#ifndef MAXPOOLINGLAYER_H_
#define MAXPOOLINGLAYER_H_

#include "Layer.h"
#include <map>


namespace MRINN{
class MaxPoolingLayer;
}/* end namespace MRINN*/


namespace boost { namespace serialization {
template<class Archive>
inline void load_construct_data( Archive & ar, MRINN::MaxPoolingLayer * t, const unsigned int file_version);
template<class Archive>
inline void save_construct_data(Archive & ar, const MRINN::MaxPoolingLayer* t, const unsigned int file_version);
}} /*end namespace boost::serialization*/

namespace MRINN {

class MaxPoolingLayer: public Layer {
public:

	/* befriend stuff for serialization*/
	friend class boost::serialization::access;

	template<class Archive> friend void boost::serialization::save_construct_data(Archive & ar, const MRINN::MaxPoolingLayer* t, const unsigned int file_version);
	template<class Archive> friend void boost::serialization::load_construct_data(Archive & ar, MRINN::MaxPoolingLayer * t, const unsigned int file_version);


    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
    	//ar & boost::serialization::base_object<Layer>(*this);
    	ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Layer);
    	ar & pooling_size_x;
    	ar & pooling_size_y;
    	ar & map;
    }
	MaxPoolingLayer(std::vector<int> inp_size, std::vector<int> out_size, int pooling_size_x, int pooling_size_y);
	virtual ~MaxPoolingLayer();

protected:
	int pooling_size_x;
	int pooling_size_y;
	std::vector< std::map<int,int> > map;
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
    Archive & ar, const MRINN::MaxPoolingLayer * t, const unsigned int file_version
){
    // save data required to construct instance
    ar << t->inp_size;
    ar << t->out_size;
    ar << t->pooling_size_x;
    ar << t->pooling_size_y;
}
template<class Archive>
inline void load_construct_data(
    Archive & ar, MRINN::MaxPoolingLayer * t, const unsigned int file_version
){
    // retrieve data from archive required to construct new instance
	std::vector<int> inp_size, out_size;
	int pooling_size_x, pooling_size_y;
	ar >> inp_size;
	ar >> out_size;
	ar >> pooling_size_x;
	ar >> pooling_size_y;
    // invoke inplace constructor to initialize instance of my_class
    ::new(t)MRINN::MaxPoolingLayer(inp_size,out_size,pooling_size_x, pooling_size_y);
}
}}

#endif /* MAXPOOLINGLAYER_H_ */
