#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/export.hpp>
#include <iostream>
#include "Network.h"
#include "Layers/Layer.h"

int main(){
    Network * net_loaded = new Network();
	std::ifstream ifsload1("saved_net_images_knees_CUDA_TEST_1_0.13.dat");
	boost::archive::binary_iarchive ia1(ifsload1);
	ia1 >> net;
	std::cout << "net loaded" << std::endl;
	ifsload1.close();
    
    for( int i = 0; i < net.Layers.size();i++){
        std::cout << typeid(*net.Layers[i]).name() << std::endl;
    }
    
    delete net;


}
