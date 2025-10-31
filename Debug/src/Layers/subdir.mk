################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/Layers/BatchNormalizationLayer.cpp \
../src/Layers/Convolution2DLayer.cpp \
../src/Layers/ConvolutionalLayer.cpp \
../src/Layers/DenseLayer.cpp \
../src/Layers/DropoutLayer.cpp \
../src/Layers/ExpLayer.cpp \
../src/Layers/Layer.cpp \
../src/Layers/MaxPoolingLayer.cpp \
../src/Layers/ReluLayer.cpp \
../src/Layers/ScewedDistanceLayer.cpp \
../src/Layers/SelfAssemblingSparseLayer.cpp \
../src/Layers/SoftmaxLayer.cpp \
../src/Layers/TanhLayer.cpp \
../src/Layers/UpSampling2D.cpp \
../src/Layers/convolution2dcudnn.cpp

OBJS += \
./src/Layers/BatchNormalizationLayer.o \
./src/Layers/Convolution2DLayer.o \
./src/Layers/ConvolutionalLayer.o \
./src/Layers/DenseLayer.o \
./src/Layers/DropoutLayer.o \
./src/Layers/ExpLayer.o \
./src/Layers/Layer.o \
./src/Layers/MaxPoolingLayer.o \
./src/Layers/ReluLayer.o \
./src/Layers/ScewedDistanceLayer.o \
./src/Layers/SelfAssemblingSparseLayer.o \
./src/Layers/SoftmaxLayer.o \
./src/Layers/TanhLayer.o \
./src/Layers/UpSampling2D.o \
./src/Layers/convolution2dcudnn.o

CPP_DEPS += \
./src/Layers/BatchNormalizationLayer.d \
./src/Layers/Convolution2DLayer.d \
./src/Layers/ConvolutionalLayer.d \
./src/Layers/DenseLayer.d \
./src/Layers/DropoutLayer.d \
./src/Layers/ExpLayer.d \
./src/Layers/Layer.d \
./src/Layers/MaxPoolingLayer.d \
./src/Layers/ReluLayer.d \
./src/Layers/ScewedDistanceLayer.d \
./src/Layers/SelfAssemblingSparseLayer.d \
./src/Layers/SoftmaxLayer.d \
./src/Layers/TanhLayer.d \
./src/Layers/UpSampling2D.d \
./src/Layers/convolution2dcudnn.d

#-fopenmp
# Each subdirectory must supply rules for building sources it contributes
src/Layers/%.o: ../src/Layers/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	g++ -std=c++0x -I/home/lennart/intel/mkl/include -I/home/lennart/boost_1_69_0 -I/home/lennart/workspace/MRI_NeuralNetworkSuite/src -I/usr/local/cuda-9.0/include -I/home/lennart/projects/cudnn_test/cuda/include -O3 -g3 -Wall -c -fmessage-length=0 -fstack-protector-all -fno-builtin -fopenmp -funroll-loops -msse4.2 -mavx -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


