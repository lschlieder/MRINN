################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/Helper_Functions.cpp \
../src/MRI_NeuralNetworkSuite.cpp \
../src/Network.cpp \
../src/MNIST_TEST.cpp \
../src/Convert_cuda_to_cpu.cpp

MAIN_OBJS += \
./src/Helper_Functions.o \
./src/MRI_NeuralNetworkSuite.o \
./src/Network.o \
./src/CudaConvolution.o

TEST_OBJS += \
./src/Helper_Functions.o \
./src/Network.o \
./src/CudaConvolution.o \
./src/MNIST_TEST.o

GPU_OBJS += \
./src/Helper_Functions.o \
./src/Network.o \
./src/CudaConvolution.o \
./src/Convert_cuda_to_cpu.o

C_API_OBJS += \
./src/Helper_Functions.o \
./src/Network.o \
./src/CudaConvolution.o \

CPP_DEPS += \
./src/Helper_Functions.d \
./src/MRI_NeuralNetworkSuite.d \
./src/Network.d \

CU_SRCS += \
../src/CudaConvolution.cu

#-fopenmp


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	g++ -std=c++0x -I/home/lennart/intel/mkl/include -I/home/lennart/boost_1_69_0 -I/home/lennart/workspace/MRI_NeuralNetworkSuite/src -I/usr/local/cuda-9.0/include -I/home/lennart/projects/cudnn_test/cuda/include -O3 -g3 -Wall -c -fmessage-length=0 -fstack-protector-all -fopenmp -fno-builtin -funroll-loops -msse4.2 -mavx -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '
	
src/%.o: ../src/%.cu
	@echo 'Building CUDA file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	nvcc -I/home/lennart/intel/mkl/include -I/home/lennart/boost_1_69_0 -I/home/lennart/workspace/MRI_NeuralNetworkSuite/src -I/usr/local/cuda-9.0/include -I/home/lennart/projects/cudnn_test/cuda/include -O3 -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '
