################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/Losses/BinaryCrossentropy.cpp \
../src/Losses/CrossEntropy.cpp \
../src/Losses/Loss.cpp \
../src/Losses/MSE.cpp \
../src/Losses/sparse_softmax_crossentropy.cpp

OBJS += \
./src/Losses/BinaryCrossentropy.o \
./src/Losses/CrossEntropy.o \
./src/Losses/Loss.o \
./src/Losses/MSE.o \
./src/Losses/sparse_softmax_crossentropy.o

CPP_DEPS += \
./src/Losses/BinaryCrossentropy.d \
./src/Losses/CrossEntropy.d \
./src/Losses/Loss.d \
./src/Losses/MSE.d \
./src/Losses/sparse_softmax_crossentropy.d

#-fopenmp
# Each subdirectory must supply rules for building sources it contributes
src/Losses/%.o: ../src/Losses/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	g++ -std=c++0x -I/home/lennart/intel/mkl/include -I/home/lennart/boost_1_69_0 -I/home/lennart/workspace/MRI_NeuralNetworkSuite/src -O3 -g3 -Wall -c -fmessage-length=0 -fstack-protector-all -fno-builtin -funroll-loops -msse4.2 -mavx -MMD -fopenmp -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


