################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/Optimizers/AdaDelta.cpp \
../src/Optimizers/AdaGrad.cpp \
../src/Optimizers/AdaMax.cpp \
../src/Optimizers/Adam.cpp \
../src/Optimizers/Optimizer.cpp \
../src/Optimizers/SGD.cpp 

OBJS += \
./src/Optimizers/AdaDelta.o \
./src/Optimizers/AdaGrad.o \
./src/Optimizers/AdaMax.o \
./src/Optimizers/Adam.o \
./src/Optimizers/Optimizer.o \
./src/Optimizers/SGD.o 




CPP_DEPS += \
./src/Optimizers/AdaDelta.d \
./src/Optimizers/AdaGrad.d \
./src/Optimizers/AdaMax.d \
./src/Optimizers/Adam.d \
./src/Optimizers/Optimizer.d \
./src/Optimizers/SGD.d 

#-fopenmp
# Each subdirectory must supply rules for building sources it contributes
src/Optimizers/%.o: ../src/Optimizers/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	g++ -std=c++0x -I/home/lennart/intel/mkl/include -I/home/lennart/Documents/boostlb/boost_1_68_0 -I/home/lennart/workspace/MRI_NeuralNetworkSuite/src -O3 -g3 -Wall -c -fmessage-length=0 -fstack-protector-all -fno-builtin -funroll-loops -msse4.2 -fopenmp -mavx -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


