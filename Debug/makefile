################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include src/Optimizers/subdir.mk
-include src/Losses/subdir.mk
-include src/Layers/subdir.mk
-include src/subdir.mk
-include subdir.mk
-include objects.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(C++_DEPS)),)
-include $(C++_DEPS)
endif
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
ifneq ($(strip $(CC_DEPS)),)
-include $(CC_DEPS)
endif
ifneq ($(strip $(CPP_DEPS)),)
-include $(CPP_DEPS)_backup
endif
ifneq ($(strip $(CXX_DEPS)),)
-include $(CXX_DEPS)
endif
ifneq ($(strip $(C_UPPER_DEPS)),)
-include $(C_UPPER_DEPS)
endif
endif

-include ../makefile.defs

# Add inputs and outputs from these tool invocations to the build variables 

# All Target
all: MRI_NeuralNetworkSuite

# Tool invocations
MRI_NeuralNetworkSuite: $(MAIN_OBJS) $(OBJS) $(USER_OBJS)
	@echo $(MAIN_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: Cross G++ Linker'
	nvcc -L/home/lennart/boost/lib -L/home/lennart/intel/mkl/lib/intel64_lin -L/usr/X11R6/lib -L/home/lennart/intel/mkl/include -o "MRI_NeuralNetworkSuite" $(MAIN_OBJS) $(OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '
	
C_API: $(OBJS) $(C_API_OBJS)
	g++ -L/home/lennart/boost/stage/lib -L/home/lennart/intel/mkl/lib/intel64_lin -L/usr/X11R6/lib -L/home/lennart/intel/mkl/include -c -I/home/lennart/intel/mkl/include -I/home/lennart/boost_1_69_0 -I/home/lennart/workspace/MRI_NeuralNetworkSuite/src -I/usr/local/cuda-9.0/include -O3 -Wall -o "C_API.o" "../src/C_API.cc" -lcuda -lcudnn
	#gcc -shared ./C_API.o -o libC_API.so
	ar rcs libmrinn.a $(OBJS) $(C_API_OBJS)  C_API.o
	
	
# Other Targets
MNIST_TEST: $(TEST_OBJS) $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: Cross G++ Linker'
	nvcc -L/home/lennart/boost/lib -L/home/lennart/intel/mkl/lib/intel64_lin -L/usr/X11R6/lib -L/home/lennart/intel/mkl/include -o "MNIST_TEST" $(TEST_OBJS) $(OBJS) $(USER_OBJS) $(LIBS) 
	@echo 'Finished building target: $@'
	@echo ' '
	
CONVERT_GPU_CPU: $(GPU_OBJS) $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: Cross G++ Linker'
	nvcc -L/home/lennart/boost/lib -L/home/lennart/intel/mkl/lib/intel64_lin -L/usr/X11R6/lib -L/home/lennart/intel/mkl/include -o "CONVERT_GPU" $(GPU_OBJS) $(OBJS) $(USER_OBJS) $(LIBS) 
	@echo 'Finished building target: $@'
	@echo ' '
clean:
	-$(RM) $(C++_DEPS)$(OBJS)$(C_DEPS)$(CC_DEPS)$(CPP_DEPS)$(EXECUTABLES)$(CXX_DEPS)$(C_UPPER_DEPS) MRI_NeuralNetworkSuite MNIST_TEST
	-@echo ' '

.PHONY: all clean dependents
.SECONDARY:

-include ../makefile.targets
