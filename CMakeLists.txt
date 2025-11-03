cmake_minimum_required(VERSION 3.18)
project(MRI_NeuralNetworkSuite LANGUAGES CXX CUDA)

# ==============================
#   Build Settings
# ==============================
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Release)

# ==============================
#   Source Files
# ==============================
file(GLOB_RECURSE SRC_CPP CONFIGURE_DEPENDS
    src/*.cpp
)
file(GLOB_RECURSE SRC_CU CONFIGURE_DEPENDS
    src/*.cu
)

# ==============================
#   Targets
# ==============================
add_executable(MRI_NeuralNetworkSuite ${SRC_CPP} ${SRC_CU})

# ==============================
#   Include Directories
# ==============================
target_include_directories(MRI_NeuralNetworkSuite PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    /usr/include
    /usr/local/include
    /usr/include/boost
)

include_directories(/home/lennart/Libraries/mklml_lnx_2018.0.3.20180406/include)
link_directories(/home/lennart/Libraries/mklml_lnx_2018.0.3.20180406/lib)



# ==============================
#   CUDA Setup (optional)
# ==============================
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION}")
    target_link_libraries(MRI_NeuralNetworkSuite PRIVATE CUDA::cudart)
    target_compile_definitions(MRI_NeuralNetworkSuite PRIVATE USE_CUDA)
else()
    message(WARNING "CUDA Toolkit not found — building without GPU support.")
endif()

# ==============================
#   cuDNN (optional)
# ==============================
find_path(CUDNN_INCLUDE_DIR cudnn.h PATHS /usr/include /usr/local/cuda/include)
find_library(CUDNN_LIBRARY cudnn PATHS /usr/lib /usr/local/cuda/lib64)

if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
    message(STATUS "Found cuDNN: ${CUDNN_LIBRARY}")
    target_include_directories(MRI_NeuralNetworkSuite PRIVATE ${CUDNN_INCLUDE_DIR})
    target_link_libraries(MRI_NeuralNetworkSuite PRIVATE ${CUDNN_LIBRARY})
    target_compile_definitions(MRI_NeuralNetworkSuite PRIVATE USE_CUDNN)
else()
    message(WARNING "cuDNN not found — continuing without it.")
endif()

# ==============================
#   BLAS / MKL Detection
# ==============================
if(DEFINED ENV{MKLROOT})
    set(MKLROOT $ENV{MKLROOT})
else()
    set(MKLROOT "/opt/intel/oneapi/mkl/latest")
endif()

if(NOT EXISTS "${MKLROOT}/include/mkl.h")
    message(FATAL_ERROR "MKL not found at ${MKLROOT}. Please install intel-oneapi-mkl and/or source setvars.sh.")
endif()

include_directories(${MKLROOT}/include)
link_directories(${MKLROOT}/lib/intel64)

find_path(OPENBLAS_INCLUDE_DIR cblas.h
    HINTS /usr/include /usr/include/openblas
)
find_library(OPENBLAS_LIB openblas)
target_include_directories(MRI_NeuralNetworkSuite PRIVATE ${OPENBLAS_INCLUDE_DIR})

# ==============================
#   Boost (serialization)
# ==============================
find_package(Boost REQUIRED COMPONENTS serialization)
target_include_directories(MRI_NeuralNetworkSuite PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(MRI_NeuralNetworkSuite PRIVATE ${Boost_LIBRARIES})

# ==============================
#  OpenMP
# ==============================
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
  target_link_libraries(MRI_NeuralNetworkSuite PUBLIC OpenMP::OpenMP_CXX)
endif()



# ==============================
#   Other System Libraries
# ==============================
find_package(X11 REQUIRED)
target_link_libraries(MRI_NeuralNetworkSuite
    PRIVATE
        ${Boost_LIBRARIES}
        ${X11_LIBRARIES}
        mkldnn 
        mkl_intel_lp64
        mkl_intel_thread
        mkl_core
        iomp5
        pthread
        dl
        m
        cuda
        cudart
        openblas
        ${OPENBLAS_LIB}
)
# ==============================
#   Compiler Flags
# ==============================
target_compile_options(MRI_NeuralNetworkSuite PRIVATE
  -O3 -fopenmp -funroll-loops -msse4.2 -mavx -fstack-protector-all -Wno-sign-compare -Wno-unused-variable -Wno-dev -Wno-char-subscripts
)
