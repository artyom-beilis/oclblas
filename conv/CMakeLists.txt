cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -O2")

find_path(OCL_PATH CL/cl.hpp)
find_library(OCL_LIB OpenCL) 

### My build always
#if(OCL_PATH AND OCL_LIB)
#	add_executable(test_gemm_my test.cpp sgemm_cpu.cpp sgemm_my.cpp)
#	target_include_directories(test_gemm_my PUBLIC  ${OCL_PATH})
#	target_link_libraries(test_gemm_my ${OCL_LIB} openblas)
#endif()

if(CUDA_PATH AND CUDNN_PATH)
	find_path(CUDA_HDR cuda_runtime.h HINTS ${CUDA_PATH}/include)
	find_path(CUDNN_HDR cudnn.h HINTS ${CUDNN_PATH}/include)

	find_library(CUDA_LIB cudart HINTS ${CUDA_PATH}/lib ${CUDA_PATH}/lib64)
	find_library(CUDNN_LIB cudnn HINTS ${CUDNN_PATH}/lib ${CUDNN_PATH}/lib64)
	add_executable(test_conv_cudnn test_conv.cpp conv_ref.cpp conv_cudnn.cpp)
	target_include_directories(test_conv_cudnn PUBLIC ${CUDA_HDR} ${CUDNN_HDR})
	target_link_libraries(test_conv_cudnn ${CUDNN_LIB} ${CUDA_LIB} openblas)
endif()


