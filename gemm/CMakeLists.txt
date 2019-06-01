cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -O2")

find_path(OCL_PATH CL/cl.hpp)
find_library(OCL_LIB OpenCL) 

### My build always
if(OCL_PATH AND OCL_LIB)
	add_executable(test_gemm_my test.cpp sgemm_cpu.cpp sgemm_my.cpp)
	target_include_directories(test_gemm_my PUBLIC  ${OCL_PATH})
	target_link_libraries(test_gemm_my ${OCL_LIB} openblas)
endif()

if(CUDA_PATH)
	find_path(CUDA_HDR cublas.h HINTS ${CUDA_PATH}/include)
	find_library(CUBLAS_LIB cublas HINTS ${CUDA_PATH}/lib ${CUDA_PATH}/lib64)
	find_library(CUDA_LIB cudart HINTS ${CUDA_PATH}/lib ${CUDA_PATH}/lib64)
	add_executable(test_gemm_cublas test.cpp sgemm_cpu.cpp sgemm_cublas.cpp)
	target_include_directories(test_gemm_cublas PUBLIC ${CUDA_HDR})
	target_link_libraries(test_gemm_cublas ${CUBLAS_LIB} ${CUDA_LIB} openblas)
endif()



if(CLBLAS_PATH)
	find_path(CLBLAS_HDR clBLAS.h HINTS ${CLBLAS_PATH}/include)
	find_library(CLBLAS_LIB clBLAS HINTS ${CLBLAS_PATH}/lib)
	add_executable(test_gemm_clblas test.cpp sgemm_cpu.cpp sgemm_clblas.cpp)
	target_include_directories(test_gemm_clblas PUBLIC ${CLBLAS_HDR} ${OCL_PATH})
	target_link_libraries(test_gemm_clblas ${CLBLAS_LIB} ${OCL_LIB} openblas)
endif()

if(MIOPENGEMM_PATH)
	find_path(MIOPENGEMM_HDR miopengemm/gemm.hpp HINTS ${MIOPENGEMM_PATH}/include)
	find_library(MIOPENGEMM_LIB miopengemm HINTS ${MIOPENGEMM_PATH}/lib)
	add_executable(test_gemm_miopengemm test.cpp sgemm_cpu.cpp sgemm_miopengemm.cpp)
	target_include_directories(test_gemm_miopengemm PUBLIC ${MIOPENGEMM_HDR} ${OCL_PATH})
	target_link_libraries(test_gemm_miopengemm ${MIOPENGEMM_LIB} ${OCL_LIB} openblas)
endif()


