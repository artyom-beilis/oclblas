cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -O2")

find_path(OCL_PATH CL/cl.hpp)
find_library(OCL_LIB OpenCL) 

### My build always
if(OCL_PATH AND OCL_LIB)
	add_executable(test_gemm_my test.cpp sgemm_cpu.cpp sgemm_my.cpp)
	target_include_directories(test_gemm_my PUBLIC  ${OCL_PATH})
	target_link_libraries(test_gemm_my ${OCL_LIB} openblas)
    target_compile_definitions(test_gemm_my PUBLIC "-DMYKERNEL_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/\"")
	find_library(CUDART_LIB nvrtc HINTS ${CUDA_PATH}/lib ${CUDA_PATH}/lib64)
    
endif()

if(CUDA_PATH)
	find_path(CUDA_HDR cublas.h HINTS ${CUDA_PATH}/include)
	find_library(CUBLAS_LIB cublas HINTS ${CUDA_PATH}/lib ${CUDA_PATH}/lib64)
	find_library(CUDA_LIB cudart HINTS ${CUDA_PATH}/lib ${CUDA_PATH}/lib64)


	add_executable(test_gemm_cublas test.cpp sgemm_cpu.cpp sgemm_cublas.cpp)
	target_include_directories(test_gemm_cublas PUBLIC ${CUDA_HDR})
	target_link_libraries(test_gemm_cublas ${CUBLAS_LIB} ${CUDA_LIB} openblas)
	
    find_path(CUDARTC_HDR nvrtc.h HINTS ${CUDA_PATH}/include)
	find_library(CUDART_LIB nvrtc HINTS ${CUDA_PATH}/lib ${CUDA_PATH}/lib64)
	find_library(CUDA_BASE_LIB cuda HINTS ${CUDA_PATH}/lib ${CUDA_PATH}/lib64)
	
    add_executable(test_gemm_mycuda test.cpp sgemm_cpu.cpp sgemm_mycuda.cpp)
	target_include_directories(test_gemm_mycuda PUBLIC ${CUDA_HDR} ${CUDARTC_HDR})
	target_link_libraries(test_gemm_mycuda  ${CUDA_LIB} ${CUDART_LIB} ${CUDA_BASE_LIB} openblas)
    target_compile_definitions(test_gemm_mycuda PUBLIC "-DMYKERNEL_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/\"")

endif()



if(CLBLAS_PATH)
	find_path(CLBLAS_HDR clBLAS.h HINTS ${CLBLAS_PATH}/include)
	find_library(CLBLAS_LIB clBLAS HINTS ${CLBLAS_PATH}/lib ${CLBLAS_PATH}/lib64)
	add_executable(test_gemm_clblas test.cpp sgemm_cpu.cpp sgemm_clblas.cpp)
	target_include_directories(test_gemm_clblas PUBLIC ${CLBLAS_HDR} ${OCL_PATH})
	target_link_libraries(test_gemm_clblas ${CLBLAS_LIB} ${OCL_LIB} openblas)
endif()

if(CLBLAST_PATH)
	find_path(CLBLAST_HDR clblast.h HINTS ${CLBLAST_PATH}/include)
	find_library(CLBLAST_LIB clblast HINTS ${CLBLAST_PATH}/lib ${CLBLAST_PATH}/lib64)
	add_executable(test_gemm_clblast test.cpp sgemm_cpu.cpp sgemm_clblast.cpp)
	target_include_directories(test_gemm_clblast PUBLIC ${CLBLAST_HDR} ${OCL_PATH})
	target_link_libraries(test_gemm_clblast ${CLBLAST_LIB} ${OCL_LIB} openblas)
endif()


if(MIOPENGEMM_PATH)
	find_path(MIOPENGEMM_HDR miopengemm/gemm.hpp HINTS ${MIOPENGEMM_PATH}/include)
	find_library(MIOPENGEMM_LIB miopengemm HINTS ${MIOPENGEMM_PATH}/lib)
	add_executable(test_gemm_miopengemm test.cpp sgemm_cpu.cpp sgemm_miopengemm.cpp)
	target_include_directories(test_gemm_miopengemm PUBLIC ${MIOPENGEMM_HDR} ${OCL_PATH})
	target_link_libraries(test_gemm_miopengemm ${MIOPENGEMM_LIB} ${OCL_LIB} openblas)
endif()


