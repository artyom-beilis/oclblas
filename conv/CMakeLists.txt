cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -O2")

find_path(OCL_PATH CL/cl.hpp)
find_library(OCL_LIB OpenCL) 

find_path(BOOST_INC boost/shared_ptr.hpp) 

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

if(CAFFE_CUDA_PATH AND CUDA_PATH)
	find_path(CAFFE_CUDA_HDR caffe/net.hpp HINTS ${CAFFE_CUDA_PATH}/include)
	find_library(CAFFE_CUDA_LIB caffe HINTS ${CAFFE_CUDA_PATH}/lib)
	add_executable(test_conv_caffe_cuda test_conv.cpp conv_ref.cpp conv_caffe.cpp)
	target_include_directories(test_conv_caffe_cuda PUBLIC ${CAFFE_CUDA_HDR} ${CUDA_HDR} ${BOOST_INC} )
	target_link_libraries(test_conv_caffe_cuda ${CAFFE_CUDA_LIB} openblas)
endif()

if(CAFFE_OCL_PATH AND OCL_PATH)
	find_path(CAFFE_OCL_HDR caffe/net.hpp HINTS ${CAFFE_OCL_PATH}/include)
	find_library(CAFFE_OCL_LIB caffe HINTS ${CAFFE_OCL_PATH}/lib)
	add_executable(test_conv_caffe_ocl test_conv.cpp conv_ref.cpp conv_caffe.cpp)
	target_include_directories(test_conv_caffe_ocl PUBLIC ${CAFFE_OCL_HDR} ${OCL_PATH} ${BOOST_INC})
	target_link_libraries(test_conv_caffe_ocl ${CAFFE_OCL_LIB} openblas)
endif()


if(CAFFE_HIP_PATH)
	find_path(CAFFE_HIP_HDR caffe/net.hpp HINTS ${CAFFE_HIP_PATH}/include)
	find_library(CAFFE_HIP_LIB caffe HINTS ${CAFFE_HIP_PATH}/lib ${CAFFE_HIP_PATH}/build/lib)
	find_path(HIP_HDR hip/hip_runtime.h HINTS /opt/rocm/hip/include)
	find_path(HIP_RND_HDR hiprand.h  HINTS /opt/rocm/hiprand/include)
	find_path(HIP_LOG_HDR CXLActivityLogger.h HINTS /opt/rocm/profiler/CXLActivityLogger/include/)
	find_path(HIP_ROCRAND_HDR rocrand.h HINTS /opt/rocm/rocrand/include/)
	add_executable(test_conv_caffe_hip test_conv.cpp conv_ref.cpp conv_caffe.cpp)
	target_include_directories(test_conv_caffe_hip PUBLIC ${CAFFE_HIP_HDR} ${HIP_HDR} ${HIP_LOG_HDR} /opt/rocm/include  ${HIP_RND_HDR} ${HIP_ROCRAND_HDR} ${CAFFE_HIP_PATH}/.build_release/src ${BOOST_INC} )
	target_compile_definitions(test_conv_caffe_hip PUBLIC __HIP_PLATFORM_HCC__) 
	target_link_libraries(test_conv_caffe_hip ${CAFFE_HIP_LIB} openblas)
endif()

