#OBJECTS=test.o sgemm_cpu.o sgemm_cublas.o sgemm_clblast.o sgemm_clblas.o sgemm_my.o sgemm_mycuda.o sgemm_viennacl.o
#OBJECTS=test.o sgemm_cpu.o sgemm_cublas.o sgemm_clblast.o sgemm_clblas.o sgemm_viennacl.o sgemm_my.o sgemm_mycuda.o
OBJECTS=test.o sgemm_cpu.o sgemm_clblast.o sgemm_clblas.o sgemm_viennacl.o sgemm_my.o sgemm_miopengemm.o sgemm_cublas.o sgemm_cutlass.o 
#OBJECTS=test.o sgemm_cpu.o sgemm_clblast.o sgemm_clblas.o sgemm_viennacl.o sgemm_my.o 
CFLAGS=-I /opt/clblast/clblas/include -I /opt/clblast/clblast-1.3.0/include -I /opt/rocm/opencl/include/ -DVIENNACL_WITH_OPENCL=1  -Wall -Wno-deprecated-declarations -O3 -g -std=c++11 -I /data/Projects/opencl/rocm/half/include/ -I /usr/local/cuda-9.2/targets/x86_64-linux/include  -I /opt/rocm/miopengemm/include
#CFLAGS=-I /usr/include -DVIENNACL_WITH_OPENCL=1 -I /opt/caffe/clblast/include/ -I /opt/caffe/clblas/include/  -I /export/home/artyo-be/stuff/caffe/viennacl/ViennaCL-1.7.1 -I /opt/cuda-8.0/include  -Wall -O3 -g -std=c++11

LINKFLAGS=-L/usr/local/cuda-9.2/lib64 -lopenblas -lcublas -lcudart -lnvrtc -lcuda -lOpenCL -Wl,-rpath=/usr/local/cuda-9.2/lib64:/opt/clblast/clblast-1.3.0/lib/:/opt/clblast/clblaslib64 -L /opt/clblast/clblast-1.3.0/lib/ -L /opt/clblast/clblas/lib64 -lclblast -lclBLAS
LINKFLAGS=-lopenblas  -lOpenCL -Wl,-rpath=/opt/rocm/miopengemm/lib/:/opt/clblast/clblast-1.3.0/lib/:/opt/clblast/clblaslib64 -L /opt/clblast/clblast-1.3.0/lib/ -L /opt/clblast/clblas/lib64 -lclblast -lclBLAS -L /opt/rocm/miopengemm/lib/ -lmiopengemm -L/usr/local/cuda-9.2/lib64 -lcublas -lcudart -lnvrtc -lcuda
#LINKFLAGS=-lopenblas -lOpenCL -Wl,-rpath=/opt/clblast/clblast-1.3.0/lib/:/opt/clblast/clblaslib64 -L /opt/clblast/clblast-1.3.0/lib/ -L /opt/clblast/clblas/lib64 -lclblast -lclBLAS


#test:$(OBJECTS) cuker.o
#	g++ -o test $(OBJECTS) cuker.o $(LINKFLAGS) 
#
#cuker.o:kernels2.cu
#	nvcc --gpu-architecture=compute_52 --gpu-code=compute_52 -c kernels2.cu -o cuker.o

all: test test_conv test_conv_hip test_cu

test:$(OBJECTS) cutlass_gemm.o 
	g++ -o test $(OBJECTS) cutlass_gemm.o $(LINKFLAGS) 

test_cu: sgemm_mycuda.cpp test.cpp
	g++ $(CFLAGS) -DCUDA_ONLY -o test_cu test.cpp sgemm_mycuda.cpp $(LINKFLAGS) 

test_conv: test_conv.cpp conv_miopen.cpp conv_base.h conv_ref.cpp conv_cudnn.cpp
	g++ -I /usr/local/cuda-9.2/targets/x86_64-linux/include/ -I /home/artik/Packages/cuda/cudnn-9.0-linux-x64-v7.1/cuda/include -I /opt/rocm/miopen-opencl/include -I /home/artik/Packages/viennacl/viennacl-dev -Wall -std=c++11 -g -O2 -o test_conv test_conv.cpp conv_cudnn.cpp conv_ref.cpp conv_miopen.cpp -L /opt/rocm/miopen-opencl/lib/ -L /home/artik/Packages/cuda/cudnn-9.0-linux-x64-v7.1/cuda/lib64/ -L /usr/local/cuda-9.2/targets/x86_64-linux/lib -lOpenCL -lMIOpen -lcudnn -lcudart -Wl,-rpath=/opt/rocm/miopen-opencl/lib/:/home/artik/Packages/cuda/cudnn-9.0-linux-x64-v7.1/cuda/lib64/:/usr/local/cuda-9.2/targets/x86_64-linux/lib -lopenblas  

test_conv_hip: test_conv.cpp conv_miopen.cpp conv_base.h conv_ref.cpp
	g++ -D __HIP_PLATFORM_HCC__   -I /opt/rocm/hip/include -I /opt/rocm/miopen/include -Wall -std=c++11 -g -O2 -o test_conv_hip test_conv.cpp conv_miopen.cpp conv_ref.cpp -L /opt/rocm/miopen/lib/ -lMIOpen -Wl,-rpath=/opt/rocm/miopen/lib/:/opt/rocm/hip/lib -L /opt/rocm/hip/lib -lhip_hcc -lopenblas


cutlass_gemm.o: cutlass_gemm.cu
	/usr/local/cuda-9.2/bin/nvcc -O2 -c -I ~/Packages/cuda/cutlass/ cutlass_gemm.cu	

$(OBJECTS): %.o: %.cpp sgemm_base.h 
	g++ -c $(CFLAGS) $< -o $@

clean:
	rm -f test *.o
