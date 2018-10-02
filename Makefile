#OBJECTS=test.o sgemm_cpu.o sgemm_cublas.o sgemm_clblast.o sgemm_clblas.o sgemm_my.o sgemm_mycuda.o sgemm_viennacl.o
OBJECTS=test.o sgemm_cpu.o sgemm_cublas.o sgemm_clblast.o sgemm_clblas.o sgemm_viennacl.o
CFLAGS=-I /opt/clblast/clblas/include -I /opt/clblast/clblast-1.3.0/include -I /opt/rocm/opencl/include/ -DVIENNACL_WITH_OPENCL=1  -Wall -Wno-deprecated-declarations -O3 -g -std=c++11 -I /data/Projects/opencl/rocm/half/include/ 
#CFLAGS=-I /usr/include -DVIENNACL_WITH_OPENCL=1 -I /opt/caffe/clblast/include/ -I /opt/caffe/clblas/include/  -I /export/home/artyo-be/stuff/caffe/viennacl/ViennaCL-1.7.1 -I /opt/cuda-8.0/include  -Wall -O3 -g -std=c++11

LINKFLAGS=-lopenblas -lcublas -lcudart -lOpenCL -Wl,-rpath=/opt/clblast/clblast-1.3.0/lib/:/opt/clblast/clblaslib64 -L /opt/clblast/clblast-1.3.0/lib/ -L /opt/clblast/clblas/lib64 -lclblast -lclBLAS
#LINKFLAGS=-lopenblas /opt/cuda-8.0/lib64/libcublas.so  /opt/cuda-8.0/lib64/libcudart.so -L /usr/lib/x86_64-linux-gnu/ /usr/lib/x86_64-linux-gnu/libOpenCL.so /opt/caffe/clblast/lib/libclblast.so /opt/caffe/clblas/lib64/libclBLAS.so   -Wl,-rpath=/usr/lib/x86_64-linux-gnu:/opt/cuda-8.0/lib64:/opt/caffe/clblast/lib:/opt/caffe/clblas/lib64


#test:$(OBJECTS) cuker.o
#	g++ -o test $(OBJECTS) cuker.o $(LINKFLAGS) 
#
#cuker.o:kernels2.cu
#	nvcc --gpu-architecture=compute_52 --gpu-code=compute_52 -c kernels2.cu -o cuker.o

test:$(OBJECTS) 
	g++ -o test $(OBJECTS) $(LINKFLAGS) 


$(OBJECTS): %.o: %.cpp sgemm_base.h 
	g++ -c $(CFLAGS) $< -o $@


