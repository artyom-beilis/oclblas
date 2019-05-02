#include "conv_base.h"
#include <cudnn.h>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <string>

#define check(expression)                               \
do{                                                          \
	cudnnStatus_t status = (expression);                     \
	if (status != CUDNN_STATUS_SUCCESS) {                    \
		std::ostringstream ss; ss << "Error on line " << __LINE__ << ": "      \
		<< cudnnGetErrorString(status) << std::endl; \
		throw std::runtime_error(ss.str()); \
	}                                                        \
}while(0)


namespace {
struct stream {
	cudaStream_t operator()() const { return s_; }
	stream() {
		cudaStreamCreate(&s_);
	}
	~stream() {
		cudaStreamDestroy(s_);
	}
	cudaStream_t s_;
};
struct buffer {
	buffer() : m_(nullptr) {}
	~buffer() 
	{
		if(m_)
			cudaFree(m_);
	}
	buffer(buffer const &) = delete;
	void operator=(buffer const &) = delete;
	buffer &operator=(buffer &&other)
	{
		if(m_)
			cudaFree(m_);
		m_ = other.m_;
		other.m_ = nullptr;
		return *this;
	}
	void *operator()() const {
		return m_;
	}
	buffer(buffer &&other) : m_(other.m_)
	{
		other.m_ = nullptr;
	}
	buffer(size_t n) : m_(nullptr)
	{
		cudaMalloc(&m_,n);
	}
	void *m_;
	
};
} // namespace

class conv_cudnn : public conv_base {
public:

	stream queue_;

	cudnnHandle_t handle_;	
	cudnnTensorDescriptor_t inp_,out_;
	cudnnFilterDescriptor_t ker_;
	cudnnConvolutionDescriptor_t desc_;
	cudnnConvolutionFwdAlgo_t perf_;
	buffer buf_in_,buf_out_,buf_kern_,buf_ws_;

	size_t ws_size_;
	float *host_out_;
	
	conv_cudnn(int device)  
	{
	}

	
	buffer dalloc(size_t n)
	{
		if(n==0)
			return buffer();
		return buffer(n);
	}
	void h2d(buffer &buf,void const *p,size_t n)
	{
		cudaMemcpy(buf(),p,n, cudaMemcpyHostToDevice);
	}
	void d2h(buffer &buf,void *p,size_t n)
	{
		cudaMemcpy(p,buf(),n, cudaMemcpyDeviceToHost);
	}


	virtual void config(conv_param const &param,int B,int C,int H,int W)
	{
		conv_base::config(param,B,C,H,W);

		check(cudnnCreate(&handle_));
		check(cudnnCreateTensorDescriptor(&inp_));
		check(cudnnCreateTensorDescriptor(&out_));
		check(cudnnCreateFilterDescriptor(&ker_));

		check(cudnnSetTensor4dDescriptor(inp_,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,b_,c_,h_,w_));
		check(cudnnSetTensor4dDescriptor(out_,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,b_,out_c_,out_h_,out_w_));
		check(cudnnSetFilter4dDescriptor(ker_,CUDNN_DATA_FLOAT,CUDNN_TENSOR_NCHW,par_.num_outputs,c_,par_.kernel_h,par_.kernel_w));

		check(cudnnCreateConvolutionDescriptor(&desc_));
		int dilation_h = 1;
		int dilation_w = 1;
		check(cudnnSetConvolution2dDescriptor(desc_,par_.pad_h,par_.pad_w,par_.stride_h,par_.stride_w,
					dilation_h,dilation_w,CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT));

		std::array<int,4> shape;
		auto intshape=get_out_shape();
		check(cudnnGetConvolution2dForwardOutputDim(desc_,inp_,ker_,&shape[0],&shape[1],&shape[2],&shape[3]));
		if(shape!=intshape) {
			fprintf(stderr,"Cidnn %d:%d:%d:%d\n",shape[0],shape[1],shape[2],shape[3]);
			fprintf(stderr,"Int   %d:%d:%d:%d\n",intshape[0],intshape[1],intshape[2],intshape[3]);
			throw std::runtime_error("Invalid shape");
		}
		
		check(cudnnGetConvolutionForwardAlgorithm(handle_,inp_,ker_,desc_,out_,CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,0,&perf_));
		std::cerr << "alogo=" << int(perf_) << std::endl;

		ws_size_ = 0;
		check(cudnnGetConvolutionForwardWorkspaceSize(handle_,inp_,ker_,desc_,out_,perf_,&ws_size_));

		std::cerr << "ws=" << ws_size_ << std::endl;

		buf_ws_ =   std::move(dalloc( ws_size_));
        	buf_in_ =   std::move(dalloc( b_*c_*h_*w_*sizeof(float)));
        	buf_out_ =  std::move(dalloc( b_*out_c_*out_h_*out_w_*sizeof(float)));
		buf_kern_ = std::move(dalloc( par_.num_outputs*c_*par_.kernel_h*par_.kernel_w*sizeof(float)));


	}
	virtual void set_kernel(float const *A)
	{
		h2d(buf_kern_,A,par_.num_outputs*c_*par_.kernel_h*par_.kernel_w*sizeof(float));
	}
	virtual void set_input(float const *A) {
        	h2d(buf_in_, A, b_*c_*h_*w_*sizeof(float));
	}
	virtual void set_output(float *outp) { host_out_ = outp; }

	virtual void calc() {
		float alpha=1.0f;
		float beta=0.0f;
		check(cudnnConvolutionForward(handle_,&alpha,inp_,buf_in_(),ker_,buf_kern_(),desc_,perf_,buf_ws_(),ws_size_,&beta,out_,buf_out_()));
	}
	virtual void sync() {
     	  	cudaDeviceSynchronize();
	}
	virtual void copy_back() {
        	d2h(buf_out_,host_out_,b_*out_w_*out_h_*out_c_*sizeof(float));
	}
};


conv_base *get_conv_cudnn(int p,int d) { return new conv_cudnn(d); };

