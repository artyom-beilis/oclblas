#include "conv_base.h"
#include <miopen/miopen.h>
#include <stdexcept>
#include <iostream>
#include <string>

#if MIOPEN_BACKEND_OPENCL
#include <CL/cl.hpp>
#endif

#define check(x) do { miopenStatus_t r=(x); if(r!=miopenStatusSuccess) throw std::runtime_error("Call failed with status " + std::to_string(int(r)) + " at line " + std::to_string(__LINE__)); } while(0)

#if !MIOPEN_BACKEND_OPENCL
struct stream {
	hipStream_t operator()() const { return s_; }
	stream() {
		hipStreamCreate(&s_);
	}
	~stream() {
		hipStreamDestroy(s_);
	}
	hipStream_t s_;
};

struct buffer {
	buffer() : m_(nullptr) {}
	~buffer() 
	{
		if(m_)
			hipFree(m_);
	}
	buffer(buffer const &) = delete;
	void operator=(buffer const &) = delete;
	buffer &operator=(buffer &&other)
	{
		if(m_)
			hipFree(m_);
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
		hipMalloc(&m_,n);
	}
	void *m_;
	
};
#endif

class conv_miopen : public conv_base {
public:

#if MIOPEN_BACKEND_OPENCL
	cl::Platform platform_;
	cl::Device device_;
	cl::Context context_;
	cl::CommandQueue queue_;
#else
	stream queue_;
#endif

	miopenHandle_t handle_;	
	miopenTensorDescriptor_t inp_,out_,ker_;
	miopenConvolutionDescriptor_t desc_;
	miopenConvAlgoPerf_t perf_;
#if MIOPEN_BACKEND_OPENCL
	cl::Buffer buf_in_,buf_out_,buf_kern_,buf_ws_;
#else
	buffer buf_in_,buf_out_,buf_kern_,buf_ws_;
#endif
	size_t ws_size_;
	float *host_out_;
	
	conv_miopen(int platform,int device)  
	{
#if MIOPEN_BACKEND_OPENCL
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		std::vector<cl::Device> devices;
		platform_ = platforms[platform];
		platform_.getDevices(CL_DEVICE_TYPE_ALL, &devices);
		device_ = devices[device];
		auto device_as_vector = std::vector<cl::Device>{device_};
		context_ = cl::Context(device_as_vector);
		queue_ = cl::CommandQueue(context_, device_);
#endif
	}

	
#if MIOPEN_BACKEND_OPENCL
	cl::Buffer dalloc(size_t n)
	{
		if(n == 0)
			return cl::Buffer();
		else
			return cl::Buffer(context_, CL_MEM_READ_WRITE, n);
	}
	void h2d(cl::Buffer &buf,void const *p,size_t n)
	{
        	queue_.enqueueWriteBuffer(buf, CL_TRUE, 0, n,p);
	}
	void d2h(cl::Buffer &buf,void *p,size_t n)
	{
        	queue_.enqueueReadBuffer(buf,CL_TRUE,0,n,p);
	}
#else
	buffer dalloc(size_t n)
	{
		if(n==0)
			return buffer();
		return buffer(n);
	}
	void h2d(buffer &buf,void const *p,size_t n)
	{
		hipMemcpy(buf(),p,n, hipMemcpyHostToDevice);
	}
	void d2h(buffer &buf,void *p,size_t n)
	{
		hipMemcpy(p,buf(),n, hipMemcpyDeviceToHost);
	}
#endif


	virtual void config(conv_param const &param,int B,int C,int H,int W)
	{
		conv_base::config(param,B,C,H,W);

		check(miopenCreateWithStream(&handle_,queue_()));
		check(miopenCreateTensorDescriptor(&inp_));
		check(miopenCreateTensorDescriptor(&out_));
		check(miopenCreateTensorDescriptor(&ker_));

		check(miopenSet4dTensorDescriptor(inp_,miopenFloat,b_,c_,h_,w_));
		check(miopenSet4dTensorDescriptor(ker_,miopenFloat,par_.num_outputs,c_,par_.kernel_h,par_.kernel_w));

		check(miopenCreateConvolutionDescriptor(&desc_));
		int dilation_h = 1;
		int dilation_w = 1;
		check(miopenInitConvolutionDescriptor(desc_,miopenConvolution,par_.pad_h,par_.pad_w,par_.stride_h,par_.stride_w,dilation_h,dilation_w));
		std::array<int,4> shape;
		auto intshape=get_out_shape();
		check(miopenGetConvolutionForwardOutputDim(desc_,inp_,ker_,&shape[0],&shape[1],&shape[2],&shape[3]));
		if(shape!=intshape) {
			fprintf(stderr,"Mi %d:%d:%d:%d\n",shape[0],shape[1],shape[2],shape[3]);
			fprintf(stderr,"Int %d:%d:%d:%d\n",intshape[0],intshape[1],intshape[2],intshape[3]);
			throw std::runtime_error("Invalid shape");
		}

		check(miopenSet4dTensorDescriptor(out_,miopenFloat,b_,out_c_,out_h_,out_w_));

		ws_size_ = 0;
		check(miopenConvolutionForwardGetWorkSpaceSize(handle_,ker_,inp_,desc_,out_,&ws_size_));

		buf_ws_ =   std::move(dalloc( ws_size_));
        	buf_in_ =   std::move(dalloc( b_*c_*h_*w_*sizeof(float)));
        	buf_out_ =  std::move(dalloc( b_*out_c_*out_h_*out_w_*sizeof(float)));
		buf_kern_ = std::move(dalloc( par_.num_outputs*c_*par_.kernel_h*par_.kernel_w*sizeof(float)));
		int cnt=0;
		check(miopenFindConvolutionForwardAlgorithm(handle_,inp_,buf_in_(),ker_,buf_kern_(),desc_,out_,buf_out_(),
							1,&cnt,&perf_,buf_ws_(),ws_size_,false));


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
		//miopenConvFwdAlgorithm_t alg = miopenConvolutionFwdAlgoDirect;
		miopenConvFwdAlgorithm_t alg = perf_.fwd_algo;
		check(miopenConvolutionForward(handle_,&alpha,inp_,buf_in_(),ker_,buf_kern_(),desc_,alg,&beta,out_,buf_out_(),buf_ws_(),ws_size_));
	}
	virtual void sync() {
#if MIOPEN_BACKEND_OPENCL
        	queue_.finish();
#else
     	  	hipDeviceSynchronize();
#endif
	}
	virtual void copy_back() {
        	d2h(buf_out_,host_out_,b_*out_w_*out_h_*out_c_*sizeof(float));
	}
};


conv_base *get_conv_miopen(int p,int d) { return new conv_miopen(p,d); };

