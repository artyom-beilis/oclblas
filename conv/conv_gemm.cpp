#include "conv_base.h"
#include <miopen/miopen.h>
#include <stdexcept>
#include <iostream>
#include <string>

#include <CL/cl.hpp>


class conv_gemm : public conv_base {
public:

	cl::Platform platform_;
	cl::Device device_;
	cl::Context context_;
	cl::CommandQueue queue_;
	
	cl::Buffer buf_in_,buf_out_,buf_kern_,buf_ws_;
	size_t ws_size_;
	float *host_out_;
	
	conv_gemm(int platform,int device)  
	{
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		std::vector<cl::Device> devices;
		platform_ = platforms[platform];
		platform_.getDevices(CL_DEVICE_TYPE_ALL, &devices);
		device_ = devices[device];
		auto device_as_vector = std::vector<cl::Device>{device_};
		context_ = cl::Context(device_as_vector);
		queue_ = cl::CommandQueue(context_, device_);
	}

	
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


	virtual void config(conv_param const &param,int B,int C,int H,int W)
	{
		conv_base::config(param,B,C,H,W);
		
		ws_size_ = out_c_*out_h_*out_w_*sizeof(float);

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
		int kernel_rows = par_.num_outputs;
		int kernel_cols_ = c_ * par_.kernel_h * par_.kernel_w;
	//miopenConvFwdAlgorithm_t alg = miopenConvolutionFwdAlgoDirect;

		kernel_.setArg(ind++,M_);
		kernel_.setArg(ind++,N_);
		kernel_.setArg(ind++,K_);
		kernel_.setArg(ind++,a_);
		kernel_.setArg(ind++,(!Atr_ ? K_ : M_ ));
		kernel_.setArg(ind++,b_);
		kernel_.setArg(ind++,(!Btr_ ? N_ : K_ ));
		kernel_.setArg(ind++,c_);
		kernel_.setArg(ind++,N_);

		int ls0 = tile_size_m_ / block_size_m_;
		int ls1 = tile_size_n_ / block_size_n_; 
		int gs0 = round_up_div(M_,tile_size_m_) * tile_size_m_ / block_size_m_;
		int gs1 = round_up_div(N_,tile_size_n_) * tile_size_n_ / block_size_n_;
		cl::NDRange global(gs0,gs1);
		cl::NDRange local(ls0,ls1);
		rc=queue_.enqueueNDRangeKernel(kernel_, cl::NullRange, global,local,nullptr,nullptr);

		if(rc!=0)
			throw std::runtime_error("Failed to enqueue the kernel: " + std::to_string(rc));


		miopenConvFwdAlgorithm_t alg = perf_.fwd_algo;
		check(miopenConvolutionForward(handle_,&alpha,inp_,buf_in_(),ker_,buf_kern_(),desc_,alg,&beta,out_,buf_out_(),buf_ws_(),ws_size_));
	}
	virtual void sync() {
        	queue_.finish();
	}
	virtual void copy_back() {
        	d2h(buf_out_,host_out_,b_*out_w_*out_h_*out_c_*sizeof(float));
	}
};


conv_base *get_conv_miopen(int p,int d) { return new conv_miopen(p,d); };

