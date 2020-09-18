#include "conv_base.h"
#include <stdexcept>
#include <iostream>
#include <string>

#include <CL/cl.hpp>
#include <clblast.h>

class conv_clblast : public conv_base {
public:

	cl::Platform platform_;
	cl::Device device_;
	cl::Context context_;
	cl::CommandQueue queue_;

	cl::Buffer buf_in_,buf_out_,buf_kern_,buf_ws_;
	size_t ws_size_;
	float *host_out_;
	
	conv_clblast(int platform,int device)  
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
        //buf_in_ =   std::move(dalloc( b_*c_*h_*w_*sizeof(float)));
        //buf_out_ =  std::move(dalloc( b_*out_c_*out_h_*out_w_*sizeof(float)));
		//buf_kern_ = std::move(dalloc( par_.num_outputs*c_*par_.kernel_h*par_.kernel_w*sizeof(float)));
        auto queue_plain = queue_();
        clblast::Convgemm<float>(clblast::KernelMode::kCrossCorrelation,
                                 c_,
                                 h_,w_,par_.kernel_h,par_.kernel_w,
                                 par_.pad_h,par_.pad_w,par_.stride_h,par_.stride_w,1,1,out_c_,b_,
                                 buf_in_(),0,
                                 buf_kern_(),0,
                                 buf_out_(),0,
                                 &queue_plain);


	}
	virtual void sync() {
        	queue_.finish();
	}
	virtual void copy_back() {
        	d2h(buf_out_,host_out_,b_*out_w_*out_h_*out_c_*sizeof(float));
	}
};


conv_base *get_conv_clblast(int p,int d) { return new conv_clblast(p,d); };
conv_base *get_external(int p,int d) { return new conv_clblast(p,d); };

