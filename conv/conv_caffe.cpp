#include "conv_base.h"
#include <caffe/greentea/libdnn.hpp>
#include <caffe/device.hpp>
#include <stdexcept>
#include <iostream>
#include <string>
#include <memory>

#include <CL/cl.hpp>


class conv_libdnn : public conv_base {
public:

	cl::Platform platform_;
	cl::Device device_;
	cl::Context context_;
	cl::CommandQueue queue_;
	std::unique_ptr<caffe::device> caffe_device_;
	std::unique_ptr<caffe::LibDNNConv<float> > lib_;
	
	cl::Buffer buf_in_,buf_out_,buf_kern_,buf_ws_;
	size_t ws_size_;
	float *host_out_;
	
	conv_libdnn(int platform,int device)  
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

	virtual ~conv_libdnn()
	{
		lib_.reset();
		caffe_device_.reset();
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
		
		caffe_device_.reset(new caffe::device(0,0,caffe::Backend::BACKEND_OpenCL));
	
		caffe::LibDNNConvConfig cfg;
		cfg.dev_ptr = caffe_device_.get();
		cfg.in_shape = { B, C, W, H};
		cfg.out_shape = { B, out_c_, out_h_, out_w_ };
		cfg.kernel = { par_.kernel_h, par_.kernel_w };
		cfg.pad = { par_.pad_h, par_.pad_w };
		cfg.stride = { par_.stride_h, par_.stride_w };
		cfg.dilation = { 1 ,  1};
		cfg.weights_backward = false;
		cfg.bias_backward = false;
		cfg.phase_test = true;
		cfg.wgalgo = caffe::LIBDNN_CONVOLUTION_WG_ALGO_ATOMIC;

		lib_.reset(new caffe::LibDNNConv<float>(cfg)); 
	
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
		lib_->Forward((float const *)buf_in_(),(float const *)buf_kern_(),nullptr,(float *)(buf_out_()),b_);
	}
	virtual void sync() {
        	queue_.finish();
	}
	virtual void copy_back() {
        	d2h(buf_out_,host_out_,b_*out_w_*out_h_*out_c_*sizeof(float));
	}
};


conv_base *get_conv_libdnn(int p,int d) { return new conv_libdnn(p,d); };

