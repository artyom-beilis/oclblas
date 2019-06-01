#include "conv_base.h"
#include <caffe/net.hpp>
#include <stdexcept>
#include <iostream>
#include <string>
#include <memory>


class conv_caffe : public conv_base {
public:

	std::unique_ptr<caffe::Net<float> > caffe_;
	
	float *host_out_;
	
	conv_caffe(int device)  
	{
		caffe::Caffe::set_mode(caffe::Caffe::GPU);
		caffe::Caffe::SetDevice(device);
	}

	virtual void config(conv_param const &param,int B,int C,int H,int W)
	{
		conv_base::config(param,B,C,H,W);
		std::ofstream proto("/tmp/conv.prototxt");
		char buf[1024];
		snprintf(buf,sizeof(buf),
			R"(
			layer { 
				name: "data"
				top:"data"
				type:"Input"
				input_param { shape { dim: %d dim: %d dim: %d dim: %d } }
			}
			layer {
				name : "conv"
				type: "Convolution"
				bottom : "data"
				top: "conv"
				convolution_param {
					num_output: %d
					kernel_size: %d
					stride: %d
					pad:%d
					bias_term:false
				}
			}
			)", B,C,H,W,out_c_,par_.kernel_h,par_.stride_h,par_.pad_h);
		proto << buf;
		proto.close();

		caffe_.reset(new caffe::Net<float>("/tmp/conv.prototxt",caffe::TEST));

	}
	virtual void set_kernel(float const *A)
	{
		memcpy(caffe_->params().at(0)->mutable_cpu_data(),A,par_.num_outputs*c_*par_.kernel_h*par_.kernel_w*sizeof(float));
	}
	virtual void set_input(float const *A) {
		memcpy(caffe_->input_blobs()[0]->mutable_cpu_data(),A,b_*c_*h_*w_*sizeof(float));
	}
	virtual void set_output(float *outp) { host_out_ = outp; }

	virtual void calc() {
		caffe_->Forward();
	}
	virtual void sync() {
		caffe_->output_blobs()[0]->mutable_cpu_data();
	}
	virtual void copy_back() {
        	memcpy(host_out_,caffe_->output_blobs()[0]->mutable_cpu_data(),b_*out_w_*out_h_*out_c_*sizeof(float));
	}
};


conv_base *get_external(int p,int d) { return new conv_caffe(p); };

