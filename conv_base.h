#pragma once
#include <vector>
#include <array>
#include <string.h>

struct conv_param {
	int kernel_w = 1;
	int kernel_h = 1;
	int pad_w = 0;
	int pad_h = 0;
	int stride_w = 1;
	int stride_h = 1;
	int num_outputs = 1;
	int groups = 1;
};

class conv_base {
public:
	virtual void flush_cache() {};
	virtual void config(conv_param const &param,int B,int C,int H,int W) 
	{
		b_ = B;
		c_ = C;
		h_ = H;
		w_ = W;
		par_ = param;

		out_w_ = (w_ + par_.pad_w * 2 - par_.kernel_w) / par_.stride_w + 1;
		out_h_ = (h_ + par_.pad_h * 2 - par_.kernel_h) / par_.stride_h + 1;
		out_c_ = par_.num_outputs;
		
	}
	std::array<int,4> get_out_shape() const
	{
		return std::array<int,4>{b_,out_c_,out_h_,out_w_};
	}
	virtual void set_kernel(float const *k) = 0;
	virtual void set_input(float const *inp) = 0;
	virtual void set_output(float *outp) = 0;

	virtual void calc() = 0;
	virtual void sync() = 0;
	virtual void copy_back() = 0;
	virtual ~conv_base() {}

	int out_w_,out_h_,out_c_;
	conv_param par_;
	int b_,c_,h_,w_;

};


