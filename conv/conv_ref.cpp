#include "conv_base.h"
#include <cblas.h>
#include <assert.h>
#include <iostream>

//#define DEBUG_CONV

class conv_ref : public conv_base {
public:
	virtual void config(conv_param const &param,int B,int C,int H,int W)
	{
		conv_base::config(param,B,C,H,W);

		im2col_rows_ = out_w_ * out_h_;
		im2col_cols_ = c_ * par_.kernel_h * par_.kernel_w;
		kernel_rows_ = par_.num_outputs;
		kernel_cols_ = c_ * par_.kernel_h * par_.kernel_w;

		#ifdef DEBUG_CONV
		printf("im2col(%d,%d) kernel(%d,%d)\n",im2col_rows_,im2col_cols_,kernel_rows_,kernel_cols_);
		#endif
		assert(im2col_cols_ == kernel_cols_);

		kernel_.resize(kernel_rows_ * kernel_cols_);
		im2col_.resize(im2col_rows_ * im2col_cols_);
	}
	virtual void set_kernel(float const *k)
	{
		memcpy(kernel_.data(),k,kernel_.size()*sizeof(float));
	}
	virtual void set_input(float const *inp) { in_ = inp; }
	virtual void set_output(float *outp) { out_ = outp; }

	void im2col(float *col,float const *img)
	{
		int inp_ch_stride = w_ * h_;
		for(int r=0;r<out_h_;r++) {
			int y_pos = -par_.pad_h + r * par_.stride_h;
			for(int c=0;c<out_w_;c++) {
				int x_pos = -par_.pad_w + c * par_.stride_w;
				for(int chan=0;chan < c_;chan++) {
					for(int dy = 0;dy < par_.kernel_h;dy++) {
						for(int dx=0;dx < par_.kernel_w;dx++) {
							int x = x_pos + dx;
							int y = y_pos + dy;
							float v = 0.0;
							if(x >=0 && x < w_ && y>= 0 && y<h_)
								v=img[chan*inp_ch_stride + y * w_ + x];
							*col++ = v;
						}
					}
				}
			}
		}
	}

	void im2col_inv(float *col,float const *img)
	{
		int kh = par_.kernel_h;
		int kw = par_.kernel_w;
		int kd = c_;
		int ph = par_.pad_h;
		int pw = par_.pad_w;
		int sh = par_.stride_h;
		int sw = par_.stride_w;
		int kernel_size = kd * kw * kh;
		int oimg_size = out_w_ * out_h_;
		for(int ipos = 0;ipos < oimg_size;ipos++) {
            int img_row0 = (ipos / out_w_) * sh - ph;
            int img_col0 = (ipos % out_w_) * sw - pw;
			for(int ppos = 0;ppos < kernel_size;ppos ++) {
				int depth = ppos / (kw*kh);
				int dy = ppos % (kw*kh) / kw;
				int dx = ppos % kw;

				int img_row = img_row0 + dy;
				int img_col = img_col0 + dx;
				float value = 0.0f;
				if(0<= img_row && img_row < h_ && 0<=img_col && img_col < w_) {
					value = img[w_ * h_ * depth + img_row * w_ + img_col];
				}
				*col++ = value;
			}
		}
	}

	virtual void calc() {
		std::vector<float> tmp_data(im2col_.size());
		for(int N=0;N<b_;N++) {
			im2col(im2col_.data(),&in_[c_*h_*w_ * N]);
#if 0
			im2col_inv(tmp_data.data(),&in_[c_*h_*w_ * N]);
			if(tmp_data != im2col_) {
				std::cerr << "Failed!!!" << std::endl;
			}
#endif            
			#ifdef DEBUG_CONV
			printf("im2col\n");
			{
				float *ptr = im2col_.data();
				for(int r=0;r<im2col_rows_;r++) {
					for(int c=0;c<im2col_cols_;c++) {
						printf("%4f,",*ptr++);
					}
					printf("\n");
				}
			}
			printf("Kernel\n");
			{
				float *ptr = kernel_.data();
				for(int r=0;r<kernel_rows_;r++) {
					for(int c=0;c<kernel_cols_;c++) {
						printf("%4f,",*ptr++);
					}
					printf("\n");
				}
			}
			#endif
			//if(N==0) {
			//	printf("M=%d N=%d K=%d\n",kernel_rows_,im2col_rows_,kernel_cols_); 
            //}
			cblas_sgemm(CblasRowMajor,CblasNoTrans, CblasTrans,
					kernel_rows_,im2col_rows_,kernel_cols_,
					1.0f,
					kernel_.data(),kernel_cols_,
					im2col_.data(),im2col_cols_,
					0.0f,
					out_ + out_c_*out_w_*out_h_ * N,
					im2col_rows_);
		}

	}
	virtual void sync() {}
	virtual void copy_back() {}
private:
	int kernel_rows_,kernel_cols_;
	int im2col_rows_,im2col_cols_;
	float const *in_;
	float *out_;
	std::vector<float> kernel_;
	std::vector<float> im2col_;
};

conv_base *get_conv_ref() { return new conv_ref(); }

