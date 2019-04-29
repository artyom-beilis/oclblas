#include "conv_base.h"
#include <cblas.h>
#include <assert.h>

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

	virtual void calc() {
		for(int N=0;N<b_;N++) {
			im2col(im2col_.data(),&in_[c_*h_*w_ * N]);
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
			if(N==0)
				printf("M=%d N=%d K=%d\n",kernel_rows_,im2col_rows_,kernel_cols_); 
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

