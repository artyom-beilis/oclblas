// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#include "cublas.h"
#include "cublas_v2.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <vector>
#include <iostream>
#include <stdexcept>

#include "sgemm_base.h"

class conv_cudnn : public conv_base {
public:
    sgemm_cublas() : a_(0),b_(0),c_(0) 
    {
        cudnnCreate(&cudnn_);
    }
    virtual ~sgemm_cublas() { 
        cudaFree(ker_);
        cudaFree(inp_);
        cudaFree(out_);
        cudnnDestroy(cudnn_);
    }
    virtual void config(int kern,int pad,int out,int stride,int B,int C,int H,int W) 
    {
        ks_ = kern;
        pad_ = pad;
        out_ = out;
        stride_ = stride;
        B_ = B;
        C_ = C;
        H_ = H;
        W_ = W
        cudaMalloc((void **)&ker_,kern*kern*C*out*sizeof(float));
        cudaMalloc((void **)&inp_,B*C*W*H*sizeof(float));
        cudaMalloc((void **)&out_,B*out*W*H*sizeof(float));
    }
    virtual void set_A(float const *A) { 
        cudaMemcpy(a_,A,M_*K_*sizeof(float),cudaMemcpyHostToDevice);
    }
    virtual void set_B(float const *B) { 
        cudaMemcpy(b_,B,K_*N_*sizeof(float),cudaMemcpyHostToDevice);
    }
    virtual void set_C(float *C) { 
        c_host_ = C;
    }

    virtual void calc()
    {
        float alpha = 1.0f;
        float betha = 0.0f;
        cublasSgemm(h_,
                    (Btr_ ? CUBLAS_OP_T : CUBLAS_OP_N), (Atr_ ? CUBLAS_OP_T : CUBLAS_OP_N),
                    N_,M_,K_,
                    &alpha,
                    b_,(!Btr_?N_:K_),
                    a_,(!Atr_?K_:M_),
                    &betha,
                    c_,N_);
    }
    virtual void sync() {
        cudaDeviceSynchronize();
    }
    virtual void copy_back() {
        cudaMemcpy(c_host_,c_,M_*N_*sizeof(float),cudaMemcpyDeviceToHost);
    }
private:
    cublasHandle_t h_;
    int M_,N_,K_;
    bool Atr_;
    bool Btr_;
    float *a_;
    float *b_;
    float *c_;
    float *c_host_;
};

sgemm_base *get_cublas() { return new sgemm_cublas(); };

