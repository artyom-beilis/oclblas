#include <cuda_runtime_api.h>
#include <cuda.h>
#include <vector>
#include <iostream>
#include <stdexcept>

#include "sgemm_base.h"

void mysgemm_call(int M,float *A,float *B,float *C);


class sgemm_mycuda : public sgemm_base {
public:
    sgemm_mycuda() : a_(0),b_(0),c_(0) 
    {
    }
    virtual ~sgemm_mycuda() { 
        cudaFree(a_);
        cudaFree(b_);
        cudaFree(c_);
    }
    virtual void config(int M,bool Atr,bool Btr)
    {
        M_= M;
        Atr_ = Atr;
        Btr_ = Btr;
        cudaMalloc((void **)&a_,M*M*sizeof(float));
        cudaMalloc((void **)&b_,M*M*sizeof(float));
        cudaMalloc((void **)&c_,M*M*sizeof(float));
    }
    virtual void set_A(float const *A) { 
        cudaMemcpy(a_,A,M_*M_*sizeof(float),cudaMemcpyHostToDevice);
    }
    virtual void set_B(float const *B) { 
        cudaMemcpy(b_,B,M_*M_*sizeof(float),cudaMemcpyHostToDevice);
    }
    virtual void set_C(float *C) { 
        c_host_ = C;
    }

    virtual void calc()
    {
        mysgemm_call(M_,a_,b_,c_);
    }
    virtual void sync() {
        cudaDeviceSynchronize();
    }
    virtual void copy_back() {
        cudaMemcpy(c_host_,c_,M_*M_*sizeof(float),cudaMemcpyDeviceToHost);
    }
private:
    int M_;
    bool Atr_;
    bool Btr_;
    float *a_;
    float *b_;
    float *c_;
    float *c_host_;
};

sgemm_base *get_mycuda() { return new sgemm_mycuda(); };

