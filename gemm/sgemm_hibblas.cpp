#include <cublas.h>
#include <cublas_v2.h>
#include <hip/hip_runtime_api.h>
#include <cuda.h>
#include <vector>
#include <iostream>
#include <stdexcept>

#include "sgemm_base.h"

class sgemm_cublas : public sgemm_base {
public:
    sgemm_cublas() : a_(0),b_(0),c_(0) 
    {
	int status;
        if((status=cublasInit())!=0)
		throw std::runtime_error(std::string("Failed to initialize cublas:") + std::to_string(status));
        if((status=cublasCreate(&h_))!=0)
		throw std::runtime_error(std::string("Failed to create cublas:") + std::to_string(status));
    }
    virtual ~sgemm_cublas() { 
        hipFree(a_);
        hipFree(b_);
        hipFree(c_);
        cublasDestroy(h_);
    }
    virtual void config(int M,int N,int K,bool Atr,bool Btr)
    {
        M_ = M;
	N_ = N;
	K_ = K;
        Atr_ = Atr;
        Btr_ = Btr;
        hipMalloc((void **)&a_,M*K*sizeof(float));
        hipMalloc((void **)&b_,K*N*sizeof(float));
        hipMalloc((void **)&c_,M*N*sizeof(float));
    }
    virtual void set_A(float const *A) { 
        hipMemcpy(a_,A,M_*K_*sizeof(float),hipMemcpyHostToDevice);
    }
    virtual void set_B(float const *B) { 
        hipMemcpy(b_,B,K_*N_*sizeof(float),hipMemcpyHostToDevice);
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
        hipDeviceSynchronize();
    }
    virtual void copy_back() {
        hipMemcpy(c_host_,c_,M_*N_*sizeof(float),hipMemcpyDeviceToHost);
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

