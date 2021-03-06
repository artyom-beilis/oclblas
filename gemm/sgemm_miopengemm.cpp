#include <miopengemm/gemm.hpp>
#include <CL/cl.hpp>

#include "sgemm_base.h"

class sgemm_miopengemm : public sgemm_base {
public:
    cl::Platform platform_;
    cl::Device device_;
    cl::Context context_;
    cl::CommandQueue queue_;
    sgemm_miopengemm(int platform,int device)  
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
    virtual ~sgemm_miopengemm() { 
    }
    virtual void config(int M,int N,int K,bool Atr,bool Btr)
    {
        M_= M;
	N_ = N;
	K_ = K;
        Atr_ = Atr;
        Btr_ = Btr;
        a_ = std::move(cl::Buffer(context_, CL_MEM_READ_WRITE, M*K*sizeof(float)));
        b_ = std::move(cl::Buffer(context_, CL_MEM_READ_WRITE, K*N*sizeof(float)));
        c_ = std::move(cl::Buffer(context_, CL_MEM_READ_WRITE, M*N*sizeof(float)));
    }
    virtual void set_A(float const *A) { 
        queue_.enqueueWriteBuffer(a_, CL_TRUE, 0, M_*K_*sizeof(float), A);
    }
    virtual void set_B(float const *B) { 
        queue_.enqueueWriteBuffer(b_, CL_TRUE, 0, K_*N_*sizeof(float), B);
    }
    virtual void set_C(float *C) { 
        c_host_ = C;
    }

    virtual void calc()
    {
        float alpha = 1.0f;
        float beta = 0.0f;
        auto queue_plain = queue_();
	MIOpenGEMM::xgemm<float>(
			false,
                        (Atr_),
                        (Btr_),
                        M_, N_, K_,
                        alpha,
                        a_(), 0, (!Atr_ ? K_ : M_ ),
                        b_(), 0, (!Btr_ ? N_ : K_ ),
                        beta,
                        c_(), 0, N_,
			0,0,0,
                        &queue_plain, 
                        0,
                        nullptr,
                        nullptr,-1);
    }
    virtual void sync() {
        queue_.finish();
    }
    virtual void copy_back() {
        queue_.enqueueReadBuffer(c_,CL_TRUE,0,M_*N_*sizeof(float),c_host_);
    }
private:
    int M_,N_,K_;
    bool Atr_;
    bool Btr_;
    cl::Buffer a_,b_,c_;
    float *c_host_;
};

sgemm_base *get_miopengemm(int p,int d) { return new sgemm_miopengemm(p,d); };
sgemm_base *get_external(int p,int d) { return new sgemm_miopengemm(p,d); };

