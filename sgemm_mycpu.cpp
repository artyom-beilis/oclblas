#include "sgemm_base.h"
#include "cblas.h"


class sgemm_cpu : public sgemm_base {
public:
    virtual void config(int M,bool Atr,bool Btr)
    {
        M_= M;
        Atr_ = Atr;
        Btr_ = Btr;
    }
    virtual void set_A(float const *A) { a_ = A; }
    virtual void set_B(float const *B) { b_ = B; }
    virtual void set_C(float *C) { c_ = C; }

    #define BLOCK_SIZE 32
    #define BLOCK_SIZE2 (BLOCK_SIZE*BLOCK_SIZE)
    void inline small_kernel()
    {
        int M = M_;
        float const *A=a_;
        float const *B=b_;
        float *C = c_;
        float  ALIGN_FLOAT4 lA[BLOCK_SIZE2];
        float  ALIGN_FLOAT4 lB[BLOCK_SIZE2];
        for(int row=0;row < M_;row++) {
            for(int col=0;col < M_;col ++) {
                lA[row*M+col] = A[row*M+col];
                lB[col*M+row] = B[row*M+col];
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);

        int aoff = row*M;
        int boff = col*M;
        __local float4 *a = (__local float4*)&lA[aoff];
        __local float4 *b = (__local float4*)&lB[boff];

        float s;
        if(M == 2)
            s=lA[aoff]*lB[boff] + lA[aoff+1]*lB[boff+1];
        else
            s=dot(a[0],b[0]);
        if(M >= 8)
            s+= dot(a[1],b[1]);
        if(M >= 16)
            s+= dot(a[2],b[2]) +  dot(a[3],b[3]);
        if(M >= 32)
            s+= dot(a[4],b[4]) +  dot(a[5],b[5]) + dot(a[6],b[6]) + dot(a[7],b[7]);
        C[row*M+col] = s;
    }
    virtual void calc()
    {
        if(M_ < 64
        cblas_sgemm(CblasRowMajor,(Atr_ ? CblasTrans : CblasNoTrans), (Btr_ ? CblasTrans : CblasNoTrans),
                    M_,M_,M_,
                    1.0f,
                    a_,M_,
                    b_,M_,
                    0.0f,
                    c_,M_);
    }
    virtual void sync() {}
    virtual void copy_back() {}
private:
    int M_;
    bool Atr_;
    bool Btr_;
    float const *a_;
    float const *b_;
    float *c_;
};

sgemm_base *get_cpu() { return new sgemm_cpu(); };

