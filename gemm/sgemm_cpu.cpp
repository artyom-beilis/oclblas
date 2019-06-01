#include "sgemm_base.h"
#include "cblas.h"


class sgemm_cpu : public sgemm_base {
public:
    virtual void config(int M,int N,int K,bool Atr,bool Btr)
    {
        M_= M;
        N_= N;
        K_= K;
        Atr_ = Atr;
        Btr_ = Btr;
    }
    virtual void set_A(float const *A) { a_ = A; }
    virtual void set_B(float const *B) { b_ = B; }
    virtual void set_C(float *C) { c_ = C; }

    virtual void calc()
    {
        cblas_sgemm(CblasRowMajor,(Atr_ ? CblasTrans : CblasNoTrans), (Btr_ ? CblasTrans : CblasNoTrans),
                    M_,N_,K_,
                    1.0f,
                    a_,(!Atr_?K_:M_),
                    b_,(!Btr_?N_:K_),
                    0.0f,
                    c_,N_);
    }
    virtual void sync() {}
    virtual void copy_back() {}
private:
    int M_,N_,K_;

    bool Atr_;
    bool Btr_;
    float const *a_;
    float const *b_;
    float *c_;
};

sgemm_base *get_cpu() { return new sgemm_cpu(); };

