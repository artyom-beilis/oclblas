#pragma once
class sgemm_base {
public:
    virtual void flush_cache() {};
    virtual void config(int M,int N,int K,bool Atr,bool Btr) = 0;
    virtual void set_A(float const *A) = 0;
    virtual void set_B(float const *B) = 0;
    virtual void set_C(float *C) = 0;

    virtual void calc() = 0;
    virtual void sync() = 0;
    virtual void copy_back() = 0;
    virtual ~sgemm_base() {}
};


