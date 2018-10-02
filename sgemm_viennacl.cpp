#include "viennacl/matrix.hpp"
#include "viennacl/matrix_proxy.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/vector_proxy.hpp"

#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/prod.hpp"


#include "sgemm_base.h"

class sgemm_viennacl : public sgemm_base {
public:
    struct matrix_helper {
        float *ptr_;
        int M_,N_;
        matrix_helper(float *a,int M,int N) : ptr_(a), M_(M), N_(N) {}
        typedef size_t size_type;
        size_t size1() const { return M_; };
        size_t size2() const { return N_; }
        float &operator()(int i,int j) { return ptr_[i*N_+j]; }
        float const &operator()(int i,int j) const { return ptr_[i*N_+j]; }
    };
    sgemm_viennacl()
    {
    }
    virtual ~sgemm_viennacl() { 
    }
    virtual void config(int M,int N,int K,bool Atr,bool Btr)
    {
        M_= M;
	N_= N;
	K_= K;
        Atr_ = Atr;
        Btr_ = Btr;
    }
    virtual void set_A(float const *Ain) { 
        matrix_helper A(const_cast<float *>(Ain),M_,K_);
        a_ = std::move(viennacl::matrix<float,viennacl::row_major>(M_,K_));
        viennacl::copy(A,a_);
    }
    virtual void set_B(float const *Bin) { 
        matrix_helper B(const_cast<float *>(Bin),K_,N_);
        b_ = std::move(viennacl::matrix<float,viennacl::row_major>(K_,N_));
        viennacl::copy(B,b_);
    }
    virtual void set_C(float *C) { 
        c_ = std::move(viennacl::matrix<float,viennacl::row_major>(M_,N_));
        c_host_ = C;
    }

    virtual void calc()
    {
        using viennacl::linalg::prod;
        if(!Atr_ && !Btr_) {
            c_ = prod(a_,b_);
        }
        else if(!Atr_ && Btr_) {
            c_ = prod(a_,trans(b_));
        }
        else if(Atr_ && !Btr_) {
            c_ = prod(trans(a_),b_);
        }
        else { // Atr && Btr
            c_ = prod(trans(a_),trans(b_));
        }
    }
    virtual void sync() {
        viennacl::backend::finish();
    }
    virtual void copy_back() {
        matrix_helper C(const_cast<float *>(c_host_),M_,N_);
        viennacl::copy(c_,C);
    }
private:
    int M_,N_,K_;
    bool Atr_;
    bool Btr_;
    viennacl::matrix<float,viennacl::row_major> a_,b_,c_;
    float *c_host_;
};

sgemm_base *get_viennacl() { return new sgemm_viennacl(); };

