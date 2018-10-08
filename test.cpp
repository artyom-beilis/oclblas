#include "sgemm_base.h"
#include <unistd.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <memory>
#include <cblas.h>
#include <math.h>
#include <string.h>


sgemm_base *get_cpu();
sgemm_base *get_cublas();
sgemm_base *get_clblast();
sgemm_base *get_clblas();
sgemm_base *get_viennacl();
sgemm_base *get_my();
sgemm_base *get_mycuda();

sgemm_base *get(std::string const &name)
{
    if(name == "cpu")
        return get_cpu();
    if(name == "cublas")
        return get_cublas();
    if(name == "clblast")
        return get_clblast();
    if(name == "clblas")
        return get_clblas();
    if(name == "viennacl")
        return get_viennacl();
    if(name == "my")
        return get_my();
    if(name == "mycuda")
        return get_mycuda();
    return nullptr;
}

void check(float *A,float *B,float *C,int M,int N,int K,bool Atr,bool Btr)
{
    std::vector<float> c_tmp(M*N);
    cblas_sgemm(CblasRowMajor,(Atr ? CblasTrans : CblasNoTrans), (Btr ? CblasTrans : CblasNoTrans),
            M,N,K,
            1.0f,
            A,(!Atr?K:M),
            B,(!Btr?N:K),
            0.0f,
            c_tmp.data(),N);
    for(int i=0;i<M*N;i++) {
        if(fabsf(c_tmp[i] - C[i]) > 1e-4f) {
            std::cerr << "ERROR AT: " << i << " " << int(c_tmp[i]) << " " << int(C[i]) << std::endl;
            {
                int pos = 0;
                for(int r=0;r<M;r++) {
                   for(int c=0;c<N;c++,pos++) 
                       printf("%d,",(fabsf(c_tmp[pos] - C[pos]) > 1e-4f));
                   printf("\n");
                }
                //throw std::runtime_error("CALC");
                    
            }
            int p=0;
            {
                printf("---------A-------\n") ;
               for(int r=0;r<M;r++) {
                   for(int c=0;c<K;c++) 
                       printf("%10d ",int(A[r*K+c]));
                    printf("\n");
                }
                    
            }
            {
                printf("---------B-------\n") ;
               for(int r=0;r<K;r++) {
                   for(int c=0;c<N;c++) 
                       printf("%10d ",int(B[r*N+c]));
                    printf("\n");
                }
                    
            }
                printf("-----------------\n") ;


            for(int r=0;r<M;r++) {
                for(int c=0;c<N;c++) {
                    printf("%10d %10d,",int(c_tmp[p]),int(C[p]));
                    ++p;
                }
                printf("\n");
            }
            throw std::runtime_error("Bad Calc");
        }
    }
}

int main(int argc,char **argv)
{
    int M = 4;
    int K = 8;
    int N = 2;
    int iters = 100;
    int skip  = 100;
    bool atr=false,btr=false;
    int opt;
    bool sync=false;
    bool sim=false;
    std::string mode="cpu";
    bool do_check = false;
    while((opt=getopt(argc,argv,"m:n:k:i:a:b:v:csw:S"))!=-1) {
        switch(opt) {
        case 'm' : M = atoi(optarg); break;
        case 'n' : N = atoi(optarg); break;
        case 'k' : K = atoi(optarg); break;
        case 'i' : iters = atoi(optarg); skip = std::max(iters / 10,1); break;
        case 'w' : skip = atoi(optarg); break;
        case 's' : sync=true; break;
        case 'a' : atr = atoi(optarg); break;
        case 'S' : sim = true; break;
        case 'b' : btr = atoi(optarg); break;
        case 'v' : mode = optarg; break;
        case 'c' : do_check = true; break;
        case '?' : return 1; 
        }
    }

    double flops = double(M)*N*(K + K - 1) * iters;
    std::unique_ptr<sgemm_base> gemm(get(mode));
    if(!gemm.get()) {
        std::cerr << "Invalid mode "  << mode << std::endl;
        return 1;
    }
    std::vector<float> A(M*K);
    std::vector<float> B(K*N);
    std::vector<float> C(M*N);
    memset(A.data(),0,4*M*K);
    memset(B.data(),0,4*K*N);
    memset(C.data(),0,4*M*N);
    int max_dim = std::max(std::max(M,K),N);
    int max_val = std::min(std::max(max_dim,10),int(sqrt(16000000/max_dim)));
    for(int i=0;i<M;i++) {
        for(int j=0;j<K;j++) {
            if(sim) {
                A[i*K+j]=2.0f;
            }
            else {
                A[i*K+j] = 1.0f * (rand() % max_val);
            }
        }
    }
    for(int i=0;i<K;i++) {
        for(int j=0;j<N;j++) {
            if(sim) {
                B[i*N+j]=3.0f;
            }
            else {
                B[i*N+j] = 1.0f * (rand() % max_val);
            }
        }
    }
    gemm->config(M,N,K,atr,btr);
    gemm->set_A(A.data());
    gemm->set_B(B.data());
    gemm->set_C(C.data());

    auto start = std::chrono::system_clock::now();
    for(int i=-skip;i<iters;i++) {
        if(i==0) {
            gemm->sync();
            if(do_check) {
                gemm->copy_back();
                check(A.data(),B.data(),C.data(),M,N,K,atr,btr);
            }
            start = std::chrono::system_clock::now(); 
        }
        gemm->calc();
        if(sync)
            gemm->sync();
    }
    gemm->sync();
    auto stop = std::chrono::system_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::duration<double> > ((stop-start)).count();
    std::cout << (flops / duration * 1e-9)  << " GFLOPS " << (duration * 1e+3) / iters << " ms"<< std::endl;
}
