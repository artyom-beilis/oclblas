#include <nvrtc.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <memory>

#include "sgemm_base.h"

#define NVRTC_SAFE_CALL(x)                                        \
  do {                                                            \
    nvrtcResult result = x;                                       \
    if (result != NVRTC_SUCCESS) {                                \
      std::cerr << "\nerror: " #x " failed with error "           \
                << nvrtcGetErrorString(result) << '\n';           \
      exit(1);                                                    \
    }                                                             \
  } while(0)

#define CUDA_SAFE_CALL(x)                                         \
  do {                                                            \
    CUresult result = x;                                          \
    if (result != CUDA_SUCCESS) {                                 \
      const char *msg;                                            \
      cuGetErrorName(result, &msg);                               \
      std::cerr << "\nerror: " #x " failed with error "           \
                << msg << '\n';                                   \
      exit(1);                                                    \
    }                                                             \
  } while(0)

class cuprog {
public:
    cuprog(char const *src,std::vector<std::string> const &defs)
    {
        nvrtcProgram prog;
        NVRTC_SAFE_CALL(
                nvrtcCreateProgram(&prog,         // prog
                    src,         // buffer
                    "gemm.cl",    // name
                    0,             // numHeaders
                    NULL,          // headers
                    NULL));        // includeNames
        // Compile the program for compute_30 with fmad disabled.
        std::vector<std::string> opts = {"-DOCL_TO_CU","--gpu-architecture=compute_52"};
        std::vector<char const *> copts;
        for(std::string const &v : defs)
            opts.push_back("-D"+v);
        for(std::string const &v: opts) {
            std::cerr << " " << v;
            copts.push_back(v.c_str());
        }
        std::cerr << std::endl;
        nvrtcResult compileResult = nvrtcCompileProgram(prog,  // prog
                copts.size(),     // numOptions
                &copts[0]); // options
        // Obtain compilation log from the program.
        size_t logSize;
        NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
        char *log = new char[logSize];
        NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));
        std::cerr << log << '\n';
        delete[] log;
        if (compileResult != NVRTC_SUCCESS) {
            exit(1);
        }
        // Obtain PTX from the program.
        size_t ptxSize;
        NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
        char *ptx = new char[ptxSize];
        NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));
	std::ofstream ff("out_cu.ptx");
	ff.write(ptx,ptxSize);
        // Destroy the program.
        NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
        // Load the generated PTX and get a handle to the SAXPY kernel.
        CUDA_SAFE_CALL(cuInit(0));
        CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
        CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
        CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
        CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "sgemm"));
	int regs=-1;
	CUDA_SAFE_CALL(cuFuncGetAttribute(&regs,CU_FUNC_ATTRIBUTE_NUM_REGS,kernel));
	fprintf(stderr,"%s numRegs=%d\n","sgemm",regs);
  }
  void call(int wg[2],int lgw[2],int M,int N,int K,float *A,int lda,float *B,int ldb,float *C,int ldc)
  {
      void *args[] = { &M, &N, &K, &A,&lda,&B,&ldb, &C, &ldc };
      CUDA_SAFE_CALL(
              cuLaunchKernel(kernel,
                  wg[0]/lgw[0], wg[1]/lgw[1], 1,    // grid dim
                  lgw[0]      , lgw[1]      , 1,   // block dim
                  0, NULL,             // shared mem and stream
                  args, 0));           // arguments

  }
  ~cuprog()
  {
      CUDA_SAFE_CALL(cuModuleUnload(module));
      CUDA_SAFE_CALL(cuCtxDestroy(context));
  }

private:
  CUdevice cuDevice;
  CUcontext context;
  CUmodule module;
  CUfunction kernel;

};

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
    static int round_up_div(int x,int y)
    {
        return (x + y - 1)/y;
    }
    void subst(int &v,char const *s)
    {
        if(!getenv(s))
            return;
        v=atoi(getenv(s));
    }
    virtual void config(int M,int N,int K,bool Atr,bool Btr)
    {
        M_= M;
        N_= N;
        K_= K;
        Atr_ = Atr;
        Btr_ = Btr;
        int off;
        
        cudaMalloc((void **)&a_,M*K*sizeof(float));
        cudaMalloc((void **)&b_,K*N*sizeof(float));
        cudaMalloc((void **)&c_,M*N*sizeof(float));
	

        std::vector<std::string> opts;

        if(M >= 512 && N >= 512) {
            tile_size_m_ = 128;
            tile_size_n_ = 128;
            block_size_m_ = 8;
            block_size_n_ = 8;
            tile_size_k_ = 16;
            off = 1;
        }
        else if(M >= 128 && N>= 128) {
            tile_size_m_ = 64;
            tile_size_n_ = 64;
            block_size_m_ = 8;
            block_size_n_ = 8;
            tile_size_k_ = 16;
            off = 1;
        }
        else {
            tile_size_m_ = 32;
            tile_size_n_ = 32;
            block_size_m_ = 4;
            block_size_n_ = 4;
            tile_size_k_ = 32;
            off = 0;
        }



	subst(tile_size_m_,"TILE_SIZE_M");
	subst(tile_size_n_,"TILE_SIZE_N");
	subst(block_size_m_,"BLOCK_M");
	subst(block_size_n_,"BLOCK_N");
	subst(off,"TILE_OFFSET");
    subst(tile_size_k_,"TILE_SIZE_K");
    int wg_size = (tile_size_m_  * tile_size_n_ / block_size_n_ / block_size_m_);
    if(tile_size_m_ % block_size_m_ != 0 || tile_size_n_ % block_size_n_ != 0) {
        std::cerr <<"FIXING TILE SIZE / BS!!" << (tile_size_m_ % block_size_m_) << " " << (tile_size_n_ % block_size_n_) << std::endl;
        throw std::runtime_error("Inv");
    }
    if(tile_size_m_ * tile_size_k_ % wg_size != 0 || tile_size_n_ * tile_size_k_ % wg_size != 0) {
        std::cerr <<"FIXING TILE SIZE!!" << std::endl;
        //tile_size_k_ = tile_size_*tile_size_ / block_size_n_ / block_size_m_;
        throw std::runtime_error("Inv");
    }
    /*if(tile_size_k_ % bin_y != 0 || tile_size_k_ % bin_x !=0) {
        std::cerr <<"FIXING TILE SIZE!!" << std::endl;
        tile_size_k_ = tile_size_;
    }*/

	block_size_n_ = std::min(tile_size_n_,block_size_n_);
	block_size_m_ = std::min(tile_size_m_,block_size_m_);

       opts.push_back("OCL_TO_CU");
        opts.push_back("TILE_OFFSET=" + std::to_string(off));
        opts.push_back("TILE_SIZE_M=" + std::to_string(tile_size_m_));
        opts.push_back("TILE_SIZE_N=" + std::to_string(tile_size_n_));
        opts.push_back("TILE_SIZE_K=" + std::to_string(tile_size_k_));
        opts.push_back("BLOCK_SIZE_M=" + std::to_string(block_size_m_));
        opts.push_back("BLOCK_SIZE_N=" + std::to_string(block_size_n_));
        if(Btr_)
            opts.push_back("BTRANS");
        if(Atr_)
            opts.push_back("ATRANS");

        std::ifstream tmp;
#ifndef MYKERNEL_PATH
#define MYKERNEL_PATH
#endif
        tmp.open(MYKERNEL_PATH  "gemm.cl");

        if(!tmp) {
            throw std::runtime_error("Failed to open gemm.cl");
        }
        std::ostringstream ss;
        ss << tmp.rdbuf();
        std::string src=ss.str();
        prog.reset(new cuprog(src.c_str(),opts));
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
        int ls[2],gs[2];
        ls[0] = tile_size_m_ / block_size_m_;
        ls[1] = tile_size_n_ / block_size_n_; 
        gs[0] = round_up_div(M_,tile_size_m_) * tile_size_m_ / block_size_m_;
        gs[1] = round_up_div(N_,tile_size_n_) * tile_size_n_ / block_size_n_;
        prog->call(gs,ls,M_,N_,K_,a_,(!Atr_ ? K_ : M_ ),b_,(!Btr_ ? N_ : K_ ),c_,N_);
    }
    virtual void sync() {
        cudaDeviceSynchronize();
    }
    virtual void copy_back() {
        cudaMemcpy(c_host_,c_,M_*N_*sizeof(float),cudaMemcpyDeviceToHost);
    }
private:
    int M_,N_,K_;
    bool Atr_;
    bool Btr_;
    float *a_;
    float *b_;
    float *c_;
    float *c_host_;
    int block_size_m_;
    int block_size_n_;
    int tile_size_m_;
    int tile_size_n_;
    int tile_size_k_;
    std::unique_ptr<cuprog> prog;
};

sgemm_base *get_mycuda() { return new sgemm_mycuda(); };
sgemm_base *get_external(int,int) { return get_mycuda(); }

