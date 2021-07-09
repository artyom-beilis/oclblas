// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#ifndef CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_ENABLE_EXCEPTIONS
#endif
#ifndef CL_HPP_MINIMUM_OPENCL_VERSION
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#endif
#ifndef CL_HPP_TARGET_OPENCL_VERSION
#define CL_HPP_TARGET_OPENCL_VERSION 120
#endif

#include <CL/cl2.hpp>
#include <math.h>
#include "sgemm_base.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sstream>

#define BLOCK_SIZE 8

class sgemm_my : public sgemm_base {
public:
    cl::Platform platform_;
    cl::Device device_;
    cl::Context context_;
    cl::CommandQueue queue_;
    cl::Program program_;
    cl::Kernel kernel_;
    int block_size_n_;
    int block_size_m_;
    int tile_size_m_;
    int tile_size_n_;
    int tile_size_k_;
    int off_;
    int zorder_;
    sgemm_my(int p,int d)  
    {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        std::vector<cl::Device> devices;
        platform_ = platforms[p];
        platform_.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        device_ = devices[d];
        auto device_as_vector = std::vector<cl::Device>{device_};
        context_ = cl::Context(device_as_vector);
        queue_ = cl::CommandQueue(context_, device_);
    }
    virtual void flush_cache()
    {
    }
    virtual ~sgemm_my() { 

    }

    void subst(int &v,char const *s)
    {
        if(!getenv(s))
            return;
        v=atoi(getenv(s));
    }

    unsigned align_to_top2n(unsigned x)
    {
        unsigned v=1;
        while(x > v) {
            v<<= 1;
        }
        return v;
    }
    virtual void config(int M,int N,int K,bool Atr,bool Btr)
    {
        M_= M;
        N_= N;
        K_= K;
        Atr_ = Atr;
        Btr_ = Btr;
        a_ = std::move(cl::Buffer(context_, CL_MEM_READ_WRITE, M*K*sizeof(float)));
        b_ = std::move(cl::Buffer(context_, CL_MEM_READ_WRITE, K*N*sizeof(float)));
        c_ = std::move(cl::Buffer(context_, CL_MEM_READ_WRITE, M*N*sizeof(float)));


        std::ostringstream opts;

        if(M >= 512 && N >= 512) {
            tile_size_m_ = 128;
            tile_size_n_ = 128;
            block_size_m_ = 8;
            block_size_n_ = 8;
            tile_size_k_ = 16;
            off_ = 1;
        }
        else if(M >= 128 && N>= 128) {
            tile_size_m_ = 64;
            tile_size_n_ = 64;
            block_size_m_ = 8;
            block_size_n_ = 8;
            tile_size_k_ = 16;
            off_ = 1;
        }
        else {
            tile_size_m_ = 32;
            tile_size_n_ = 32;
            block_size_m_ = 4;
            block_size_n_ = 4;
            tile_size_k_ = 32;
            off_ = 0;
        }
        
        subst(tile_size_m_,"TILE_SIZE_M");
        subst(tile_size_n_,"TILE_SIZE_N");
        subst(block_size_m_,"BLOCK_M");
        subst(block_size_n_,"BLOCK_N");
        subst(off_,"TILE_OFFSET");
        subst(tile_size_k_,"TILE_SIZE_K");
        subst(zorder_,"ZORDER");
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

        block_size_n_ = std::min(tile_size_n_,block_size_n_);
        block_size_m_ = std::min(tile_size_m_,block_size_m_);


        if(getenv("SAVE_TEMPS"))
          opts << " -save-temps=./ ";
        opts << " -DTILE_SIZE_M=" << tile_size_m_ << " -DTILE_SIZE_N=" << tile_size_n_ << " -DBLOCK_SIZE_N=" << block_size_n_ << " -DBLOCK_SIZE_M=" << block_size_m_ << " -DTILE_SIZE_K="<<tile_size_k_ << " -DTILE_OFFSET="<<off_ << " -DZORDER=" << zorder_;

        if(Btr_)
            opts << " -DBTRANS";
        if(Atr_)
            opts << " -DATRANS";

        std::cerr << opts.str() << std::endl;
        
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
//        cl::Program::Sources sources(1,std::make_pair(src.c_str(),src.size()));
        program_ = std::move(cl::Program(context_,src.c_str()));
        
        std::vector<cl::Device> devices { device_ };
        try {
            program_.build(devices,opts.str().c_str());
        }
        catch(cl::BuildError const &e) {
            std::string log;
            auto cl_log = e.getBuildLog();
            for(size_t i=0;i<cl_log.size();i++) {
                log += "For device: ";
                log += cl_log[i].first.getInfo<CL_DEVICE_NAME>();
                log += "\n";
                log += cl_log[i].second;
            }
            std::cerr << log << std::endl;
            throw std::runtime_error("Build Failed");
        }

        /// Query binary (PTX file) size
        size_t bin_sz;
        int rc = clGetProgramInfo(program_(), CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &bin_sz, NULL);

        // Read binary (PTX file) to memory buffer
        std::vector<unsigned char> bin(bin_sz+1);
        unsigned char *ptr = &bin[0];
        rc = clGetProgramInfo(program_(), CL_PROGRAM_BINARIES, sizeof(unsigned char *), &ptr, NULL);
        std::ofstream ptx("out.ptx");
        ptx.write((char *)(&bin[0]),bin_sz);
        ptx.close();


        kernel_ = std::move(cl::Kernel(program_, "sgemm"));
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
    static int round_up_div(int x,int y)
    {
        return (x + y - 1)/y;
    }
    virtual void calc()
    {
        int rc=0;
        int ind=0;
        kernel_.setArg(ind++,M_);
        kernel_.setArg(ind++,N_);
        kernel_.setArg(ind++,K_);
        kernel_.setArg(ind++,a_);
        kernel_.setArg(ind++,(!Atr_ ? K_ : M_ ));
        kernel_.setArg(ind++,b_);
        kernel_.setArg(ind++,(!Btr_ ? N_ : K_ ));
        kernel_.setArg(ind++,c_);
        kernel_.setArg(ind++,N_);
       
        cl::NDRange g;

        
        int ls0 = tile_size_m_ / block_size_m_;
        int ls1 = tile_size_n_ / block_size_n_; 
        int grid_m = round_up_div(M_,tile_size_m_);
        int grid_n = round_up_div(N_,tile_size_n_);
        if(zorder_ == 1) {
            int grid = align_to_top2n(std::max(grid_m,grid_n));
            grid_m = grid_n = grid;
        }
        int gs0 = grid_m * ls0;
        int gs1 = grid_n * ls1;
        g=cl::NDRange(gs0,gs1);
        cl::NDRange local(ls0,ls1);
        rc=queue_.enqueueNDRangeKernel(kernel_, cl::NullRange, g,local,nullptr,nullptr);

        if(rc!=0)
            throw std::runtime_error("Failed to enqueue the kernel: " + std::to_string(rc));

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
    cl::Buffer a_,b_,c_,tmp1_,tmp2_;
    float *c_host_;
};

sgemm_base *get_my(int p,int d) { return new sgemm_my(p,d); };
sgemm_base *get_external(int p,int d) { return new sgemm_my(p,d); };

