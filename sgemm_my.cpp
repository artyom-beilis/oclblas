#include <CL/cl.hpp>
#include <math.h>
#include "sgemm_base.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sstream>

#define __CL_ENABLE_EXCEPTIONS

#define BLOCK_SIZE 8

class sgemm_my : public sgemm_base {
public:
    cl::Platform platform_;
    cl::Device device_;
    cl::Context context_;
    cl::CommandQueue queue_;
    cl::Program program_;
    cl::Kernel kernel_;
    int block_size_x_;
    int block_size_y_;
    int tile_size_;
    int tile_size_k_;
    sgemm_my()  
    {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        std::vector<cl::Device> devices;
        platform_ = platforms[0];
        platform_.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        device_ = devices[0];
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
    virtual void config(int M,bool Atr,bool Btr)
    {

        M_= M;
        Atr_ = Atr;
        Btr_ = Btr;
        a_ = std::move(cl::Buffer(context_, CL_MEM_READ_WRITE, M*M*sizeof(float)));
        b_ = std::move(cl::Buffer(context_, CL_MEM_READ_WRITE, M*M*sizeof(float)));
        c_ = std::move(cl::Buffer(context_, CL_MEM_READ_WRITE, M*M*sizeof(float)));


        std::ostringstream opts;

	tile_size_ = 32;
	block_size_y_ = 4;
	block_size_x_ = 4;

	subst(tile_size_,"TILE_SIZE");
	subst(block_size_y_,"BLOCK_Y");
	subst(block_size_x_,"BLOCK_X");
    tile_size_k_ = tile_size_;
    subst(tile_size_k_,"TILE_SIZE_K");
    int bin_y = tile_size_ /block_size_y_;
    int bin_x = tile_size_ /block_size_x_;
    if(tile_size_k_ % bin_y != 0 || tile_size_k_ % bin_x !=0) {
        std::cerr <<"FIXING TILE SIZE!!" << std::endl;
        tile_size_k_ = tile_size_;
    }

	tile_size_ = std::min(M_,tile_size_);
	block_size_x_ = std::min(tile_size_,block_size_x_);
	block_size_y_ = std::min(tile_size_,block_size_y_);

        opts << " -DTILE_SIZE=" << tile_size_ << " -DBLOCK_SIZE_X=" << block_size_x_ << " -DBLOCK_SIZE_Y=" << block_size_y_ << " -DTILE_SIZE_K="<<tile_size_k_;

        if(Btr_)
            opts << " -DBTRANS";
        if(Atr_)
            opts << " -DATRANS";

        std::cerr << opts.str() << std::endl;
        
        std::ifstream tmp;
        if(getenv("GEN")) {
                remove("k.cl");
            char call[256];
            snprintf(call,sizeof(call),"php5 generator.php %d %d %d %d >k.cl",tile_size_,tile_size_k_,block_size_x_,block_size_y_);
            if(system(call)!=0) {
                throw std::runtime_error("System failed " + std::string(call));
            }
            tmp.open("k.cl");
        }
        else
            tmp.open("gemm.cl");
        
        if(!tmp) {
            throw std::runtime_error("Failed to open k.cl");
        }
        std::ostringstream ss;
        ss << tmp.rdbuf();
        std::string src=ss.str();
        cl::Program::Sources sources(1,std::make_pair(src.c_str(),src.size()));
        program_ = std::move(cl::Program(context_,sources));
        
        std::vector<cl::Device> devices { device_ };
        int rc = program_.build(devices,opts.str().c_str());
        if(rc!=0){
                size_t len=0;
                static char buffer[16384*32];
                clGetProgramBuildInfo(program_(), device_(), CL_PROGRAM_BUILD_LOG, sizeof(buffer)-1, buffer, &len);
                buffer[std::min(len,sizeof(buffer)-1)]=0;
                if(rc!=0)
                    throw std::runtime_error("Failed to build program, log:\n" + std::string(buffer));
                //if(len!=0)
                //    std::cerr << "LOG: " << buffer << std::endl;
        }

        /// Query binary (PTX file) size
        size_t bin_sz;
        rc = clGetProgramInfo(program_(), CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &bin_sz, NULL);

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
        queue_.enqueueWriteBuffer(a_, CL_TRUE, 0, M_*M_*sizeof(float), A);
    }
    virtual void set_B(float const *B) { 
        queue_.enqueueWriteBuffer(b_, CL_TRUE, 0, M_*M_*sizeof(float), B);
    }
    virtual void set_C(float *C) { 
        c_host_ = C;
    }
    virtual void calc()
    {
        int rc=0;
        int ind=0;
        kernel_.setArg(ind++,M_);
        kernel_.setArg(ind++,a_);
        kernel_.setArg(ind++,b_);
        kernel_.setArg(ind++,c_);
        
        cl::NDRange global(M_/block_size_y_,M_/block_size_x_);
        cl::NDRange local(tile_size_ / block_size_y_,tile_size_ / block_size_x_);
        rc=queue_.enqueueNDRangeKernel(kernel_, cl::NullRange, global,local,nullptr,nullptr);

        if(rc!=0)
            throw std::runtime_error("Failed to enqueue the kernel: " + std::to_string(rc));
    }
    virtual void sync() {
        queue_.finish();
    }
    virtual void copy_back() {
        queue_.enqueueReadBuffer(c_,CL_TRUE,0,M_*M_*sizeof(float),c_host_);
    }
private:
    int M_;
    bool Atr_;
    bool Btr_;
    cl::Buffer a_,b_,c_,tmp1_,tmp2_;
    float *c_host_;
};

sgemm_base *get_my() { return new sgemm_my(); };

