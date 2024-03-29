#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include "conv_base.h"
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>

#include <CL/cl2.hpp>

class conv_winograd : public conv_base {
public:

	cl::Platform platform_;
	cl::Device device_;
	cl::Context context_;
	cl::CommandQueue queue_;
    cl::Program prog_;
    cl::Kernel tiles_conv_,kernel_conv_,win_conv_;

	cl::Buffer buf_in_,buf_out_,buf_kern_;
    cl::Buffer buf_tiled_in_,buf_conv_kern_;

    std::vector<double> total_times_;
    int counts_;

	size_t ws_size_;
	float *host_out_;
    
    std::string get_kernel()
    {
        #ifndef MYKERNEL_PATH
        #define MYKERNEL_PATH 
        #endif
        std::ifstream tmp(MYKERNEL_PATH "conv_winograd_t2.cl");
        std::ostringstream ss;
        ss << tmp.rdbuf();
        return ss.str();
    }

    virtual void warmup_done()
    {
        total_times_.assign(0,total_times_.size());
        counts_ = 0;
    }
    
    void make_param(int &v,char const *env,char const *name,std::string &s,std::string &log)
    {
        if(getenv(env)!=0) 
            v=atoi(getenv(env));
        s+="#define ";
        s+=name;
        s+=" ";
        s+=std::to_string(v);
        s+="\n";
        log+=name + ("=" + (std::to_string(v) + " "));
    }

    void build()
    {
        std::string header,log;
        tiles_in_wg_ = 32;
        kerns_in_wg_ = 32;
        wgdim_t_ = 8;
        wgdim_k_ = 8;
        so4_ = 0;
        //make_param(tiles_in_wg_,"T_INWG","TILES_IN_WG",header,log);
        //make_param(kerns_in_wg_,"K_INWG","KERNELS_IN_WG",header,log);
        //make_param(wgdim_t_,"WGDIM_T","WG_DIM_TILES",header,log);
        //make_param(wgdim_k_,"WGDIM_K","WG_DIM_KERNELS",header,log);
        //make_param(so4_,"SO4","SUM_OF_4",header,log);
        block_t_ = tiles_in_wg_/wgdim_t_;
        block_k_ = kerns_in_wg_/wgdim_k_;
        std::string src = header + get_kernel();
        std::ofstream tmp("/tmp/out.cl");
        tmp << src;
        prog_ = std::move(cl::Program(context_,src.c_str()));
        int rc;
        std::vector<cl::Device> devices(1,device_);
        char const *extra = "";
        if(getenv("SAVE_TEMPS")) {
            extra = "-save-temps=./ ";
        }
        try {
            rc=prog_.build(devices,extra);
        }
        catch(...) {
            size_t len=0;
            static char buffer[16384*32];
            clGetProgramBuildInfo(prog_(), devices[0](), CL_PROGRAM_BUILD_LOG, sizeof(buffer)-1, buffer, &len);
            buffer[std::min(len,sizeof(buffer)-1)]=0;
            std::cout << "LOG: rc="<<rc <<"\n" << buffer << std::endl;
            if(rc != 0)
                throw std::runtime_error("Failed to build");
        }
        kernel_conv_ = std::move(cl::Kernel(prog_,"winconv_calc_gkgt_3x3"));

        //tiles_conv_ = std::move(cl::Kernel(prog_,"winconv_im2tile_4x4_x4"));
        //tiles_conv_ = std::move(cl::Kernel(prog_,"winconv_im2tile_4x4"));

        win_conv_ = std::move(cl::Kernel(prog_,"winconv_3x3"));
        
        /// Query binary (PTX file) size
        size_t bin_sz;
        rc = clGetProgramInfo(prog_(), CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &bin_sz, NULL);

        // Read binary (PTX file) to memory buffer
        std::vector<unsigned char> bin(bin_sz+1);
        unsigned char *ptr = &bin[0];
        rc = clGetProgramInfo(prog_(), CL_PROGRAM_BINARIES, sizeof(unsigned char *), &ptr, NULL);
        std::ofstream ptx("conv.ptx");
        ptx.write((char *)(&bin[0]),bin_sz);
        ptx.close();
    }
    	
	conv_winograd(int platform,int device)  
	{
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		std::vector<cl::Device> devices;
		platform_ = platforms[platform];
		platform_.getDevices(CL_DEVICE_TYPE_ALL, &devices);
		device_ = devices[device];
		auto device_as_vector = std::vector<cl::Device>{device_};
		context_ = cl::Context(device_as_vector);
		queue_ = cl::CommandQueue(context_, device_,CL_QUEUE_PROFILING_ENABLE);
        build();
	}

    ~conv_winograd()
    {
        if(counts_ != 0) {
            printf("Times ms\n");
            double sum = 0;
            for(unsigned i=0;i<total_times_.size();i++) {
                printf("%10.3f,",total_times_[i] / counts_);
                sum+=total_times_[i];
            }
            printf("\nRalative %%\n");
            for(unsigned i=0;i<total_times_.size();i++) {
                printf("%5.2f%%,",total_times_[i]/sum*100.0);
            }
            printf("\n");
        }
    }

	
	cl::Buffer dalloc(size_t n)
	{
		if(n == 0)
			return cl::Buffer();
		else
			return cl::Buffer(context_, CL_MEM_READ_WRITE, n);
	}
	void h2d(cl::Buffer &buf,void const *p,size_t n)
	{
        queue_.enqueueWriteBuffer(buf, CL_TRUE, 0, n,p);
	}
	void d2h(cl::Buffer &buf,void *p,size_t n)
	{
        queue_.enqueueReadBuffer(buf,CL_TRUE,0,n,p);
	}

	virtual void config(conv_param const &param,int B,int C,int H,int W)
	{
		conv_base::config(param,B,C,H,W);
        if(param.stride_h != 1 || param.stride_w != 1 || param.kernel_w != 3 || param.kernel_h != 3 || param.pad_h != 1 || param.pad_w != 1)
            throw std::runtime_error("Unsupported");

        buf_in_ =   std::move(dalloc( b_*c_*h_*w_*sizeof(float)));
        htiles_ = (out_h_ + 1) / 2;
        wtiles_ = (out_w_ + 1) / 2;
        int tiled_size = b_*c_*htiles_*wtiles_*16 * sizeof(float);
        buf_tiled_in_ = std::move(dalloc(tiled_size));

        buf_out_ =  std::move(dalloc( b_*out_c_*out_h_*out_w_*sizeof(float)));
		buf_kern_ = std::move(dalloc( par_.num_outputs*c_*par_.kernel_h*par_.kernel_w*sizeof(float)));
        buf_conv_kern_ = std::move(dalloc(par_.num_outputs*c_*par_.kernel_h*par_.kernel_w*sizeof(float)*16/9));

	}
	virtual void set_kernel(float const *A)
	{
		h2d(buf_kern_,A,par_.num_outputs*c_*par_.kernel_h*par_.kernel_w*sizeof(float));
	}
	virtual void set_input(float const *A) 
    {
           h2d(buf_in_, A, b_*c_*h_*w_*sizeof(float));
	}
	virtual void set_output(float *outp) { host_out_ = outp; }

    int align_up(int v,int gran)
    {
        return (v+gran-1)/gran*gran;
    }
    void calc_new()
    {
        std::vector<cl::Event> ev(2);
        int ind=0;
        kernel_conv_.setArg(ind++,out_c_);
        kernel_conv_.setArg(ind++,c_);
        kernel_conv_.setArg(ind++,buf_kern_);
        kernel_conv_.setArg(ind++,buf_conv_kern_);
        cl::NDRange gk(align_up(out_c_,8),align_up(c_,8));
        cl::NDRange gl(8,8);
        //cl::NDRange gk(out_c_,c_);
        //cl::NDRange gl = cl::NullRange;
        queue_.enqueueNDRangeKernel(kernel_conv_,cl::NullRange,gk,gl,nullptr,&ev[0]);
        
        ind=0;
        win_conv_.setArg(ind++,b_);
        win_conv_.setArg(ind++,out_c_);
        win_conv_.setArg(ind++,c_);
        win_conv_.setArg(ind++,out_h_);
        win_conv_.setArg(ind++,out_w_);
        win_conv_.setArg(ind++,buf_in_);
        win_conv_.setArg(ind++,buf_conv_kern_);
        win_conv_.setArg(ind++,buf_out_);

        int tile_dim = (htiles_*wtiles_ + block_t_ - 1) / block_t_;
        int kern_dim = (out_c_ + block_k_ - 1) / block_k_;
        cl::NDRange loc(256,1);
        int tiles = ((out_h_ + 1) / 2 * (out_w_ + 1) / 2 * b_ + 31)/32;
        cl::NDRange glob(tiles * 256,(out_c_ + 31) / 32);
        queue_.enqueueNDRangeKernel(win_conv_,cl::NullRange,glob,loc,nullptr,&ev[1]);
        if(getenv("PROF")) {
            const int N = ev.size();
            total_times_.resize(N);
            cl::Event::waitForEvents(ev);
            cl_ulong start=0,stop=0;

            for(int i=0;i<int(ev.size());i++)  {
                clGetEventProfilingInfo(ev[i](),CL_PROFILING_COMMAND_START,sizeof(start),&start,0);
                clGetEventProfilingInfo(ev[i](),CL_PROFILING_COMMAND_END,sizeof(stop),&stop,0);
                total_times_[i] += (stop - start) * 1e-6;
            }
            counts_++;
        }
    }
	virtual void calc() {
        try {
            calc_new();
        }
        catch(cl::Error const &e) {
            std::cerr << "clError"  << e.what() << " "  << e.err() << std::endl;;
            throw;
        }
    }
	void calc_old() {
        std::vector<cl::Event> ev(3);
		float alpha=1.0f;
		float beta=0.0f;
        //buf_in_ =   std::move(dalloc( b_*c_*h_*w_*sizeof(float)));
        //buf_out_ =  std::move(dalloc( b_*out_c_*out_h_*out_w_*sizeof(float)));
		//buf_kern_ = std::move(dalloc( par_.num_outputs*c_*par_.kernel_h*par_.kernel_w*sizeof(float)));
        auto queue_plain = queue_();
        int ind;
        ind = 0;
        tiles_conv_.setArg(ind++,b_*c_);
        tiles_conv_.setArg(ind++,h_);
        tiles_conv_.setArg(ind++,w_);
        tiles_conv_.setArg(ind++,par_.pad_h);
        tiles_conv_.setArg(ind++,par_.pad_w);
        tiles_conv_.setArg(ind++,htiles_);
        tiles_conv_.setArg(ind++,wtiles_);
        tiles_conv_.setArg(ind++,buf_in_);
        tiles_conv_.setArg(ind++,buf_tiled_in_);

        cl::NDRange tg = cl::NDRange(b_*c_ * 4,align_up(htiles_,8),align_up(wtiles_,8));
        //cl::NDRange tg = cl::NDRange(align_up(b_*c_,4),align_up(htiles_,8),align_up(wtiles_,8));
        cl::NDRange lg = cl::NDRange(4,8,8);
        
        //cl::NDRange tg = cl::NDRange(b_*c_,htiles_,wtiles_);
        //cl::NDRange lg = cl::NullRange;

        queue_.enqueueNDRangeKernel(tiles_conv_,
                                        cl::NullRange,
                                        tg,
                                        lg,nullptr,&ev[0]);
        ind=0;
        kernel_conv_.setArg(ind++,out_c_);
        kernel_conv_.setArg(ind++,c_);
        kernel_conv_.setArg(ind++,buf_kern_);
        kernel_conv_.setArg(ind++,buf_conv_kern_);
        cl::NDRange gk(align_up(out_c_,8),align_up(c_,8));
        cl::NDRange gl(8,8);
        //cl::NDRange gk(out_c_,c_);
        //cl::NDRange gl = cl::NullRange;
        queue_.enqueueNDRangeKernel(kernel_conv_,cl::NullRange,gk,gl,nullptr,&ev[1]);
        
        ind=0;
        win_conv_.setArg(ind++,b_);
        win_conv_.setArg(ind++,c_);
        win_conv_.setArg(ind++,out_c_);
        win_conv_.setArg(ind++,out_h_);
        win_conv_.setArg(ind++,out_w_);
        win_conv_.setArg(ind++,htiles_);
        win_conv_.setArg(ind++,wtiles_);
        win_conv_.setArg(ind++,buf_tiled_in_);
        win_conv_.setArg(ind++,buf_conv_kern_);
        win_conv_.setArg(ind++,buf_out_);

        #ifndef USE_1_OUTPUT_PER_THREAD
        int tile_dim = (htiles_*wtiles_ + block_t_ - 1) / block_t_;
        int kern_dim = (out_c_ + block_k_ - 1) / block_k_;
        cl::NDRange loc(1,wgdim_t_,wgdim_k_);
        cl::NDRange glob(b_,align_up(tile_dim,wgdim_t_),align_up(kern_dim,wgdim_k_));
        #else
        cl::NDRange glob(b_,align_up(htiles_*wtiles_),align_up(out_c_,kerns_in_wg_));
        cl::NDRange loc(1,tiles_in_wg_,kerns_in_wg_);
        #endif
        queue_.enqueueNDRangeKernel(win_conv_,cl::NullRange,glob,loc,nullptr,&ev[2]);
        if(getenv("PROF")) {
            const int N = ev.size();
            total_times_.resize(N);
            cl::Event::waitForEvents(ev);
            cl_ulong start=0,stop=0;

            for(int i=0;i<sizeof(ev)/sizeof(ev[0]);i++)  {
                clGetEventProfilingInfo(ev[i](),CL_PROFILING_COMMAND_START,sizeof(start),&start,0);
                clGetEventProfilingInfo(ev[i](),CL_PROFILING_COMMAND_END,sizeof(stop),&stop,0);
                total_times_[i] += (stop - start) * 1e-6;
            }
            counts_++;
        }

	}
	virtual void sync() {
        	queue_.finish();
	}
	virtual void copy_back() {
        	d2h(buf_out_,host_out_,b_*out_w_*out_h_*out_c_*sizeof(float));
	}
private:
    int htiles_,wtiles_;
    int tiles_in_wg_,kerns_in_wg_,wgdim_t_,wgdim_k_,so4_,block_k_,block_t_;
};


conv_base *get_conv_winograd(int p,int d) { return new conv_winograd(p,d); };

conv_base *get_external(int p,int d) { return new conv_winograd(p,d); };

