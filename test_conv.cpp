#include "conv_base.h"
#include <unistd.h>
#include <iostream>
#include <chrono>
#include <assert.h>
#include <vector>
#include <memory>
#include <math.h>
#include <string.h>

conv_base *get_conv_miopen(int p,int d);

conv_base *get(std::string const &name,int plat,int dev)
{
    if(name == "miopen")
        return get_conv_miopen(plat,dev);
    return nullptr;
}


int main(int argc,char **argv)
{
    int Batch=10;
    int Chan=32;
    int Dim = 250;
    int Kernel = 5;
    int sTride = 1;
    int NumOutputs = 64;
    int Groups=1;
    int Pad=0;

    int iters = 100;
    int skip  = 10;
    int plat=0;
    int dev=0;
    int opt;
    bool sync=false;
    std::string mode="miopen";
    bool do_check = false;
    while((opt=getopt(argc,argv,"K:v:p:P:d:B:C:D:N:S:scT:G:"))!=-1) {
        switch(opt) {
	case 'G' : Groups=atoi(optarg); break;
	case 'K' : Kernel=atoi(optarg); break;
	case 'T' : sTride=atoi(optarg); break;
	case 'v' : mode = optarg; break;
	case 'p' : plat=atoi(optarg); break;
	case 'P' : Pad=atoi(optarg); break;
	case 'd' : dev=atoi(optarg); break;
        case 'B' : Batch = atoi(optarg); break;
        case 'C' : Chan = atoi(optarg); break;
        case 'D' : Dim = atoi(optarg); break;
	case 'N' : NumOutputs = atoi(optarg); break;
        case 's' : sync=true; break;
        case 'c' : do_check = true; break;
        case '?' : return 1; 
        }
    }

   
    int out_dim = (Dim+2*Pad - Kernel)/sTride+1;
    double flops = (double(out_dim) * out_dim * NumOutputs * Batch) * (2 * Chan * Kernel * Kernel - 1  ) * iters;
    std::unique_ptr<conv_base> conv(get(mode,plat,dev));
    if(!conv.get()) {
        std::cerr << "Invalid mode "  << mode << std::endl;
        return 1;
    }
    std::vector<float> vA(Batch*Dim*Dim*Chan);
    std::vector<float> vB(Kernel*Kernel*Chan*NumOutputs);
    std::vector<float> vC(out_dim*out_dim*NumOutputs*Batch);
    
    memset(vA.data(),0,4*vA.size());
    memset(vB.data(),0,4*vB.size());
    memset(vC.data(),0,4*vC.size());

    conv_param par;
    par.kernel_w = par.kernel_h = Kernel;
    par.num_outputs = NumOutputs;
    par.stride_w = par.stride_h = sTride;
    par.pad_w = par.pad_h = Pad;
    par.groups = Groups;
    conv->config(par,Batch,Chan,Dim,Dim);
    conv->set_input(vA.data());
    conv->set_kernel(vB.data());
    conv->set_output(vC.data());

    auto shape = conv->get_out_shape();
    fprintf(stderr,"Input %d:%d:%d:%d, Output %d:%d:%d:%d Kernel=%d, Stride=%d, Pad=%d\n",
    			Batch,Chan,Dim,Dim, shape[0],shape[1],shape[2],shape[3],Kernel,sTride,Pad);

    assert(int(vC.size()) == shape[0]*shape[1]*shape[2]*shape[3]);

    std::cerr << "Starting " << std::endl;

    auto start = std::chrono::system_clock::now();
    for(int i=-skip;i<iters;i++) {
        if(i==0) {
            conv->sync();
            if(do_check) {
                conv->copy_back();
            }
            start = std::chrono::system_clock::now(); 
        }
        conv->calc();
        if(sync)
            conv->sync();
    }
    conv->sync();
    auto stop = std::chrono::system_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::duration<double> > ((stop-start)).count();
    std::cout << (flops / duration * 1e-9)  << " GFLOPS " << (duration * 1e+3) / iters << " ms"<< std::endl;
}
