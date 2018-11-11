// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#include <iostream>
#include <vector>
#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <stdlib.h>

namespace {
    int BLOCK_SIZE_M;
    int BLOCK_SIZE_N;
    int TILE_SIZE_K;
    int TILE_SIZE_M;
    int TILE_SIZE_N;

    int local_id[2];
    #define get_local_id(n) (local_id[n])
    int current_thread;
    int total_threads;
    int bank_size;
    unsigned local_memory_limit; 
    bool a_tr,b_tr;

}

struct bank {
    int bank_size;
    std::vector<std::vector<int> > memory_access;
    std::vector<int> pc;
    void init()
    {
        this->bank_size = ::bank_size;
        pc.assign(total_threads,0);
        memory_access.clear();
    }
    void access(int addr,int thread)
    {
        int cpc = pc[thread]++;
        if(cpc >= int(memory_access.size()))
            memory_access.push_back(std::vector<int>(total_threads,-1));
        memory_access[cpc][thread]=addr;
    }
    void calc_conflicts(int &conflicts,int &broadcasts,bool print=false)
    {
        conflicts = 0;
        broadcasts = 0;
        std::vector<int> addrs;
        int gdim = TILE_SIZE_M/BLOCK_SIZE_M;
        for(unsigned i=0;i<memory_access.size();i++) {
            addrs.assign(bank_size,-1);
            for(int j=0;j<total_threads;j++) {
                if(print) {
                    if(j % gdim == 0)
                        printf("  %2d: ", j / gdim);
                    if(memory_access[i][j]==-1)
                        printf("%02d:none ",j);
                    else
                        printf("%03d:%04x:%02x; ",j,memory_access[i][j],memory_access[i][j] % bank_size);
                    if(j % gdim == gdim - 1)
                        printf("\n"); 
                }
                if(memory_access[i][j]==-1)
                    continue;
                int addr = memory_access[i][j];
                int bank_id = addr % bank_size;
                if(addrs[bank_id] == -1)
                    addrs[bank_id]=addr;
                else if(addrs[bank_id] != addr)
                    conflicts ++;
                else
                    broadcasts ++;
                addrs[bank_id] = addr;
            }
            if(print)
                printf("\n");
            if(i==2)
               print = false;
        }
        
    }
} the_bank;

struct array {
    int d0;
    int d1;
    int d2;
    int start;
    int end;
    void init(int x,int y,int z,int s)
    {
        d0 = x;
        d1 = y;
        d2 = z;
        start = (s+3)/4*4;
        end = start+d0*d1*d2;
    }
    void access(int x,int y,int z) {
        the_bank.access(start + x * d1*d2 + y*d2,current_thread);
    }
};

array b_tile,a_tile;


void kernel()
{
//    int block_row = get_local_id(0)*BLOCK_SIZE_M;
//    int block_col = get_local_id(1)*BLOCK_SIZE_N;
        
    for(int dk=0;dk<TILE_SIZE_K;dk++) {
        for(int dc=0;dc<BLOCK_SIZE_N;dc++) {
            b_tile.access(dk,get_local_id(1),dc);
        }
        for(int dr=0;dr<BLOCK_SIZE_M;dr++) {
            a_tile.access(dk,get_local_id(0),dr);
        }
    }
}

void do_test(int ts_m=32,int ts_n=32,int tk=16,int bx=2,int by=2,int off = 0)
{
    TILE_SIZE_M = ts_m;
    TILE_SIZE_N = ts_n;
    TILE_SIZE_K = tk;
    BLOCK_SIZE_N = bx;
    BLOCK_SIZE_M = by;

    total_threads = TILE_SIZE_M * TILE_SIZE_N / BLOCK_SIZE_N / BLOCK_SIZE_M;
    int small_tiles_in_m = TILE_SIZE_M / BLOCK_SIZE_M;
    int small_tiles_in_n = TILE_SIZE_N / BLOCK_SIZE_N;

    a_tile.init(TILE_SIZE_K,small_tiles_in_m,BLOCK_SIZE_M + off,0);
    b_tile.init(TILE_SIZE_K,small_tiles_in_n,BLOCK_SIZE_N + off,a_tile.end);

    the_bank.init();

    current_thread=0;
    for(int x=0;x<TILE_SIZE_N/BLOCK_SIZE_N;x++) {
        for(int y=0;y<TILE_SIZE_M/BLOCK_SIZE_M;y++) {
            local_id[0]=x;
            local_id[1]=y;
            kernel();
            current_thread++;
        }
    }
    int conf,broad;
    the_bank.calc_conflicts(conf,broad,false);
    double flops_fill_rate = 100.0 * (BLOCK_SIZE_N * BLOCK_SIZE_M) / (BLOCK_SIZE_M + BLOCK_SIZE_N + BLOCK_SIZE_N * BLOCK_SIZE_M);
    if(!conf) {
        printf("%5.3f%% Fill TILE_SIZE_M=%-3d TILE_SIZE_N=%-3d BLOCK_M=%-2d BLOCK_N=%-2d TILE_SIZE_K=%-2d off=%d\n",flops_fill_rate,TILE_SIZE_M,TILE_SIZE_N,BLOCK_SIZE_M,BLOCK_SIZE_N,TILE_SIZE_K,off);
        //printf("     conf=%d broad=%d\n",conf,broad);
    }
    
}


int main(int argc,char **argv)
{
    bank_size = 32;
    local_memory_limit = 49152;
    int wg_start = 32;
    int wg_end = 1024;

    bool force_equal = false;
    int opt;
    while((opt=getopt(argc,argv,"eb:l:ABW:"))!= -1) {
        switch(opt) {
        case 'A':
            a_tr = true;
            break;
        case 'B':
            b_tr = true;
            break;
        case 'e':
            force_equal = true; 
            break;
        case 'W':
            wg_start=wg_end = atoi(optarg);
            break;
        case 'b':
            bank_size = atoi(optarg);
            break;
        case 'l':
            local_memory_limit = atoi(optarg);
            break;
        default:
            std::cerr << "Usage [-e ] [ -b banks no ] [ -l loc_mem_size ]" << std::endl;
            return 1;
        }
    }

    if(argc==2 && std::string(argv[1]) == "-e")
	    force_equal = true;

    for(int wg_size = wg_start;wg_size<=wg_end;wg_size+=32) {
        for(int wg_size_n=1;wg_size_n <= wg_size;wg_size_n++) {
            if(wg_size % wg_size_n != 0)
                continue;
            int wg_size_m = wg_size / wg_size_n;
            for(int bs_n = 1;bs_n <= wg_size;bs_n++) {
                for(int bs_m = 1;bs_m <= wg_size;bs_m++) {
                    int tile_size_n = wg_size_n * bs_n;
                    int tile_size_m = wg_size_m * bs_m;
		    if(force_equal && tile_size_m != tile_size_n)
			    continue;
                    if(tile_size_n > 256 || tile_size_m > 256)
                        continue;
                    for(int tile_size_k = 1;tile_size_k <= 32;tile_size_k ++) {
                        if(tile_size_m * tile_size_k % wg_size != 0 || tile_size_n * tile_size_k % wg_size != 0)
                            continue;
                        for(int off = 0;off < 3;off++) {
                            if((tile_size_k + off) * (2*off + tile_size_n + tile_size_m) * sizeof(float) + 16 >= local_memory_limit)
                                continue;
                          //if(!(tile_size_n == 128 && tile_size_m==128 && bs_m == 8 && bs_n == 8))
                          //    continue;
                            do_test(tile_size_m,tile_size_n,tile_size_k,bs_n,bs_m,off);
                        }
                    }
                }
            }
        }
    }


}
