#include <iostream>
#include <vector>
#include <stdio.h>
#include <assert.h>

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
        for(unsigned i=0;i<memory_access.size();i++) {
            addrs.assign(bank_size,-1);
            for(int j=0;j<total_threads;j++) {
                if(print) {
                    if(memory_access[i][j]==-1)
                        printf("%02d:none ",j);
                    else
                        printf("%02d:%04x ",j,memory_access[i][j]);
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
        }
        
    }
} the_bank;

struct array {
    int d0;
    int d1;
    int start;
    int end;
    void init(int x,int y,int s)
    {
        d0 = x;
        d1 = y;
        start = s;
        end = start+d0*d1;
    }
    void access(int x,int y) {
        the_bank.access(start + x * d1 + y,current_thread);
    }
};

array b_tile,a_tile;


void kernel()
{
    int block_row = get_local_id(0)*BLOCK_SIZE_M;
    int block_col = get_local_id(1)*BLOCK_SIZE_N;
        
    for(int dk=0;dk<TILE_SIZE_K;dk++) {
        for(int dc=0;dc<BLOCK_SIZE_N;dc++) {
            b_tile.access(dk,block_col+dc);
        }
        for(int dr=0;dr<BLOCK_SIZE_M;dr++) {
            a_tile.access(dk,block_row+dr);
        }
    }
}

void do_test(int ts_m=32,int ts_n=32,int tk=16,int bx=2,int by=2)
{
    TILE_SIZE_M = ts_m;
    TILE_SIZE_N = ts_n;
    TILE_SIZE_K = tk;
    BLOCK_SIZE_N = bx;
    BLOCK_SIZE_M = by;

    total_threads = TILE_SIZE_M * TILE_SIZE_N / BLOCK_SIZE_N / BLOCK_SIZE_M;

    a_tile.init(TILE_SIZE_K,TILE_SIZE_M,0);
    b_tile.init(TILE_SIZE_K,TILE_SIZE_N,a_tile.end);
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
    the_bank.calc_conflicts(conf,broad);
    if(!conf) {
        double flops_fill_rate = 100.0 * (BLOCK_SIZE_N * BLOCK_SIZE_M) / (BLOCK_SIZE_M + BLOCK_SIZE_N + BLOCK_SIZE_N * BLOCK_SIZE_M);
        printf("%5.3f%% Fill tm=%-3d tn=%-3d bm=%-2d bn=%-2d k=%-2d  Conf=%-8d BC=%-8d\n",flops_fill_rate,TILE_SIZE_M,TILE_SIZE_N,BLOCK_SIZE_N,BLOCK_SIZE_M,TILE_SIZE_K,conf,broad);
    }
    
}


int main()
{
    bank_size = 32;
    local_memory_limit = 49152;
    do_test();

    for(int wg_size = 32;wg_size<=512;wg_size+=32) {
        for(int wg_size_n=1;wg_size_n <= wg_size;wg_size_n++) {
            if(wg_size % wg_size_n != 0)
                continue;
            int wg_size_m = wg_size / wg_size_n;
            for(int bs_n = 1;bs_n <= wg_size;bs_n++) {
                for(int bs_m = 1;bs_m <= wg_size;bs_m++) {
                    int tile_size_n = wg_size_n * bs_n;
                    int tile_size_m = wg_size_m * bs_m;
                    if(tile_size_n > 128 || tile_size_m > 128)
                        continue;
                    for(int tile_size_k = 1;tile_size_k <= 32;tile_size_k ++) {
                        if(tile_size_m * tile_size_k % wg_size != 0 || tile_size_n * tile_size_k % wg_size != 0)
                            continue;
                        if(tile_size_k * (tile_size_n + tile_size_m) * sizeof(float) + 16 >= local_memory_limit)
                            continue;
                        do_test(tile_size_m,tile_size_n,tile_size_k,bs_n,bs_m);
                    }
                }
            }
        }
    }


}
