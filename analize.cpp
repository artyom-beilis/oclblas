#include <iostream>
#include <vector>
#include <stdio.h>
#include <assert.h>

namespace {
    int BLOCK_SIZE_Y;
    int BLOCK_SIZE_X;
    int TILE_SIZE_K;
    int TILE_SIZE;

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
    int block_row = get_local_id(0)*BLOCK_SIZE_Y;
    int block_col = get_local_id(1)*BLOCK_SIZE_X;
        
    for(int dk=0;dk<TILE_SIZE_K;dk++) {
        for(int dc=0;dc<BLOCK_SIZE_X;dc++) {
            b_tile.access(dk,block_col+dc);
        }
        for(int dr=0;dr<BLOCK_SIZE_Y;dr++) {
            a_tile.access(dk,block_row+dr);
        }
    }
}

void do_test(int ts=32,int tk=16,int bx=2,int by=2)
{
    TILE_SIZE = ts;
    TILE_SIZE_K = tk;
    BLOCK_SIZE_X = bx;
    BLOCK_SIZE_Y = by;

    total_threads = TILE_SIZE * TILE_SIZE / BLOCK_SIZE_X / BLOCK_SIZE_Y;

    a_tile.init(TILE_SIZE_K,TILE_SIZE,0);
    b_tile.init(TILE_SIZE_K,TILE_SIZE,a_tile.end);
    the_bank.init();

    current_thread=0;
    for(int x=0;x<TILE_SIZE/BLOCK_SIZE_X;x++) {
        for(int y=0;y<TILE_SIZE/BLOCK_SIZE_Y;y++) {
            local_id[0]=x;
            local_id[1]=y;
            kernel();
            current_thread++;
        }
    }
    int conf,broad;
    the_bank.calc_conflicts(conf,broad);
    if(!conf) {
        double flops_fill_rate = 100.0 * (BLOCK_SIZE_X * BLOCK_SIZE_Y) / (BLOCK_SIZE_Y + BLOCK_SIZE_X + BLOCK_SIZE_X * BLOCK_SIZE_Y);
        printf("%5.3f%% Fill tile=%-3d by=%-2d bx=%-2d k=%-2d  Conf=%-8d BC=%-8d\n",flops_fill_rate,TILE_SIZE,BLOCK_SIZE_X,BLOCK_SIZE_Y,TILE_SIZE_K,conf,broad);
    }
    
}


int main()
{
    bank_size = 32;
    local_memory_limit = 49152;
    do_test();

    for(int wg_size = 32;wg_size<=512;wg_size+=32) {
        for(int wg_size_x=1;wg_size_x <= wg_size;wg_size_x++) {
            if(wg_size % wg_size_x != 0)
                continue;
            int wg_size_y = wg_size / wg_size_x;
            for(int bs_x = 1;bs_x <= wg_size;bs_x++) {
                for(int bs_y = 1;bs_y <= wg_size;bs_y++) {
                    if(wg_size_x * bs_x != wg_size_y * bs_y)
                        continue;
                    int tile_size = wg_size_x * bs_x;
                    if(tile_size > 128)
                        continue;
                    for(int tile_size_k = 1;tile_size_k <= wg_size;tile_size_k ++) {
                        if(tile_size * tile_size_k % wg_size != 0)
                            continue;
                        if(tile_size_k * 2 * tile_size * sizeof(float) >= local_memory_limit)
                            continue;
                        do_test(tile_size,tile_size_k,bs_x,bs_y);
                    }
                }
            }
        }
    }


}
