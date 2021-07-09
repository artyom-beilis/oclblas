// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#ifndef TILE_SIZE_M
#define TILE_SIZE_M 32
#endif
#ifndef TILE_SIZE_N
#define TILE_SIZE_N 32
#endif
#ifndef BLOCK_SIZE_N
#define BLOCK_SIZE_N 4
#endif
#ifndef BLOCK_SIZE_M
#define BLOCK_SIZE_M 4
#endif

#ifndef TILE_SIZE_K
#define TILE_SIZE_K 8
#endif 

#ifndef ZORDER
#define ZORDER 0
#endif

#define BLOCK_SIZE_NY (BLOCK_SIZE_N*BLOCK_SIZE_M)
#define BLOCKS_IN_TILE_N (TILE_SIZE_N / BLOCK_SIZE_N)
#define BLOCKS_IN_TILE_M (TILE_SIZE_M / BLOCK_SIZE_M)
#define WG_SIZE (BLOCKS_IN_TILE_M * BLOCKS_IN_TILE_N)


#define ALIGN_FLOAT4 __attribute__ ((aligned (16)))

#ifndef BTRANS
#define get_B(r,c) (B[(r)*ldb + (c)])
#else
#define get_B(r,c) (B[(c)*ldb + (r)])
#endif

#ifndef ATRANS
#define get_A(r,c) (A[(r)*lda + (c)])
#else
#define get_A(r,c) (A[(c)*lda + (r)])
#endif


#ifdef OCL_TO_CU

#define __kernel extern "C" __global__
#define __global 
#define restrict
#define reqd_work_group_size(a,b,c)
#define __local __shared__
#define mad(x,y,z) fma(x,y,z)
#define get_global_id(dim_sel)  (dim_sel == 0 ? (blockIdx.x*blockDim.x + threadIdx.x) : (blockIdx.y*blockDim.y + threadIdx.y))
#define get_group_id(dim_sel)   (dim_sel == 0 ? (blockIdx.x) : (blockIdx.y))
#define get_local_size(dim_sel) (dim_sel == 0 ? (blockDim.x) : (blockDim.y))
#define get_local_id(dim_sel)   (dim_sel == 0 ? (threadIdx.x) : (threadIdx.y))
#define barrier(x) __syncthreads() 

#endif

//#define SIM

#ifndef TILE_OFFSET
#define TILE_OFFSET 0
#endif

#define lA(x,y) a_tile[(x)][(y) / BLOCK_SIZE_M][(y) % BLOCK_SIZE_M]
#define lB(x,y) b_tile[(x)][(y) / BLOCK_SIZE_N][(y) % BLOCK_SIZE_N]

int zorder_a(int x)
{
    return
          ((x & (1<<0)) >> 0 )
        | ((x & (1<<2)) >> 1 )
        | ((x & (1<<4)) >> 2 )
        | ((x & (1<<6)) >> 3 )
        | ((x & (1<<8)) >> 4 )
        | ((x & (1<<10)) >> 5 );
}
int zorder_b(int x)
{
    return zorder_a(x>>1);
}




#if TILE_SIZE_M != TILE_SIZE_N
#error "Unsupported condif"
#endif



__kernel 
__attribute__((reqd_work_group_size(BLOCKS_IN_TILE_M, BLOCKS_IN_TILE_N, 1)))
void    sgemm(    int M,int N,int K,
        __global const float * restrict A,int lda,
        __global const float * restrict B,int ldb,
        __global float * restrict C,int ldc)
{
    ALIGN_FLOAT4 __local float a_tile[TILE_SIZE_K][BLOCKS_IN_TILE_M][BLOCK_SIZE_M+TILE_OFFSET];
    ALIGN_FLOAT4 __local float b_tile[TILE_SIZE_K][BLOCKS_IN_TILE_N][BLOCK_SIZE_N+TILE_OFFSET];

    float c[BLOCK_SIZE_M][BLOCK_SIZE_N] = {{0.0f}};
    float ap[BLOCK_SIZE_M];
    float bp[BLOCK_SIZE_N];

#define DIM_M 0
#define DIM_N 1
#if ZORDER == 1
    int gr_m = get_group_id(DIM_M);
    int gr_n = get_group_id(DIM_N);
    int gr_size_m = get_num_groups(DIM_M);
    int gr_size_n = get_num_groups(DIM_N);
    if(gr_size_m == gr_size_n && (gr_size_m == 8 || gr_size_m == 16 || gr_size_m == 32 || gr_size_m == 64 || gr_size_m == 128)) {
        int grs  = gr_n * gr_size_m + gr_m;
        gr_n = zorder_a(grs);
        gr_m = zorder_b(grs);
    }
#else
    int gr_m = get_group_id(DIM_M);
    int gr_n = get_group_id(DIM_N);
#endif
    int tile_row0 = gr_m*TILE_SIZE_M;
    int tile_col0 = gr_n*TILE_SIZE_N;

    if(tile_row0 >= M)
        return;
    if(tile_col0 >= N)
        return;

    int row = tile_row0 + get_local_id(DIM_M) * BLOCK_SIZE_M;
    int col = tile_col0 + get_local_id(DIM_N) * BLOCK_SIZE_N;


    int lid0 = get_local_id(DIM_M);
    int lid1 = get_local_id(DIM_N);
    
    
    int local_tile_id = lid0 * get_local_size(1) + lid1;
    //const int local_wg_size = BLOCKS_IN_TILE_M * BLOCKS_IN_TILE_N;
    //const int load_step = TILE_SIZE_M * TILE_SIZE_K / local_wg_size;
    #define local_wg_size (BLOCKS_IN_TILE_M * BLOCKS_IN_TILE_N)
    #define load_step (TILE_SIZE_M * TILE_SIZE_K / local_wg_size)
    
    int dM[load_step];
    int dN[load_step];
    int dK [load_step];
    __local float *aP[load_step];
    __local float *bP[load_step];

    for(int i=0,read_pos = local_tile_id;i<load_step;i++,read_pos+=WG_SIZE) {
        //int tile_kdir = read_pos / TILE_SIZE_M;
        //int tile_tdir = read_pos % TILE_SIZE_M;
        int tile_kdir = read_pos % TILE_SIZE_K;
        int tile_tdir = read_pos / TILE_SIZE_K;
        dM[i] = tile_tdir + tile_row0;
        dN[i] = tile_tdir + tile_col0;
        dK[i]  = tile_kdir;
        aP[i] = &lA(tile_kdir,tile_tdir);
        bP[i] = &lB(tile_kdir,tile_tdir);
    }


    int k=0;
    for(k=0;k<K;k+=TILE_SIZE_K) {

        #ifdef SIM
        if(k==0) {
            #pragma unroll
            for(int i=0,read_pos = local_tile_id;i<load_step;i++,read_pos+=WG_SIZE) {
                int tile_kdir = read_pos / TILE_SIZE_M;
                int tile_tdir = read_pos % TILE_SIZE_M;
                lA(tile_kdir,tile_tdir) = 1.3f;
            }
            #pragma unroll
            for(int i=0,read_pos = local_tile_id;i<load_step;i++,read_pos+=WG_SIZE) {
                int tile_kdir = read_pos / TILE_SIZE_N;
                int tile_tdir = read_pos % TILE_SIZE_N;
                lB(tile_kdir,tile_tdir) = 2.3f;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        #elif TILE_SIZE_N == TILE_SIZE_M && TILE_SIZE_K % load_step  == 0 && load_step <= TILE_SIZE_K
        {
            int tile_kdir0 = local_tile_id / TILE_SIZE_M;
            int tile_tdir  = local_tile_id % TILE_SIZE_M;
            int a_row = tile_tdir + tile_row0;
            int b_col = tile_tdir + tile_col0;

            if(a_row >= M) {
                #pragma unroll
                for(int i=0,tile_kdir=tile_kdir0;i<load_step;i++,tile_kdir+=WG_SIZE / TILE_SIZE_M) {
                    lA(tile_kdir,tile_tdir) = 0.0f;
                }
            }
            else {
                if(tile_kdir0 + k <= K - load_step * (WG_SIZE / TILE_SIZE_M)) {
                    #pragma unroll
                    for(int i=0,tile_kdir=tile_kdir0;i<load_step;i++,tile_kdir+=WG_SIZE / TILE_SIZE_M) {
                        int k_rc  = tile_kdir + k;
                        lA(tile_kdir,tile_tdir) = get_A(a_row,k_rc);
                    }
                }
                else {
                    #pragma unroll
                    for(int i=0,tile_kdir=tile_kdir0;i<load_step;i++,tile_kdir+=WG_SIZE / TILE_SIZE_M) {
                        int k_rc  = tile_kdir + k;
                        lA(tile_kdir,tile_tdir) = k_rc < K ? get_A(a_row,k_rc) : 0.0f;
                    }
                }
            }
            if(b_col >= N) {
                #pragma unroll
                for(int i=0,tile_kdir=tile_kdir0;i<load_step;i++,tile_kdir+=WG_SIZE / TILE_SIZE_M) {
                    lB(tile_kdir,tile_tdir) = 0.0f;
                }
            }
            else {
                if(tile_kdir0 + k <= K - load_step * (WG_SIZE / TILE_SIZE_N)) {
                    #pragma unroll
                    for(int i=0,tile_kdir=tile_kdir0;i<load_step;i++,tile_kdir+=WG_SIZE / TILE_SIZE_N) {
                        int k_rc  = tile_kdir + k;
                        lB(tile_kdir,tile_tdir) = get_B(k_rc,b_col);
                    }
                }
                else {
                    #pragma unroll
                    for(int i=0,tile_kdir=tile_kdir0;i<load_step;i++,tile_kdir+=WG_SIZE / TILE_SIZE_N) {
                        int k_rc  = tile_kdir + k;
                        lB(tile_kdir,tile_tdir) = k_rc < K ? get_B(k_rc,b_col) : 0.0f;
                    }
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
        #else
        {
            if(tile_row0 + TILE_SIZE_M <= M && k + TILE_SIZE_K <= K) {
                #pragma unroll
                for(int i=0;i<load_step;i++) {
                    int a_row = dM[i];
                    int k_rc  = dK[i] + k;
                    *aP[i] =  get_A(a_row,k_rc);
                }
            }
            else {
                #pragma unroll
                for(int i=0;i<load_step;i++) {
                    int a_row = dM[i];
                    int k_rc  = dK[i] + k;
                    *aP[i] = (a_row < M && k_rc < K) ?  get_A(a_row,k_rc) : 0.0f;
                }
            }
            if(tile_col0 + TILE_SIZE_N <= N && k + TILE_SIZE_K <= K) {
                #pragma unroll
                for(int i=0;i<load_step;i++) {
                    int k_rc  = dK[i]  + k;
                    int b_col = dN[i];
                    *bP[i] = get_B(k_rc,b_col);
                }
            }
            else {
                #pragma unroll
                for(int i=0;i<load_step;i++) {
                    int k_rc  = dK[i]  + k;
                    int b_col = dN[i];
                    *bP[i] = (b_col < N && k_rc < K) ? get_B(k_rc,b_col) : 0.0f;

                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        #endif

        #if 0
        // Mutliplication loop
        __local float *ptr_A = &a_tile[0][lid0][0]; 
        __local float *ptr_B = &b_tile[0][lid1][0];

        #pragma unroll
        for(int dk=0;dk<TILE_SIZE_K;dk++) {
            #pragma unroll
            for(int dr=0;dr<BLOCK_SIZE_M;dr++) {
                ap[dr] = ptr_A[dr];
            }
            #pragma unroll
            for(int dc=0;dc<BLOCK_SIZE_N;dc++) {
                bp[dc] = ptr_B[dc];
            }
            #pragma unroll
            for(int dr=0;dr<BLOCK_SIZE_M;dr++) {
                #pragma unroll
                for(int dc=0;dc<BLOCK_SIZE_N;dc++) {
                    c[dr][dc] = mad(ap[dr],bp[dc],c[dr][dc]);
                }
            }
            ptr_A += BLOCKS_IN_TILE_M * (BLOCK_SIZE_M+TILE_OFFSET);
            ptr_B += BLOCKS_IN_TILE_N * (BLOCK_SIZE_N+TILE_OFFSET);
        }
        #else

        #pragma unroll
        for(int dk=0;dk<TILE_SIZE_K;dk++) {
            #pragma unroll
            for(int dr=0;dr<BLOCK_SIZE_M;dr++) {
                ap[dr] = a_tile[dk][lid0][dr];
            }
            #pragma unroll
            for(int dc=0;dc<BLOCK_SIZE_N;dc++) {
                bp[dc] = b_tile[dk][lid1][dc];
            }
            #pragma unroll
            for(int dr=0;dr<BLOCK_SIZE_M;dr++) {
                #pragma unroll
                for(int dc=0;dc<BLOCK_SIZE_N;dc++) {
                    c[dr][dc] = mad(ap[dr],bp[dc],c[dr][dc]);
                }
            }
        }
        #endif

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    
    {
        #pragma unroll
        for(int dr=0;dr<BLOCK_SIZE_M;dr++) {
            #pragma unroll
            for(int dc=0;dc<BLOCK_SIZE_N;dc++) {
                if(row + dr < M && col+dc < N)
                    C[(row+dr)*ldc+col+dc] = c[dr][dc];
            }
        }
    }

}


