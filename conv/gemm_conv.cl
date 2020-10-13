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

#define BLOCK_SIZE_NY (BLOCK_SIZE_N*BLOCK_SIZE_M)
#define BLOCKS_IN_TILE_N (TILE_SIZE_N / BLOCK_SIZE_N)
#define BLOCKS_IN_TILE_M (TILE_SIZE_M / BLOCK_SIZE_M)
#define WG_SIZE (BLOCKS_IN_TILE_M * BLOCKS_IN_TILE_N)


#define ALIGN_FLOAT4 __attribute__ ((aligned (16)))

#define get_B(r,c) get_img_at_r_c(B,c,r,out_w,out_h,int_w,in_h)

inline float get_img_at_r_c(__global const float * restrict B,int ipos,int ppos,int out_w,int out_h,int in_w,int in_h)
{
    int depth = ppos / (CONV_KERNEL_W * CONV_KERNEL_H);
    int dy    = ppos % (CONV_KERNEL_W * CONV_KERNEL_H) / CONV_KERNEL_W - CONV_PAD_H;
    int dx    = ppos % CONV_KERNEL_W - CONV_PAD_W;
    int img_row = (ipos / out_w) * CONV_STRIDE_H + dy;
    int img_col = (ipos % out_w) * CONV_STRIDE_W + dx;
    if(img_row < 0 || img_row >= out_w)
        return 0.0;
    if(img_col < 0 || img_col >= out_h)
        return 0.0;

    return B[in_w * in_h * depth + img_row * in_w + img_col];
}


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


#if TILE_SIZE_M != TILE_SIZE_N
#error "Unsupported condif"
#endif

__kernel 
__attribute__((reqd_work_group_size(BLOCKS_IN_TILE_M, BLOCKS_IN_TILE_N, 1)))
void    conv_sgemm(    int M,int N,int K,
        __global const float * restrict A,int lda,
        __global const float * restrict B,int in_w,int  in_h,int out_w,int out_h,
        __global float * restrict C,int ldc)
{
    ALIGN_FLOAT4 __local float a_tile[TILE_SIZE_K][BLOCKS_IN_TILE_M][BLOCK_SIZE_M+TILE_OFFSET];
    ALIGN_FLOAT4 __local float b_tile[TILE_SIZE_K][BLOCKS_IN_TILE_N][BLOCK_SIZE_N+TILE_OFFSET];

    float c[BLOCK_SIZE_M][BLOCK_SIZE_N] = {{0.0f}};
    float ap[BLOCK_SIZE_M];
    float bp[BLOCK_SIZE_N];
    
    int row = get_global_id(0) * BLOCK_SIZE_M;
    int col = get_global_id(1) * BLOCK_SIZE_N;

    int lid0 = get_local_id(0);
    int lid1 = get_local_id(1);
    
    int local_tile_id = lid0 * get_local_size(1) + lid1;

    int tile_row0 = get_group_id(0)*TILE_SIZE_M;
    int tile_col0 = get_group_id(1)*TILE_SIZE_N;

    const int local_wg_size = BLOCKS_IN_TILE_M * BLOCKS_IN_TILE_N;
    const int load_step = TILE_SIZE_M * TILE_SIZE_K / local_wg_size;

    int k=0;
    for(k=0;k<K;k+=TILE_SIZE_K) {

        #if (TILE_SIZE_M == 32  && TILE_SIZE_N == 32  && BLOCK_SIZE_M==4 && BLOCK_SIZE_N == 4 && (TILE_SIZE_K==16 || TILE_SIZE_K==32 || TILE_SIZE_K==64)) \
          ||  (TILE_SIZE_M == 64  && TILE_SIZE_N == 64  && BLOCK_SIZE_M==8 && BLOCK_SIZE_N == 8 && (TILE_SIZE_K==16 || TILE_SIZE_K==32 || TILE_SIZE_K==64)) \
          ||  (TILE_SIZE_M == 128 && TILE_SIZE_N == 128 && BLOCK_SIZE_M==8 && BLOCK_SIZE_N == 8 && (TILE_SIZE_K==16 || TILE_SIZE_K==32 || TILE_SIZE_K==64))
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
            #pragma unroll
            for(int i=0,read_pos = local_tile_id;i<load_step;i++,read_pos+=WG_SIZE) {
                int tile_kdir = read_pos / TILE_SIZE_M;
                int tile_tdir = read_pos % TILE_SIZE_M;
                int a_row = tile_tdir + tile_row0;
                int k_rc  = tile_kdir + k;
                lA(tile_kdir,tile_tdir) = (a_row < M && k_rc < K) ?  get_A(a_row,k_rc) : 0.0f;
            }
            #pragma unroll
            for(int i=0,read_pos = local_tile_id;i<load_step;i++,read_pos+=WG_SIZE) {
                int tile_kdir = read_pos / TILE_SIZE_N;
                int tile_tdir = read_pos % TILE_SIZE_N;
                int k_rc  = tile_kdir + k;
                int b_col = tile_tdir + tile_col0;
                lB(tile_kdir,tile_tdir) = (b_col < N && k_rc < K) ? get_B(k_rc,b_col) : 0.0f;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        #endif

        // Mutliplication loop
        #pragma unroll(4)
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


