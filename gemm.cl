// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif
#ifndef BLOCK_SIZE_X
#define BLOCK_SIZE_X 4
#endif
#ifndef BLOCK_SIZE_Y
#define BLOCK_SIZE_Y 4
#endif

#ifndef TILE_SIZE_K
#define TILE_SIZE_K 8
#endif 

#define BLOCK_SIZE_XY (BLOCK_SIZE_X*BLOCK_SIZE_Y)
#define BLOCKS_IN_TILE_X (TILE_SIZE / BLOCK_SIZE_X)
#define BLOCKS_IN_TILE_Y (TILE_SIZE / BLOCK_SIZE_Y)

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



__kernel 
__attribute__((reqd_work_group_size(BLOCKS_IN_TILE_Y, BLOCKS_IN_TILE_X, 1)))
void    sgemm(    int M,int N,int K,
        __global const float * restrict A,int lda,
        __global const float * restrict B,int ldb,
        __global float * restrict C,int ldc)
{
    ALIGN_FLOAT4 __local float a_tile[TILE_SIZE_K][TILE_SIZE];
    ALIGN_FLOAT4 __local float b_tile[TILE_SIZE_K][TILE_SIZE];

    float c[BLOCK_SIZE_Y][BLOCK_SIZE_X] = {{0.0f}};
    float bp[BLOCK_SIZE_X];
    float av;
    
    int row = get_global_id(0) * BLOCK_SIZE_Y;
    int col = get_global_id(1) * BLOCK_SIZE_X;

    int lid0 = get_local_id(0);
    int lid1 = get_local_id(1);
    
    int block_row = get_local_id(0)*BLOCK_SIZE_Y;
    int block_col = get_local_id(1)*BLOCK_SIZE_X;

    int tile_row0 = get_group_id(0)*TILE_SIZE;
    int tile_col0 = get_group_id(1)*TILE_SIZE;

    bool good_row = (tile_row0 + TILE_SIZE <= M) && (tile_col0 + TILE_SIZE <= N);

    const int local_wg_size = BLOCKS_IN_TILE_Y * BLOCKS_IN_TILE_X;
    const int load_step = TILE_SIZE * TILE_SIZE_K / local_wg_size;
    
    int local_tile_id = lid0 * get_local_size(1) + lid1;

    int k=0;
    for(k=0;k<K;k+=TILE_SIZE_K) {
        int read_pos = local_tile_id;
        if(good_row && k + TILE_SIZE_K <= K) {
            #pragma unroll
            for(int i=0;i<load_step;i++) {
                int tile_kdir = read_pos / TILE_SIZE;
                int tile_tdir = read_pos % TILE_SIZE;

                int k_rc  = tile_kdir + k;
                int a_row = tile_tdir + tile_row0;
                int b_col = tile_tdir + tile_col0;

                a_tile[tile_kdir][tile_tdir] = get_A(a_row,k_rc );
                b_tile[tile_kdir][tile_tdir] = get_B(k_rc ,b_col);

                read_pos += local_wg_size;
            }
        }
        else {
            #pragma unroll
            for(int i=0;i<load_step;i++) {
                int tile_kdir = read_pos / TILE_SIZE;
                int tile_tdir = read_pos % TILE_SIZE;

                int k_rc  = tile_kdir + k;
                if(k_rc < K) {
                    int a_row = tile_tdir + tile_row0;
                    int b_col = tile_tdir + tile_col0;
                    a_tile[tile_kdir][tile_tdir] = (a_row < M) ? get_A(a_row,k_rc ) : 0.0f;
                    b_tile[tile_kdir][tile_tdir] = (b_col < N) ? get_B(k_rc ,b_col) : 0.0f;
                } 
                else {
                    a_tile[tile_kdir][tile_tdir] = 0.0f;
                    b_tile[tile_kdir][tile_tdir] = 0.0f;
                }

                read_pos += local_wg_size;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        int lmt = min(K-k,TILE_SIZE_K);
        for(int dk=0;dk<lmt;dk++) {
            
            #pragma unroll
            for(int dc=0;dc<BLOCK_SIZE_X;dc++) {
                bp[dc] = b_tile[dk][block_col+dc];
            }
            #pragma unroll
            for(int dr=0;dr<BLOCK_SIZE_Y;dr++) {
                av =  a_tile[dk][block_row+dr];
                #pragma unroll
                for(int dc=0;dc<BLOCK_SIZE_X;dc++) {
                    c[dr][dc] = mad(av,bp[dc],c[dr][dc]);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(row + BLOCK_SIZE_Y <= M && col + BLOCK_SIZE_X <= N) {
        #pragma unroll
        for(int dr=0;dr<BLOCK_SIZE_Y;dr++) {
            #pragma unroll
            for(int dc=0;dc<BLOCK_SIZE_X;dc++) {
                C[(row+dr)*ldc+col+dc] = c[dr][dc];
            }
        }
    }
    else {
        #pragma unroll
        for(int dr=0;dr<BLOCK_SIZE_Y;dr++) {
            #pragma unroll
            for(int dc=0;dc<BLOCK_SIZE_X;dc++) {
                if(row + dr < M && col+dc < N)
                    C[(row+dr)*ldc+col+dc] = c[dr][dc];
            }
        }
    }

}


