// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#ifndef TILE_SIZE_M
#define TILE_SIZE_M TILE_SIZE
#endif
#ifndef TILE_SIZE_N
#define TILE_SIZE_N TILE_SIZE
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
//#define INV_TILE_ORDER

#define lA(x,y) a_tile[(x)][(y) / BLOCK_SIZE_M][(y) % BLOCK_SIZE_M]
#define lB(x,y) b_tile[(x)][(y) / BLOCK_SIZE_N][(y) % BLOCK_SIZE_N]
//#define lA(x,y) a_tile[(x)][0][(y)]
//#define lB(x,y) b_tile[(x)][0][(y)]

__kernel 
__attribute__((reqd_work_group_size(BLOCKS_IN_TILE_M, BLOCKS_IN_TILE_N, 1)))
void    sgemm(    int M,int N,int K,
        __global const float * restrict A,int lda,
        __global const float * restrict B,int ldb,
        __global float * restrict C,int ldc)
{
    ALIGN_FLOAT4 __local float a_tile[TILE_SIZE_K][BLOCKS_IN_TILE_M][BLOCK_SIZE_M+1];
    ALIGN_FLOAT4 __local float b_tile[TILE_SIZE_K][BLOCKS_IN_TILE_N][BLOCK_SIZE_N+1];

    float c[BLOCK_SIZE_M][BLOCK_SIZE_N] = {{0.0f}};
    float ap[BLOCK_SIZE_M];
    float bp[BLOCK_SIZE_N];
    
    int row = get_global_id(0) * BLOCK_SIZE_M;
    int col = get_global_id(1) * BLOCK_SIZE_N;

    int lid0 = get_local_id(0);
    int lid1 = get_local_id(1);
    
    int local_tile_id = lid0 * get_local_size(1) + lid1;
    #if 0

    int warp_id = local_tile_id / 32;
    int thread_in_wrap_id = local_tile_id % 32;

    #if TILE_SIZE_N != TILE_SIZE_M || TILE_SIZE_M != 128 || BLOCK_SIZE_N != 8 || BLOCK_SIZE_M != 8
    #error
    #endif
    int warp_row = warp_id / 4;
    int warp_col = warp_id % 4;
    int tinw_row = thread_in_wrap_id / 4;
    int tinw_col = thread_in_wrap_id % 4;
    int block_row = BLOCK_SIZE_M * (warp_row * 8 + tinw_row);
    int block_col = BLOCK_SIZE_N * (warp_col * 4 + tinw_col);
    
    row = get_group_id(0)*TILE_SIZE_M + block_row;
    col = get_group_id(1)*TILE_SIZE_N + block_col;
    
    #else

    int block_row = get_local_id(0)*BLOCK_SIZE_M;
    int block_col = get_local_id(1)*BLOCK_SIZE_N;

    #endif

    int tile_row0 = get_group_id(0)*TILE_SIZE_M;
    int tile_col0 = get_group_id(1)*TILE_SIZE_N;

    bool good_row = (tile_row0 + TILE_SIZE_M <= M);
    bool good_col = (tile_col0 + TILE_SIZE_N <= N);

    const int local_wg_size = BLOCKS_IN_TILE_M * BLOCKS_IN_TILE_N;
    #if TILE_SIZE_M == TILE_SIZE_N
    const int load_step = TILE_SIZE_M * TILE_SIZE_K / local_wg_size;
    #else
    const int load_step_a = TILE_SIZE_M * TILE_SIZE_K / local_wg_size;
    const int load_step_b = TILE_SIZE_N * TILE_SIZE_K / local_wg_size;
    #endif
    

    int k=0;
    for(k=0;k<K;k+=TILE_SIZE_K) {

        #ifdef SIM
        if(k == 0) {
            //#pragma unroll
            for(int i=0,read_pos = local_tile_id;i<load_step;i++) {
                int tile_kdir = read_pos / TILE_SIZE_M;
                int tile_tdir = read_pos % TILE_SIZE_M;
                lA(tile_kdir,tile_tdir) = 1.3f * tile_kdir + tile_tdir;
            }
            for(int i=0,read_pos = local_tile_id;i<load_step;i++) {
                int tile_kdir = read_pos / TILE_SIZE_N;
                int tile_tdir = read_pos % TILE_SIZE_N;
                lB(tile_kdir,tile_tdir) = 2.3f * tile_kdir + tile_tdir;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        #else
        #if TILE_SIZE_M == TILE_SIZE_N
            #if  1 // TILE_SIZE_K > WG_SIZE || WG_SIZE % TILE_SIZE_K != 0
            //#if 1
            //#if TILE_SIZE_K > WG_SIZE || WG_SIZE % TILE_SIZE_K != 0
            //#endif
            if(1) // good_row && good_col && k_rc < K) 
            {
                const int mn_step = TILE_SIZE_M / ( WG_SIZE / TILE_SIZE_K );
                int mn_pos = mn_step * (local_tile_id / TILE_SIZE_K);
                int k_pos = local_tile_id % TILE_SIZE_K;
                int a_row = tile_row0 + mn_pos;
                int b_col = tile_col0 + mn_pos;
                int k_rc  = k_pos + k;
                int tile_mm = mn_pos;
                #pragma unroll
                for(int mn=0;mn<mn_step;mn++)
                {
                    lA(k_pos,tile_mm) = get_A(a_row,k_rc) ;
                    lB(k_pos,tile_mm) = get_B(k_rc, b_col);
                    tile_mm ++;
                    a_row   ++;
                    b_col   ++;
                }
            }
            /*else {
                if(k_rc >= K) {
                    for(int mn=0;mn<mn_step;mn++)
                    {
                        lA(k_pos,tile_mm) = 0.0f;
                        lB(k_pos,tile_mm) = 0.0f;
                        tile_mm ++;
                    }
                }
                else {
                    for(int mn=0;mn<mn_step;mn++)
                    {
                        lA(k_pos,tile_mm) = (a_row < M) ? get_A(a_row,k_rc)  : 0.0f;
                        lB(k_pos,tile_mm) = (b_col < M) ? get_B(k_rc, b_col) : 0.0f;
                        tile_mm ++;
                        a_row   ++;
                        b_col   ++;
                    }
                }
            }*/
            #else
            

            int read_pos = local_tile_id;
            if(good_row && good_col && k + TILE_SIZE_K <= K) {
                //#pragma unroll
                for(int i=0;i<load_step;i++) {
                    int tile_kdir = read_pos / TILE_SIZE_M;
                    int tile_tdir = read_pos % TILE_SIZE_M;
                    int k_rc  = tile_kdir + k;
                    int a_row = tile_tdir + tile_row0;
                    int b_col = tile_tdir + tile_col0;
                    lA(tile_kdir,tile_tdir) = get_A(a_row,k_rc );
                    lB(tile_kdir,tile_tdir) = get_B(k_rc ,b_col);
                    read_pos += local_wg_size;
                }
            }
            else {
                //#pragma unroll
                for(int i=0;i<load_step;i++) {
                    int tile_kdir = read_pos / TILE_SIZE_M;
                    int tile_tdir = read_pos % TILE_SIZE_M;

                    int k_rc  = tile_kdir + k;
                    if(k_rc < K) {
                        int a_row = tile_tdir + tile_row0;
                        int b_col = tile_tdir + tile_col0;
                        lA(tile_kdir,tile_tdir) = (a_row < M) ? get_A(a_row,k_rc ) : 0.0f;
                        lB(tile_kdir,tile_tdir) = (b_col < N) ? get_B(k_rc ,b_col) : 0.0f;
                    } 
                    else {
                        lA(tile_kdir,tile_tdir) = 0.0f;
                        lB(tile_kdir,tile_tdir) = 0.0f;
                    }

                    read_pos += local_wg_size;
                }
            }
            #endif

        #else // TILE_SIZE_M != TILE_SIZE_N
            int read_pos = local_tile_id;
            if(good_row && k + TILE_SIZE_K <= K) {
                //#pragma unroll
                for(int i=0;i<load_step_a;i++) {
                    int tile_kdir = read_pos / TILE_SIZE_M;
                    int tile_tdir = read_pos % TILE_SIZE_M;
                    int k_rc  = tile_kdir + k;
                    int a_row = tile_tdir + tile_row0;
                    lA(tile_kdir,tile_tdir) = get_A(a_row,k_rc );
                    read_pos += local_wg_size;
                }
            }
            else {
                //#pragma unroll
                for(int i=0;i<load_step_a;i++) {
                    int tile_kdir = read_pos / TILE_SIZE_M;
                    int tile_tdir = read_pos % TILE_SIZE_M;

                    int k_rc  = tile_kdir + k;
                    if(k_rc < K) {
                        int a_row = tile_tdir + tile_row0;
                        lA(tile_kdir,tile_tdir) = (a_row < M) ? get_A(a_row,k_rc ) : 0.0f;
                    } 
                    else {
                        lA(tile_kdir,tile_tdir) = 0.0f;
                    }

                    read_pos += local_wg_size;
                }
            }

            read_pos = local_tile_id;


            if(good_col && k + TILE_SIZE_K <= K) {
                //#pragma unroll
                for(int i=0;i<load_step_b;i++) {
                    int tile_kdir = read_pos / TILE_SIZE_N;
                    int tile_tdir = read_pos % TILE_SIZE_N;

                    int k_rc  = tile_kdir + k;
                    int b_col = tile_tdir + tile_col0;

                    lB(tile_kdir,tile_tdir) = get_B(k_rc ,b_col);
                    read_pos += local_wg_size;
                }
            }
            else {
                //#pragma unroll
                for(int i=0;i<load_step_b;i++) {
                    int tile_kdir = read_pos / TILE_SIZE_N;
                    int tile_tdir = read_pos % TILE_SIZE_N;

                    int k_rc  = tile_kdir + k;
                    if(k_rc < K) {
                        int b_col = tile_tdir + tile_col0;
                        lB(tile_kdir,tile_tdir) = (b_col < N) ? get_B(k_rc ,b_col) : 0.0f;
                    } 
                    else {
                        lB(tile_kdir,tile_tdir) = 0.0f;
                    }

                    read_pos += local_wg_size;
                }
            }
        #endif
        barrier(CLK_LOCAL_MEM_FENCE);
        #endif

        int lmt = min(K-k,TILE_SIZE_K);
        for(int dk=0;dk<lmt;dk++) {
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

    if(row + BLOCK_SIZE_M <= M && col + BLOCK_SIZE_N <= N) {
        //#pragma unroll
        for(int dr=0;dr<BLOCK_SIZE_M;dr++) {
            #//pragma unroll
            for(int dc=0;dc<BLOCK_SIZE_N;dc++) {
                C[(row+dr)*ldc+col+dc] = c[dr][dc];
            }
        }
    }
    else {
        //#pragma unroll
        for(int dr=0;dr<BLOCK_SIZE_M;dr++) {
            //#pragma unroll
            for(int dc=0;dc<BLOCK_SIZE_N;dc++) {
                if(row + dr < M && col+dc < N)
                    C[(row+dr)*ldc+col+dc] = c[dr][dc];
            }
        }
    }

}


