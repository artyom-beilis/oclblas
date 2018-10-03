// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif
#ifndef BLOCK_SIZE_X
#define BLOCK_SIZE_X 2
#endif
#ifndef BLOCK_SIZE_Y
#define BLOCK_SIZE_Y 4
#endif

#ifndef TILE_SIZE_K
#define TILE_SIZE_K TILE_SIZE
#endif 

#define BLOCK_SIZE_XY (BLOCK_SIZE_X*BLOCK_SIZE_Y)
#define BLOCKS_IN_TILE_X (TILE_SIZE / BLOCK_SIZE_X)
#define BLOCKS_IN_TILE_Y (TILE_SIZE / BLOCK_SIZE_Y)

#define ALIGN_FLOAT4 __attribute__ ((aligned (16)))



__kernel 
__attribute__((reqd_work_group_size(BLOCKS_IN_TILE_Y, BLOCKS_IN_TILE_X, 1)))
void    sgemm(    int M,int N,int K,
        __global const float * restrict A,int lda,
        __global const float * restrict B,int ldb,
        __global float * restrict C,int ldc)
{
    ALIGN_FLOAT4 __local float a_tile[TILE_SIZE_K][TILE_SIZE];
    ALIGN_FLOAT4
    __local float b_tile[TILE_SIZE_K][TILE_SIZE];

    float c[BLOCK_SIZE_Y][BLOCK_SIZE_X];
    float bp[BLOCK_SIZE_X];
    float av;
    
    int row = get_global_id(0) * BLOCK_SIZE_Y;
    int col = get_global_id(1) * BLOCK_SIZE_X;

    int lid0 = get_local_id(0);
    int lid1 = get_local_id(1);
    
    int block_row = get_local_id(0)*BLOCK_SIZE_Y;
    int block_col = get_local_id(1)*BLOCK_SIZE_X;
    
    #pragma unroll
    for(int i=0;i<BLOCK_SIZE_Y;i++) { 
        #pragma unroll
        for(int j=0;j<BLOCK_SIZE_X;j++) { 
            c[i][j]=0.0f;
        }
    }

    const int k_steps_y = TILE_SIZE_K / BLOCKS_IN_TILE_Y;
    int k_offset_row = lid0 * k_steps_y;
    const int k_steps_x = TILE_SIZE_K / BLOCKS_IN_TILE_X;
    int k_offset_col = lid1 * k_steps_x;


    int k=0;
    for(k=0;k<K;k+=TILE_SIZE_K) {
        #pragma unroll
        for(int dr=0;dr<k_steps_y;dr++) {
            #pragma unroll
            for(int dc=0;dc<BLOCK_SIZE_X;dc++) {
                b_tile[k_offset_row + dr][block_col + dc] = B[(k+k_offset_row+dr)*ldb+col+dc]; 
            }
        }
        #pragma unroll
        for(int dr=0;dr<BLOCK_SIZE_Y;dr++) {
            #pragma unroll
            for(int dc=0;dc<k_steps_x;dc++) {
                a_tile[k_offset_col + dc][block_row + dr] = A[(row+dr)*lda+k+k_offset_col+dc];
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

    #pragma unroll
    for(int dr=0;dr<BLOCK_SIZE_Y;dr++) {
        #pragma unroll
        for(int dc=0;dc<BLOCK_SIZE_X;dc++) {
            C[(row+dr)*ldc+col+dc] = c[dr][dc];
        }
    }


}


