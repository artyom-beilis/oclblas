#define SIM 0
#define INV_DK 0
#define REV_KERNELS 0
#define USE_LOCAL_MEM 1

#define LOCAL_TMEM_PAD 0
#define LOCAL_TMEM_PAD0 0
#define LOCAL_TMEM_PADX 0

#define LOCAL_KMEM_PAD 1
#define LOCAL_KMEM_PAD0 0

#ifdef __NV_CL_C_VERSION
#   define NVIDIA_PLATFORM 1
#   define INTEL_PLATFORM 0
#   define AMD_PLATFORM 0
#elif defined(cl_intel_subgroups)
#   define NVIDIA_PLATFORM 0
#   define INTEL_PLATFORM 1
#   define AMD_PLATFORM 0
#elif defined(__AMD__)
#   define NVIDIA_PLATFORM 0
#   define INTEL_PLATFORM 0
#   define AMD_PLATFORM 1
#else
#   define NVIDIA_PLATFORM 0
#   define INTEL_PLATFORM 0
#   define AMD_PLATFORM 0
#endif

#if AMD_PLATFORM == 1
#   define USE_KSHIFT 1
#else
#   define USE_KSHIFT 0
#endif

float16 load_4x4_tile_and_transform(__global const float * restrict channel,int stride, 
                                                int H, int W,
                                                int row, int col)
{
    float4 a[4];
    __global const float * frame = channel + row * stride + col;
    
    int r=row;
    #pragma unroll
    for(int i=0;i<4;i++,r++,frame+=stride) {
        if(r >= 0 && r < H) {
            if(col >= 0 && col + 3 < W) {
                a[i] = vload4(0,frame);
            }
            else {
                int c=col;
                a[i].s0 = c >= 0 && c < W ? frame[0] : 0.0f;
                c++;
                a[i].s1 = c >= 0 && c < W ? frame[1] : 0.0f;
                c++;
                a[i].s2 = c >= 0 && c < W ? frame[2] : 0.0f;
                c++;
                a[i].s3 = c >= 0 && c < W ? frame[3] : 0.0f;
                c++;
            }
        }
        else {
            a[i] = 0.0f;
        }
    }


    
    float bta[4][4];

    bta[0][0] = a[0].s0 - a[2].s0;
    bta[0][1] = a[0].s1 - a[2].s1;
    bta[0][2] = a[0].s2 - a[2].s2;
    bta[0][3] = a[0].s3 - a[2].s3;

    bta[1][0] = a[1].s0 + a[2].s0;
    bta[1][1] = a[1].s1 + a[2].s1;
    bta[1][2] = a[1].s2 + a[2].s2;
    bta[1][3] = a[1].s3 + a[2].s3;

    bta[2][0] = a[2].s0 - a[1].s0;
    bta[2][1] = a[2].s1 - a[1].s1;
    bta[2][2] = a[2].s2 - a[1].s2;
    bta[2][3] = a[2].s3 - a[1].s3;

    bta[3][0] = a[1].s0 - a[3].s0;
    bta[3][1] = a[1].s1 - a[3].s1;
    bta[3][2] = a[1].s2 - a[3].s2;
    bta[3][3] = a[1].s3 - a[3].s3;

    float4 btab[4];
    #pragma unroll
    for(int i=0;i<4;i++) {
        btab[i].s0 = bta[i][0] - bta[i][2];
        btab[i].s1 = bta[i][1] + bta[i][2];
        btab[i].s2 = bta[i][2] - bta[i][1];
        btab[i].s3 = bta[i][1] - bta[i][3];
    }
    
    return (float16)(btab[0],btab[1],btab[2],btab[3]);
}

float16 load_3x3_kernel_and_transform(__global const float *kern_ptr)
{
    float4 gk[3];
    float k[9];
    
    #pragma unroll
    for(int i=0;i<9;i++)
        k[i]=kern_ptr[i];

    gk[0].s0 = k[0];
    gk[1].s0 = k[1];
    gk[2].s0 = k[2];

    gk[0].s1 = 0.5f * (k[0] + k[3] + k[6]);
    gk[1].s1 = 0.5f * (k[1] + k[4] + k[7]);
    gk[2].s1 = 0.5f * (k[2] + k[5] + k[8]);

    gk[0].s2 = 0.5f * (k[0] - k[3] + k[6]);
    gk[1].s2 = 0.5f * (k[1] - k[4] + k[7]);
    gk[2].s2 = 0.5f * (k[2] - k[5] + k[8]);

    gk[0].s3 = k[6];
    gk[1].s3 = k[7];
    gk[2].s3 = k[8];

    float16 k4;

    k4.s048c = gk[0];
    k4.s159d = 0.5f * (gk[0] + gk[1] + gk[2]);
    k4.s26ae = 0.5f * (gk[0] - gk[1] + gk[2]);
    k4.s37bf = gk[2];
    return k4;
}

float4 tile4x4_after_wingorad_to_2x2(float16 tile)
{
    float4 p0 = tile.s0123;
    float4 p1 = tile.s4567;
    float4 p2 = tile.s89ab;
    float4 p3 = tile.scdef;

    float4 Atp[2];
    Atp[0] = p0 + p1 + p2;
    Atp[1] = p1 - p2 - p3;
    
    float2 r[2];

    #pragma unroll
    for(int i=0;i<2;i++) {
        r[i].s0 = Atp[i].s0 + Atp[i].s1 + Atp[i].s2;
        r[i].s1 = Atp[i].s1 - Atp[i].s2 - Atp[i].s3;
    }

    return (float4)(r[0],r[1]);
}


__kernel void winconv_calc_gkgt_3x3(int N,int C,__global const float * restrict gk3,__global float16 *k4)
{
    int n = get_global_id(0);
    int c = get_global_id(1);
    if(n >= N || c>= C)
        return;
    float16 kern = load_3x3_kernel_and_transform(gk3 + (C * n + c) * 9);
#if REV_KERNELS == 1
    k4[N * c + n] = kern;
#else
    k4[C * n + c] = kern;
#endif
}


#define WG_SIZE 256
#define TILES_IN_WG 32
#define KERNELS_IN_WG 32
#define WG_K 8

#if  WG_K * TILES_IN_WG  != WG_SIZE || KERNELS_IN_WG * WG_K != WG_SIZE
#error "Parameters do not match"
#endif

#define WG_DIM 0

void store_local(__local float *l_val,int strd,float16 v)
{
        l_val[ 0*strd] = v.s0;
        l_val[ 1*strd] = v.s1;
        l_val[ 2*strd] = v.s2;
        l_val[ 3*strd] = v.s3;
        l_val[ 4*strd] = v.s4;
        l_val[ 5*strd] = v.s5;
        l_val[ 6*strd] = v.s6;
        l_val[ 7*strd] = v.s7;
        l_val[ 8*strd] = v.s8;
        l_val[ 9*strd] = v.s9;
        l_val[10*strd] = v.sa;
        l_val[11*strd] = v.sb;
        l_val[12*strd] = v.sc;
        l_val[13*strd] = v.sd;
        l_val[14*strd] = v.se;
        l_val[15*strd] = v.sf;
}

#define PAD_H 1
#define PAD_W 1

#define PATCH_K 8
#define PATCH_T 8

#if TILES_IN_WG * KERNELS_IN_WG * 16 != PATCH_K * PATCH_T * WG_SIZE
#error
#endif

unsigned z_order_a(unsigned x)
{
    return ( (x & (1<<0))  >> 0 )
         | ( (x & (1<<2))  >> 1 )  
         | ( (x & (1<<4))  >> 2 )  
         | ( (x & (1<<6))  >> 3 );
}

unsigned z_order_b(unsigned x)
{
    return z_order_a(x>>1);
}

__kernel 
__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void winconv_3x3(int B, int N,int C,int H,int W,
                  __global float const * restrict image,
                  __global float16 const * restrict kernels,
                  __global float *restrict result)
{
    int half_W = (W+1)/2;
    int half_H = (H+1)/2;
    int half_WG = half_W * half_H;

    __local float wg_local_memory[(TILES_IN_WG + KERNELS_IN_WG) * WG_K * 16];

#define l_tiles(a,b,c) wg_local_memory[(((a)*WG_K + (b))*TILES_IN_WG) + (c)]
#define l_kernels(a,b,c) wg_local_memory[(((a)*WG_K + (b))*KERNELS_IN_WG) + (c) + (TILES_IN_WG*WG_K*16)]


    // Loading data
    int l_tile_rc = get_local_id(WG_DIM) % TILES_IN_WG;
    int l_tile_k  = get_local_id(WG_DIM) / TILES_IN_WG;

    int l_kern_n  = get_local_id(WG_DIM) % KERNELS_IN_WG;
    int l_kern_k  = get_local_id(WG_DIM) / KERNELS_IN_WG;
   
    int wg_brc    = get_group_id(0) * TILES_IN_WG;
    int wg_feature= get_global_id(1) * KERNELS_IN_WG; 

    int l_brc     = wg_brc + l_tile_rc;
    int l_feature = wg_feature + l_kern_n;

    int l_b   = l_brc / half_WG;
    int l_rc  = l_brc % half_WG;
    int l_row = l_rc / half_W * 2;
    int l_col = l_rc % half_W * 2;

    image += l_b * H * W * C;


    #define l_tile_stride (TILES_IN_WG * WG_K)
    #define l_kern_stride (KERNELS_IN_WG * WG_K)

    #define s_img_tile(k,t,indx) wg_local_memory[(k)*(TILES_IN_WG/2)*16 + (t/2) * 16 + (indx)]
    
    __local float *l_tile_ptr = &l_tiles(0,l_tile_k,l_tile_rc);
    __local float *l_kern_ptr = &l_kernels(0,l_kern_k,l_kern_n);

    // GEMM

    int my_gemm_tile_b  = get_local_id(WG_DIM) / 16;
    int my_gemm_tile_tk = get_local_id(WG_DIM) % 16;
    //int my_gemm_tile_kr = z_order_a(my_gemm_tile_tk) * PATCH_K; 
    //int my_gemm_tile_tl = z_order_b(my_gemm_tile_tk) * PATCH_T;
    int my_gemm_tile_kr = my_gemm_tile_tk / (TILES_IN_WG / PATCH_T) * PATCH_K;
    int my_gemm_tile_tl = my_gemm_tile_tk % (TILES_IN_WG / PATCH_T) * PATCH_T;

    //printf("%d %d\n",my_gemm_tile_kr,my_gemm_tile_tl);

    float p_C[PATCH_K][PATCH_T]={{0}};

    // store

    for(int k=0;k < C;k+=WG_K) {

        // load relevant tile        
        float16 my_tile = (l_b < B && k + l_tile_k < C) ? load_4x4_tile_and_transform(image + (k + l_tile_k) * (H * W),W,H,W,l_row - PAD_H,l_col - PAD_W) : 0; 
        store_local(l_tile_ptr,l_tile_stride,my_tile);
        
        // load relevant kernel
        float16 my_kern = (l_feature < N && k+l_kern_k < C) ? kernels[l_feature * C + k + l_kern_k] : 0;
        store_local(l_kern_ptr,l_kern_stride,my_kern);
/*
        if(get_local_id(WG_DIM) == 0) {
            printf("%f %f %f %f\n",my_tile.s0,my_tile.s1,my_tile.s2,my_tile.s3);
            printf("%f %f %f %f\n",my_tile.s4,my_tile.s5,my_tile.s6,my_tile.s7);
            printf("%f %f %f %f\n",my_tile.s8,my_tile.s9,my_tile.sa,my_tile.sb);
            printf("%f %f %f %f\n",my_tile.sc,my_tile.sd,my_tile.se,my_tile.sf);
            printf("---\n");
            printf("%f %f %f %f\n",my_kern.s0,my_kern.s1,my_kern.s2,my_kern.s3);
            printf("%f %f %f %f\n",my_kern.s4,my_kern.s5,my_kern.s6,my_kern.s7);
            printf("%f %f %f %f\n",my_kern.s8,my_kern.s9,my_kern.sa,my_kern.sb);
            printf("%f %f %f %f\n",my_kern.sc,my_kern.sd,my_kern.se,my_kern.sf);
        }
*/

        barrier(CLK_LOCAL_MEM_FENCE);

        // GEMM
        #pragma unroll(4)
        for(int dk=0;dk<WG_K;dk++) {
            float p_kern[PATCH_K];

            #pragma unroll
            for(int dr=0;dr<PATCH_K;dr++) {
                p_kern[dr] = l_kernels(my_gemm_tile_b,dk,dr + my_gemm_tile_kr);
            }
            float p_tile[PATCH_T];
            #pragma unroll
            for(int dc=0;dc<PATCH_T;dc++) {
                p_tile[dc] = l_tiles(my_gemm_tile_b,dk,dc + my_gemm_tile_tl);
            }
            #pragma unroll
            for(int dr=0;dr<PATCH_K;dr++) {
                #pragma unroll
                for(int dc=0;dc<PATCH_T;dc++) {
                    p_C[dr][dc] += p_kern[dr]*p_tile[dc];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    int tile0 = get_local_id(WG_DIM) * 4;
    int s_col_t0 = tile0 % TILES_IN_WG;
    int s_row_t = tile0 / TILES_IN_WG;


    #pragma unroll
    for(int dc_split = 0; dc_split < 2;dc_split++) {
        
        // transpose part A

        #pragma unroll
        for(int dr=0;dr<PATCH_K;dr++) {
            int tile_r = dr + my_gemm_tile_kr;
            #pragma unroll
            for(int dc=0;dc<PATCH_T;dc+=2) {
                int tile_c = dc + dc_split + my_gemm_tile_tl;
                s_img_tile(tile_r,tile_c,my_gemm_tile_b) = p_C[dr][dc + dc_split];
             }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for(int dc = 0;dc < 4;dc+=2) {
            
            int s_col_t = s_col_t0 + dc + dc_split;

            float16 s_tile = vload16(0,&s_img_tile(s_row_t,s_col_t,0));
            /*if(get_local_id(WG_DIM) == 0 && dr1 == 0 && dr2 == 0) {
                printf("R: %d %d\n",s_row_t,s_col_t);
                printf("R %f %f %f %f\n",s_tile.s0,s_tile.s1,s_tile.s2,s_tile.s3);
                printf("R %f %f %f %f\n",s_tile.s4,s_tile.s5,s_tile.s6,s_tile.s7);
                printf("R %f %f %f %f\n",s_tile.s8,s_tile.s9,s_tile.sa,s_tile.sb);
                printf("R %f %f %f %f\n",s_tile.sc,s_tile.sd,s_tile.se,s_tile.sf);
            }*/
            float4 s_img_tile = tile4x4_after_wingorad_to_2x2(s_tile);

            int s_brc = wg_brc + s_col_t;

            int s_b   = s_brc / half_WG;
            int s_rc  = s_brc % half_WG;
            int s_row = s_rc / half_W * 2;
            int s_col = s_rc % half_W * 2;

            int s_feature = wg_feature + s_row_t;

            __global float *ptr = result + ((s_b * N + s_feature) * H + s_row) * W + s_col;


            if(s_b < B && s_feature < N) {
                /*if(s_b == 0 && s_feature == 0 && s_row == 6 && s_col == 6)  {
                    printf("%f %f\n%f %f %d\n",s_img_tile.s0,s_img_tile.s1,s_img_tile.s2,s_img_tile.s3);
                }*/
                bool rv0 = s_row < H;
                bool rv1 = s_row + 1 < H;
                bool cv0 = s_col < W;
                bool cv1 = s_col + 1 < W;
                if(rv0) {
                    if(cv0)
                        ptr[0] = s_img_tile.s0;
                    if(cv1)
                        ptr[1] = s_img_tile.s1;
                }
                ptr += W;
                if(rv1) {
                    if(cv0)
                        ptr[0] = s_img_tile.s2;
                    if(cv1)
                        ptr[1] = s_img_tile.s3;
                }
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }

}

