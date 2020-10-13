#define SIM 0
#define INV_DK 0
#define REV_KERNELS 1
#define USE_LOCAL_MEM 1
#define USE_KSHIFT 1 

#define LOCAL_TMEM_PAD 4
#define LOCAL_TMEM_PAD0 0
#define LOCAL_TMEM_PADX 1

#define LOCAL_KMEM_PAD 1
#define LOCAL_KMEM_PAD0 0
#define SPLIT_LOAD 1


#define INTEL_PLATFORM 0
#define AMD_PLATFORM 1

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
            a[i] = (float4)(0.0f,0.0f,0.0f,0.0f);
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
#if INTEL_PLATFORM == 1
#define shuffle_x4(value,flag) intel_sub_group_shuffle((value),(get_sub_group_local_id() & 0xFC) | (((flag) >> ((get_sub_group_local_id() & 3) << 1)) & 3))
#elif AMD_PLATFORM == 1
#define shuffle_x4(value,flag) as_float(__builtin_amdgcn_mov_dpp(as_int(value),flag,0xF,0xF,0))
#endif

float4 load_4x4_tile_and_transform_x4(__global const float * restrict channel,int stride, 
                                                int H, int W,
                                                int row, int col)
{
    
    int l_id = get_local_id(0) % 4;
    __global const float * frame = channel + row * stride + col;

    frame += l_id * stride;
    int r=row + l_id;

    float4 my_a;
    if(r >= 0 && r < H) {
        if(col >= 0 && col + 3 < W) {
            my_a = vload4(0,frame);
        }
        else {
            int c=col;
            my_a.s0 = c >= 0 && c < W ? frame[0] : 0.0f;
            c++;
            my_a.s1 = c >= 0 && c < W ? frame[1] : 0.0f;
            c++;
            my_a.s2 = c >= 0 && c < W ? frame[2] : 0.0f;
            c++;
            my_a.s3 = c >= 0 && c < W ? frame[3] : 0.0f;
            c++;
        }
    }
    else {
        my_a = 0.0f;
    }
    
    float4 p1,p2;
    
    #define sel_a (0 + (1 << 2) + (2 << 4) + (1 << 6))
    #define sel_b (2 + (2 << 2) + (1 << 4) + (3 << 6))
    const unsigned int neg_b = 1 + (0 << 1) + (1 << 2) + (1 << 3);

    p1.s0 = shuffle_x4(my_a.s0,sel_a);
    p1.s1 = shuffle_x4(my_a.s1,sel_a);
    p1.s2 = shuffle_x4(my_a.s2,sel_a);
    p1.s3 = shuffle_x4(my_a.s3,sel_a);

    p2.s0 = shuffle_x4(my_a.s0,sel_b);
    p2.s1 = shuffle_x4(my_a.s1,sel_b);
    p2.s2 = shuffle_x4(my_a.s2,sel_b);
    p2.s3 = shuffle_x4(my_a.s3,sel_b);


    float sign = (1 & (neg_b >> l_id)) ? -1.0 : 1.0;

    float4 bta = p1 + sign * p2;
    float4 btab;
    
    btab.s0 = bta.s0 - bta.s2;
    btab.s1 = bta.s1 + bta.s2;
    btab.s2 = bta.s2 - bta.s1;
    btab.s3 = bta.s1 - bta.s3;;
    
    return btab;
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

__kernel void winconv_im2tile_4x4(int BC,int H, int W,int pH,int pW,
                                    int h_tiles,int w_tiles,
                                    __global const float * restrict data,
                                    __global float16 *tiles)
{
    int bc  = get_global_id(0);
    int r = get_global_id(1);
    int c = get_global_id(2);
    if(bc >= BC || r >= h_tiles || c>= w_tiles)
        return;
    int row = r * 2 - pH;
    int col = c * 2 - pW;
    float16 tile = load_4x4_tile_and_transform(data + bc*H*W,W,H,W,row,col);
    tiles[bc * w_tiles * h_tiles + r*w_tiles + c] = tile;
}


__kernel void winconv_im2tile_4x4_x4(int BC,int H, int W,int pH,int pW,
                                    int h_tiles,int w_tiles,
                                    __global const float * restrict data,
                                    __global float4 *tiles)
{
    int bc  = get_global_id(0) / 4;
    int r = get_global_id(1);
    int c = get_global_id(2);
    if(bc >= BC || r >= h_tiles || c>= w_tiles)
        return;
    int row = r * 2 - pH;
    int col = c * 2 - pW;
    float4 tile = load_4x4_tile_and_transform_x4(data + bc*H*W,W,H,W,row,col);
    tiles[4*(bc * w_tiles * h_tiles + r*w_tiles + c) + get_local_id(0) % 4] = tile;
}




#ifndef WG_DIM_TILES
#define WG_DIM_TILES 8
#endif

#ifndef WG_DIM_KERNELS
#define WG_DIM_KERNELS 8
#endif

#ifndef TILES_IN_WG
#define TILES_IN_WG 8
#endif
#ifndef KERNELS_IN_WG
#define KERNELS_IN_WG 8
#endif

#define TILES_IN_BLOCK (TILES_IN_WG / WG_DIM_TILES)
#define KERNELS_IN_BLOCK (KERNELS_IN_WG / WG_DIM_KERNELS)

#ifndef SUM_OF_4 
#define SUM_OF_4 0
#endif

#define THREADS (WG_DIM_TILES * WG_DIM_KERNELS)

#if 1

#define TILE_ITEMS_PER_THREAD (TILES_IN_WG * 16 / THREADS)
#define KERNEL_ITEMS_PER_THREAD (KERNELS_IN_WG * 16 / THREADS)
#if USE_KLOCAL_MEM == 0

void load_tiles_priv(float res[TILE_ITEMS_PER_THREAD],__global float const *tiles_base,int wg_tile_id,int total_tiles)
{
    int local_wg_id = get_local_id(1) + WG_DIM_TILES * get_local_id(2);
    int local_tile_id     = local_wg_id / (16 / TILE_ITEMS_PER_THREAD );
    int local_tile_offset = local_wg_id % (16 / TILE_ITEMS_PER_THREAD ) * TILE_ITEMS_PER_THREAD;

    __global float const *ldptr = tiles_base + local_tile_id * 16 + local_tile_offset;

#if SIM == 1
    #pragma unroll
    for(int i=0;i<TILE_ITEMS_PER_THREAD;i++)
        res[i] = i;
#else

    if(local_tile_id + wg_tile_id < total_tiles) {
    #if TILE_ITEMS_PER_THREAD == 16
        float16 val = vload16(0,ldptr);
    #elif TILE_ITEMS_PER_THREAD == 8
        float8 val = vload8(0,ldptr);
    #elif TILE_ITEMS_PER_THREAD == 4
        float4 val = vload4(0,ldptr);
    #elif TILE_ITEMS_PER_THREAD == 2
        float2 val = vload2(0,ldptr);
    #elif TILE_ITEMS_PER_THREAD == 1
        res[0] = *ldptr;
    #else
        #error "Unsupported configuration"
    #endif
    #if TILE_ITEMS_PER_THREAD >= 2
        res[0] = val.s0;
        res[1] = val.s1;
    #endif
    #if TILE_ITEMS_PER_THREAD >= 4
        res[2] = val.s2;
        res[3] = val.s3;
    #endif
    #if TILE_ITEMS_PER_THREAD >= 8
        res[4] = val.s4;
        res[5] = val.s5;
        res[6] = val.s6;
        res[7] = val.s7;
    #endif
    #if TILE_ITEMS_PER_THREAD >= 16
        res[ 8] = val.s9;
        res[ 9] = val.s9;
        res[10] = val.sa;
        res[11] = val.sb;
        res[12] = val.sc;
        res[13] = val.sd;
        res[14] = val.se;
        res[15] = val.sf;
    #endif
    }
    else {
        #pragma unroll
        for(int i=0;i<TILE_ITEMS_PER_THREAD;i++)
            res[i] = 0.0f;
    }
#endif
}
#endif

#if (16 % KERNEL_ITEMS_PER_THREAD) != 0 || (16 % TILE_ITEMS_PER_THREAD) != 0
#error "Invalid config"
#endif

#define KERNEL_ITEMS_PER_WINMAT (16 / KERNEL_ITEMS_PER_THREAD)
#define TILE_ITEMS_PER_WINMAT (16 / TILE_ITEMS_PER_THREAD)

#if TILE_ITEMS_PER_THREAD == 16
#define vstoreT(x,y,z) vstore16(x,y,z)
#define vloadT(x,y) vload16(x,y)
#elif TILE_ITEMS_PER_THREAD == 8
#define vstoreT(x,y,z) vstore8(x,y,z)
#define vloadT(x,y) vload8(x,y)
#elif TILE_ITEMS_PER_THREAD == 4
#define vstoreT(x,y,z) vstore4(x,y,z)
#define vloadT(x,y) vload4(x,y)
#elif TILE_ITEMS_PER_THREAD == 2
#define vstoreT(x,y,z) vstore2(x,y,z)
#define vloadT(x,y) vload2(x,y)
#elif TILE_ITEMS_PER_THREAD == 1
#define vstoreT(x,y,z) z[(y)] = (x)
#define vloadT(x,y) y[x]
#else
#error "Unsupported configuration"
#endif

#if KERNEL_ITEMS_PER_THREAD == 16
#define vstoreK(x,y,z) vstore16(x,y,z)
#define vloadK(x,y,z) vload16(x,y,z)
#elif KERNEL_ITEMS_PER_THREAD == 8
#define vstoreK(x,y,z) vstore8(x,y,z)
#define vloadK(x,y,z) vload8(x,y,z)
#elif KERNEL_ITEMS_PER_THREAD == 4
#define vstoreK(x,y,z) vstore4(x,y,z)
#define vloadK(x,y,z) vload4(x,y,z)
#elif KERNEL_ITEMS_PER_THREAD == 2
#define vstoreK(x,y,z) vstore2(x,y,z)
#define vloadK(x,y,z) vload2(x,y,z)
#elif KERNEL_ITEMS_PER_THREAD == 1
#define vstoreK(x,y,z) z[(y)] = (x)
#define vloadK(x,y) y[x]
#else
#error "Unsupported configuration"
#endif


#define LOCAL_TILES_D1 (TILE_ITEMS_PER_THREAD + LOCAL_TMEM_PAD)
#define LOCAL_TILES_D2 (TILE_ITEMS_PER_WINMAT + LOCAL_TMEM_PAD0)
#define LOCAL_TILES_F2 (LOCAL_TILES_D1 * LOCAL_TILES_D2 + LOCAL_TMEM_PADX)
#define ltiles(a,b,c) local_tiles[(a) * LOCAL_TILES_F2 + (b) * LOCAL_TILES_D1 + (c)] 
#define LOCAL_TILES_SIZE (TILES_IN_WG * LOCAL_TILES_F2)

void load_tiles(__local float local_tiles[LOCAL_TILES_SIZE],
                __global float const *tiles_base,int wg_tile_id,int total_tiles,
                    int local_tile_id,int local_tile_offset)
{

    __global float const *ldptr = tiles_base + local_tile_id * 16 + local_tile_offset * TILE_ITEMS_PER_THREAD;

    __local  float *stptr = &ltiles(local_tile_id,local_tile_offset,0);
#if SIM == 1
    #if TILE_ITEMS_PER_THREAD == 8
        vstore8((float8)(1,2,3,4,5,6,7,8),0,stptr);
    #elif TILE_ITEMS_PER_THREAD == 4
        vstore4((float4)(1,2,3,4),0,stptr);
    #elif TILE_ITEMS_PER_THREAD == 2
        vstore2((float2)(0,0),0,stptr);
    #else
        #error "Unsupported configuration"
    #endif
    
#else
    if(local_tile_id + wg_tile_id < total_tiles) {
        //int off = get_local_id(1) % TILE_ITEMS_PER_THREAD;
        //#pragma unroll
        //for(int i=0;i<TILE_ITEMS_PER_THREAD;i++)
        //    stptr[i] = ldptr[i];
        //if(get_group_id(0) == 0 && get_group_id(1) == 0 && get_group_id(2) == 0) {
        //    int th = get_local_id(1) + WG_DIM_TILES * get_local_id(2);
        //    printf("%d -> %d\n", th,((int)(stptr) / 4) % 32);
        //}
        vstoreT(vloadT(0,ldptr),0,stptr);
    }
    else {
        #pragma unroll
        for(int i=0;i<TILE_ITEMS_PER_THREAD;i++)
            stptr[i] = 0.0f;
    }
#endif
}


#if USE_LOCAL_MEM == 0
void load_kernels_priv(float res[KERNEL_ITEMS_PER_THREAD],__global float const *kernels_base,int stride_floats,int wg_kernel_id,int total_kernels)
{
    int local_wg_id = get_local_id(1) + WG_DIM_TILES * get_local_id(2);
    int local_kern_id     = local_wg_id / (16 / KERNEL_ITEMS_PER_THREAD );
    int local_kern_offset = local_wg_id % (16 / KERNEL_ITEMS_PER_THREAD ) * KERNEL_ITEMS_PER_THREAD;

    __global float const *ldptr = kernels_base + local_kern_id * stride_floats + local_kern_offset;

#if SIM == 1
    #pragma unroll
    for(int i=0;i<KERNEL_ITEMS_PER_THREAD;i+=1)
        res[i] = i;
#else

    if(local_kern_id + wg_kernel_id < total_kernels) {
    #if KERNEL_ITEMS_PER_THREAD == 16
        float16 val = vload16(0,ldptr);
    #elif KERNEL_ITEMS_PER_THREAD == 8
        float8 val = vload8(0,ldptr);
    #elif KERNEL_ITEMS_PER_THREAD == 4
        float4 val = vload4(0,ldptr);
    #elif KERNEL_ITEMS_PER_THREAD == 2
        float2 val = vload2(0,ldptr);
    #elif KERNEL_ITEMS_PER_THREAD == 1
        res[0] = *ldptr;
    #else
        #error "Unsupported configuration"
    #endif
    #if KERNEL_ITEMS_PER_THREAD >= 2
        res[0] = val.s0;
        res[1] = val.s1;
    #endif
    #if KERNEL_ITEMS_PER_THREAD >= 4
        res[2] = val.s2;
        res[3] = val.s3;
    #endif
    #if KERNEL_ITEMS_PER_THREAD >= 8
        res[4] = val.s4;
        res[5] = val.s5;
        res[6] = val.s6;
        res[7] = val.s7;
    #endif
    #if KERNEL_ITEMS_PER_THREAD >= 16
        res[ 8] = val.s9;
        res[ 9] = val.s9;
        res[10] = val.sa;
        res[11] = val.sb;
        res[12] = val.sc;
        res[13] = val.sd;
        res[14] = val.se;
        res[15] = val.sf;
    #endif
    }
    else {
        #pragma unroll
        for(int i=0;i<KERNEL_ITEMS_PER_THREAD;i++)
            res[i] = 0.0f;
    }
#endif
}

#endif
void load_kernels(__local float local_kernels[KERNELS_IN_WG][KERNEL_ITEMS_PER_WINMAT + LOCAL_KMEM_PAD0][KERNEL_ITEMS_PER_THREAD+LOCAL_KMEM_PAD],__global float const *kernels_base,int stride_floats,int wg_kernel_id,int total_kernels,
    int local_kern_id,int local_kern_offset)
{
    __global float const *ldptr = kernels_base + local_kern_id * stride_floats + local_kern_offset * KERNEL_ITEMS_PER_THREAD;
    __local  float *stptr = local_kernels[local_kern_id][local_kern_offset];

#if SIM == 1
    #if KERNEL_ITEMS_PER_THREAD == 8
        vstore8((float8)(1,2,3,4,5,6,7,8),0,stptr);
    #elif KERNEL_ITEMS_PER_THREAD == 4
        vstore4((float4)(1,2,3,4),0,stptr);
    #elif KERNEL_ITEMS_PER_THREAD == 2
        vstore2((float2)(1,2),0,stptr);
    #else
        #pragma unroll
        for(int i=0;i<KERNEL_ITEMS_PER_THREAD;i++)
            stptr[i]=i;
    #endif
#else    

    if(local_kern_id + wg_kernel_id < total_kernels) {
        vstoreT(vloadT(0,ldptr),0,stptr);
    }
    else {
        #pragma unroll
        for(int i=0;i<KERNEL_ITEMS_PER_THREAD;i++)
            stptr[i] = 0.0f;
    }
#endif
}




#if INTEL_PLATFORM == 1
#define mad_permute(sum,mp1,mp_per,line) sum = mad(mp1,intel_sub_group_shuffle(mp_per,(get_sub_group_local_id() & 0x1c) | line),sum)
#elif AMD_PLATFORM == 1
#define mad_permute(sum,mp1,mp_per,line) \
    __asm__ volatile ("v_mac_f32_dpp %0,%1,%2 quad_perm:[" #line "," #line  "," #line ","#line "] row_mask:0xf bank_mask:0xf " \
                    : "+v" (sum) : "v" (mp_per) , "v" (mp1)   )
#endif


__kernel 
__attribute__((reqd_work_group_size(1,WG_DIM_TILES, WG_DIM_KERNELS)))
//__attribute__((intel_reqd_sub_group_size(8)))
void winconv_3x3(int B, int C,int oC, int oH, int oW,int tilesH,int tilesW,
                          __global float const * restrict tiles,
                          __global float const * restrict kernels,
                          __global float *restrict result)
{
    #if USE_LOCAL_MEM == 1
    // +1 to prevent bank collisions
    __local float local_tiles[LOCAL_TILES_SIZE]; 
    __local float local_kernels[KERNELS_IN_WG][KERNEL_ITEMS_PER_WINMAT + LOCAL_KMEM_PAD0][KERNEL_ITEMS_PER_THREAD + LOCAL_KMEM_PAD];
    #endif
    float16 s[TILES_IN_BLOCK][KERNELS_IN_BLOCK]={};

    int batch = get_global_id(0);

    int tile_row_col = get_global_id(1) * TILES_IN_BLOCK;
    int tile_wg_id   = get_group_id(1) * TILES_IN_WG;
    int tiles_no = tilesH*tilesW;
    int tile_row = tile_row_col / tilesW;
    int tile_col = tile_row_col % tilesW;
    int tile_local_base = get_local_id(1) * TILES_IN_BLOCK;

    int kernel_id = get_global_id(2) * KERNELS_IN_BLOCK;
    int kernel_wg_id = get_group_id(2) * KERNELS_IN_WG;
    int kernel_local_base = get_local_id(2) * KERNELS_IN_BLOCK;

#if USE_LOCAL_MEM == 1
    int local_wg_id = get_local_id(1) + WG_DIM_TILES * get_local_id(2);
    
    int local_tile_id     = local_wg_id / (16 / TILE_ITEMS_PER_THREAD );
    int local_tile_offset = local_wg_id % (16 / TILE_ITEMS_PER_THREAD );

    int local_kern_id     = local_wg_id / (16 / KERNEL_ITEMS_PER_THREAD );
    int local_kern_offset = local_wg_id % (16 / KERNEL_ITEMS_PER_THREAD );
#endif


    int tiles2d = tilesH * tilesW;
    __global float const * tile_base    =  tiles   + 16*(batch * tiles2d * C + tile_wg_id);
#if REV_KERNELS == 1
    __global float const * kernels_base = kernels + 16*kernel_wg_id;
    const int kern_step = C * 16;
    const int kernels_stride = 16;
#else
    __global float const * kernels_base = kernels + 16*(C * kernel_wg_id);
    const int kern_step = 16;
    const int kernels_stride = 16 * C;
#endif

    float my_tiles[TILE_ITEMS_PER_THREAD];
    float tvs[KERNELS_IN_BLOCK][16];
    #if USE_KSHIFT==1
    float4 kr;
    float4 krs[KERNELS_IN_BLOCK];
    #else
    float16 kr;
    float16 krs[KERNELS_IN_BLOCK];
    #endif
    for(int channel=0;channel<C;channel++,tile_base += tiles2d*16,kernels_base +=kern_step) {
        
        #if USE_LOCAL_MEM == 1
        if(1){  
            load_tiles(local_tiles,tile_base,tile_wg_id,tiles_no,local_tile_id,local_tile_offset);
            load_kernels(local_kernels,kernels_base,kernels_stride,kernel_wg_id,oC,local_kern_id,local_kern_offset);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        //continue;
        
         
        #pragma unroll
        for(int dt=0;dt<TILES_IN_BLOCK;dt++) {
            #pragma unroll
            for(int i=0;i<TILE_ITEMS_PER_WINMAT;i++) { 
                vstoreT(vloadT(0,&ltiles(tile_local_base + dt,i,0)),i,tvs[dt]);
            }
        }


        #else
        #if THREADS != 64
        #error "Code Requires WG size = 64
        #endif



        {
            load_tiles_priv(my_tiles,tile_base,tile_wg_id,tiles_no);

            #pragma unroll
            for(int dt=0;dt<TILES_IN_BLOCK;dt++) {
                int thread_addr = 4 * (tile_local_base + dt) * (16 / TILE_ITEMS_PER_THREAD);
                #pragma unroll
                for(int i=0;i < 16;i++) {
                    int thread_offset = 4 * (i / TILE_ITEMS_PER_THREAD);
                    int my_offset = i % TILE_ITEMS_PER_THREAD;
                    tvs[dt][i] = as_float(__builtin_amdgcn_ds_bpermute(thread_addr + thread_offset,as_int(my_tiles[my_offset])));
                }
            }
        }
        #endif // local memory vs data sharing
       
        // load kernels local memory 
        #pragma unroll
        for(int dk=0;dk<KERNELS_IN_BLOCK;dk++) {
            #if USE_LOCAL_MEM == 1
                #if USE_KSHIFT==1
                    int my_id = get_local_id(1) % 4;
                    #if TILE_ITEMS_PER_WINMAT == 4
                    kr = vload4(0,local_kernels[kernel_local_base + dk][my_id]);
                    #elif TILE_ITEMS_PER_WINMAT == 8
                    kr.s01 = vload2(0,local_kernels[kernel_local_base + dk][my_id*2]);
                    kr.s23 = vload2(0,local_kernels[kernel_local_base + dk][my_id*2+1]);
                    #elif TILE_ITEMS_PER_WINMAT == 2
                    kr = vload4(my_id % 2,local_kernels[kernel_local_base + dk][my_id / 2]);
                    #elif TILE_ITEMS_PER_WINMAT == 16
                    kr.s0 = local_kernels[kernel_local_base + dk][my_id*4+0][0];
                    kr.s1 = local_kernels[kernel_local_base + dk][my_id*4+0][1];
                    kr.s2 = local_kernels[kernel_local_base + dk][my_id*4+0][2];
                    kr.s3 = local_kernels[kernel_local_base + dk][my_id*4+0][3];
                    #else
                    #error
                    #endif
                #else
                    kr = vload16(0,local_kernels[kernel_local_base + dk]);
                #endif
            #else
                #if INV_DK==1
                    int current_id = kernel_wg_id + WG_DIM_KERNELS * dk + get_local_id(2);
                    if(current_id < oC) {
                        int my_id = get_local_id(1) % 4 * 4;
                        kr = vload4(0,kernels_base + kernels_stride * current_id + my_id );
                    }
                    else {
                        kr = 0.0f;
                    }
                #else
                    if(kernel_id + dk < oC) {
                        int my_id = get_local_id(1) % 4 * 4;
                        kr = vload4(0,kernels_base + kernels_stride * (kernel_local_base + dk) + my_id );
                    }
                    else {
                        kr = 0.0f;
                    }
                    
                #endif
            #endif
            krs[dk]=kr;
        }
        #pragma unroll
        for(int dk=0;dk<KERNELS_IN_BLOCK;dk++) {
            kr = krs[dk];
            
            #pragma unroll
            for(int dt=0;dt<TILES_IN_BLOCK;dt++) {
                #if USE_KSHIFT==1
                mad_permute(s[dt][dk].s0,tvs[dt][0],kr.s0,0);
                mad_permute(s[dt][dk].s1,tvs[dt][1],kr.s1,0);
                mad_permute(s[dt][dk].s2,tvs[dt][2],kr.s2,0);
                mad_permute(s[dt][dk].s3,tvs[dt][3],kr.s3,0);

                mad_permute(s[dt][dk].s4,tvs[dt][4],kr.s0,1);
                mad_permute(s[dt][dk].s5,tvs[dt][5],kr.s1,1);
                mad_permute(s[dt][dk].s6,tvs[dt][6],kr.s2,1);
                mad_permute(s[dt][dk].s7,tvs[dt][7],kr.s3,1);

                mad_permute(s[dt][dk].s8,tvs[dt][8],kr.s0,2);
                mad_permute(s[dt][dk].s9,tvs[dt][9],kr.s1,2);
                mad_permute(s[dt][dk].sa,tvs[dt][10],kr.s2,2);
                mad_permute(s[dt][dk].sb,tvs[dt][11],kr.s3,2);

                mad_permute(s[dt][dk].sc,tvs[dt][12],kr.s0,3);
                mad_permute(s[dt][dk].sd,tvs[dt][13],kr.s1,3);
                mad_permute(s[dt][dk].se,tvs[dt][14],kr.s2,3);
                mad_permute(s[dt][dk].sf,tvs[dt][15],kr.s3,3);
                #else
                float16 tv=(float16)(
                    tvs[dt][0],tvs[dt][1],tvs[dt][2],tvs[dt][3],
                    tvs[dt][4],tvs[dt][5],tvs[dt][6],tvs[dt][7],
                    tvs[dt][8],tvs[dt][9],tvs[dt][10],tvs[dt][11],
                    tvs[dt][12],tvs[dt][13],tvs[dt][14],tvs[dt][15]);
                s[dt][dk] = mad(tv,kr,s[dt][dk]);
                #endif
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    #if 0
    #pragma unroll
    for(int dk=0;dk<KERNELS_IN_BLOCK;dk++) {
        #pragma unroll
        for(int dt=0;dt<TILES_IN_BLOCK;dt++) {
            float4 tile_res = tile4x4_after_wingorad_to_2x2(s[dt][dk]);
            //float4 tile_res = s[dt][dk].s0123 + s[dt][dk].s4567 + s[dt][dk].s89ab  + s[dt][dk].scdef;
            __asm__ volatile ("v_mac_f32 %0,%1,%2" : "+v"(tile_res.s0) : "v"(tile_res.s0) , "v" (tile_res.s0));
            __asm__ volatile ("v_mac_f32 %0,%1,%2" : "+v"(tile_res.s1) : "v"(tile_res.s1) , "v" (tile_res.s1));
            __asm__ volatile ("v_mac_f32 %0,%1,%2" : "+v"(tile_res.s2) : "v"(tile_res.s2) , "v" (tile_res.s2));
            __asm__ volatile ("v_mac_f32 %0,%1,%2" : "+v"(tile_res.s3) : "v"(tile_res.s3) , "v" (tile_res.s3));
        }
    }
    return;
    #endif


    int oHW = oH*oW;
    #if INV_DK == 0
    int doff0 = oHW;
    int off0= batch*oC*oHW + oHW*kernel_id;
    #else
    int doff0 = oH*oW * WG_DIM_KERNELS;
    int off0= batch*oC*oHW + oHW*(kernel_wg_id +  get_local_id(2));
    #endif
    #pragma unroll
    for(int dk=0;dk<KERNELS_IN_BLOCK;dk++,off0 += doff0) {
        #if INV_DK==1
        int kid = kernel_wg_id + dk * WG_DIM_KERNELS + get_local_id(2);
        #else
        int kid = kernel_id + dk;
        #endif
        if(kid >= oC)
            continue;
        int crow = tile_row;
        int ccol = tile_col;
        #pragma unroll
        for(int dt=0;dt<TILES_IN_BLOCK;dt++) {
            int tid = tile_row_col + dt;
            #if SUM_OF_4 == 0
            float4 tile_res = tile4x4_after_wingorad_to_2x2(s[dt][dk]);
            #else
            float4 tile_res = res[dt][dk];
            #endif
            int offset = off0+ 2 * (crow * oW + ccol);
            __global float *output = result + offset;
            int row = crow*2;
            int col = ccol*2;
            if(row < oH) {
                if(col < oW) {
                    output[0] = tile_res.s0;
                }
                if(col+1 < oW) {
                    output[1] = tile_res.s1;
                }
            }
            output += oW;
            if(row + 1 < oH) {
                if(col < oW) {
                    output[0] = tile_res.s2;
                }
                if(col+1 < oW) {
                    output[1] = tile_res.s3;
                }
            }
            // update for next block
            ccol++;
            if(ccol >= tilesW) {
                crow ++;
                ccol = 0;
            }
        }
    }

}
#elif 0
__kernel 
__attribute__((reqd_work_group_size(1,WG_DIM_TILES, WG_DIM_KERNELS)))
void winconv_3x3(int B, int C,int oC, int oH, int oW,int tilesH,int tilesW,
                          __global float const *tiles,
                          __global float const *kernels,
                          __global float *result)
{
    // +1 to prevent bank collisions
#if SUM_OF_4==0
    float16 s[TILES_IN_BLOCK][KERNELS_IN_BLOCK]={};
#else
    float4  res[TILES_IN_BLOCK][KERNELS_IN_BLOCK]={};
#endif

    int batch = get_global_id(0);

    int tile_row_col = get_global_id(1) * TILES_IN_BLOCK;
    int tile_wg_id   = get_group_id(1) * TILES_IN_WG;
    int tiles_no = tilesH*tilesW;
    int tile_row = tile_row_col / tilesW;
    int tile_col = tile_row_col % tilesW;
    int tile_local_base = get_local_id(1) * TILES_IN_BLOCK;

    int kernel_id = get_global_id(2) * KERNELS_IN_BLOCK;
    int kernel_wg_id = get_group_id(2) * KERNELS_IN_WG;
    int kernel_local_base = get_local_id(2) * KERNELS_IN_BLOCK;

    int tiles2d = tilesH * tilesW;
    __global float const * tile_base    =  tiles   + 16*(batch * tiles2d * C + tile_wg_id);
#if REV_KERNELS == 1
    __global float const * kernels_base = kernels + 16*kernel_wg_id;
    const int kern_step = C * 16;
    const int kernels_stride = 16;
#else
    __global float const * kernels_base = kernels + 16*(C * kernel_wg_id);
    const int kern_step = 16;
    const int kernels_stride = 16 * C;
#endif

    for(int channel=0;channel<C;channel++,tile_base += tiles2d*16,kernels_base +=kern_step) {

        float16 krs[KERNELS_IN_BLOCK];
        #pragma unroll
        for(int dk=0;dk<KERNELS_IN_BLOCK;dk++)
            krs[dk] = kernel_id + dk < oC ? vload16(0,kernels_base + kernels_stride * (kernel_local_base + dk)) : 0.0f;

        #pragma unroll
        for(int dt=0;dt<TILES_IN_BLOCK;dt++) {
            float16 tv = tile_row_col + dt < tiles2d ? vload16(0,tile_base + 16*(tile_local_base + dt)) : 0.0f;
            #pragma unroll
            for(int dk=0;dk<KERNELS_IN_BLOCK;dk++) {
                #if SUM_OF_4 == 1
                res[dt][dk] += tile4x4_after_wingorad_to_2x2(krs[dk]*tv);
                #else   
                s[dt][dk] = mad(tv,krs[dk],s[dt][dk]);         
                #endif
            }
        }

    }

    int oHW = oH*oW;
    int off0= batch*oC*oHW + oHW*kernel_id;
    #pragma unroll
    for(int dk=0;dk<KERNELS_IN_BLOCK;dk++,off0 += oHW) {
        int kid = kernel_id + dk;
        if(kid >= oC)
            continue;
        int crow = tile_row;
        int ccol = tile_col;
        #pragma unroll
        for(int dt=0;dt<TILES_IN_BLOCK;dt++) {
            int tid = tile_row_col + dt;
            #if SUM_OF_4 == 0
            volatile float4 tile_res = tile4x4_after_wingorad_to_2x2(s[dt][dk]);
            #else
            float4 tile_res = res[dt][dk];
            #endif
            int offset = off0+ 2 * (crow * oW + ccol);
            __global float *output = result + offset;
            int row = crow*2;
            int col = ccol*2;
            if(row < oH) {
                if(col < oW) {
                    output[0] = tile_res.s0;
                }
                if(col+1 < oW) {
                    output[1] = tile_res.s1;
                }
            }
            output += oW;
            if(row + 1 < oH) {
                if(col < oW) {
                    output[0] = tile_res.s2;
                }
                if(col+1 < oW) {
                    output[1] = tile_res.s3;
                }
            }
            // update for next block
            ccol++;
            if(ccol >= tilesW) {
                crow ++;
                ccol = 0;
            }
        }
    }

}


#else 
__kernel 
__attribute__((reqd_work_group_size(1,TILES_IN_WG, KERNELS_IN_WG)))
void winconv_3x3(int B, int C,int oC, int oH, int oW,int tilesH,int tilesW,
                          __global float const *tiles,
                          __global float const *kernels,
                          __global float *result)
{
    float s[16]={0.0f};

    int batch = get_global_id(0);

    int tile_row_col = get_global_id(1);
    int tile_wg_id   = get_local_id(1);
    int tiles_no = tilesH*tilesW;
    int row = tile_row_col / tilesW * 2;
    int col = tile_row_col % tilesW * 2;

    int kernel_id = get_global_id(2);
    int kernel_wg_id = get_local_id(2);
    
    int tiles2d = tilesH * tilesW;
    __global float const * tile_base    = tiles   + 16*(batch * tiles2d * C + tile_row_col);
    __global float const * kernels_base = kernels + 16*(C * kernel_id);

    #pragma unroll 4
    for(int channel=0;channel<C;channel++,tile_base += tiles2d*16,kernels_base +=16) {
        const int kstep = 16 / KERNELS_IN_WG;
        const int tstep = 16 / TILES_IN_WG;
        float my_loaded_tile[kstep];
        float my_loaded_kern[tstep];

        if(tile_row_col < tiles2d) {
            #pragma unroll
            for(int i=0;i<kstep;i++)
                my_loaded_tile[i] = tile_base[kernel_wg_id*kstep + i];
        }
        else {
            #pragma unroll
            for(int i=0;i<kstep;i++)
                my_loaded_tile[i] = 0.0f;
        }
        if(kernel_id < oC) {
            #pragma unroll
            for(int i=0;i<tstep;i++)
                my_loaded_kern[i] = kernels_base[tile_wg_id*tstep+i];
        }
        else {
            #pragma unroll
            for(int i=0;i<tstep;i++)
                my_loaded_kern[i] = 0.0f;
        }

#if KERNELS_IN_WG == TILES_IN_WG
        int my_tile_in_perm = tile_wg_id * 4;
        int my_kernel_index = kernel_wg_id * 8 * 4;
        #pragma unroll
        for(int i=0;i<KERNELS_IN_WG;i++,my_tile_in_perm+=KERNELS_IN_WG*4,my_kernel_index+=4) {
            #pragma unroll
            for(int j=0;j<kstep;j++) {
                float t = as_float(__builtin_amdgcn_ds_bpermute(my_tile_in_perm, as_int(my_loaded_tile[j])));
                float k = as_float(__builtin_amdgcn_ds_bpermute(my_kernel_index, as_int(my_loaded_kern[j])));
                s[i*kstep+j] += t*k;
            }
        }
#else
        float my_tile[16];
        float my_kern[16];
        int my_tile_in_perm = tile_wg_id * 4;
        #pragma unroll
        for(int i=0;i<KERNELS_IN_WG;i++,my_tile_in_perm+=KERNELS_IN_WG*4) {
            #pragma unroll
            for(int j=0;j<kstep;j++) {
                my_tile[i*kstep+j] = as_float(__builtin_amdgcn_ds_bpermute(my_tile_in_perm, as_int(my_loaded_tile[j])));
            }
        }

        int my_kernel_index = kernel_wg_id * 8 * 4;
        #pragma unroll
        for(int i=0;i<TILES_IN_WG;i++,my_kernel_index+=4) {
            #pragma unroll
            for(int j=0;j<tstep;j++) {
                my_kern[i*tstep+j] = as_float(__builtin_amdgcn_ds_bpermute( my_kernel_index, as_int(my_loaded_kern[j])));
            }
        }

        #pragma unroll
        for(int i=0;i<16;i++) {
            s[i] = mad(my_tile[i],my_kern[i],s[i]);
        }
#endif
    }

    if(kernel_id >= oC)
        return;

    __global float *gres = result;
    
    float16 sum16={s[0],s[1],s[2],s[3], s[4],s[5],s[6],s[7], s[8],s[9],s[10],s[11], s[12],s[13],s[14],s[15] };

    float4 res = tile4x4_after_wingorad_to_2x2(sum16);
    int offset = batch*oC*oH*oW + oH*oW*kernel_id+ row * oW + col;
    result += offset;

    if(row < oH) {
        if(col < oW) {
            result[0] = res.s0;
        }
        if(col+1 < oW) {
            result[1] = res.s1;
        }
    }
    result += oW;
    if(row + 1 < oH) {
        if(col < oW) {
            result[0] = res.s2;
        }
        if(col+1 < oW) {
            result[1] = res.s3;
        }
    }
}

#endif
