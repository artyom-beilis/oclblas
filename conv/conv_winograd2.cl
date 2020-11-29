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
#if INTEL_PLATFORM == 1
#define shuffle_x4(value,flag) intel_sub_group_shuffle((value),(get_sub_group_local_id() & 0xFC) | (((flag) >> ((get_sub_group_local_id() & 3) << 1)) & 3))
#elif AMD_PLATFORM == 1
#define shuffle_x4(value,flag) as_float(__builtin_amdgcn_mov_dpp(as_int(value),flag,0xF,0xF,0))
#elif NVIDIA_PLATFORM == 1
inline float shuffle_x4_func(float value,int flag)
{
    volatile float r;
    int lane = get_local_id(0) + 4 * get_local_id(1);
    int shift = (get_local_id(0) & 3) << 1;
    int indx =  ((flag >> shift) & 3);
    //asm volatile ("shfl.sync.idx.b32 %0,%1,%2,0x1C1F,0xFFFFFFFF;" : "=r"(r) : "r"(value), "r"(indx));
    asm volatile ("shfl.idx.b32 %0,%1,%2,0x1C1F;" : "=r"(r) : "r"(value), "r"(indx));
    return r;
}
#define shuffle_x4(a,b) shuffle_x4_func(a,b)
#endif


float4 load_4x4_tile_and_transform_x4(__global const float * restrict channel,int stride, 
                                                int H, int W,
                                                int row, int col)
{
    int l_id = get_local_id(0) % 4;
    float4 my_a;

    #define sel_a (0 + (1 << 2) + (2 << 4) + (1 << 6))
    #define sel_b (2 + (2 << 2) + (1 << 4) + (3 << 6))

#ifdef shuffle_x4
    __global const float * frame = channel + row * stride + col;
    frame += l_id * stride;
    int r=row + l_id;
    {
#else
    float4 px[2];
    const int sel_ab = (sel_b << 8) + sel_a;
    #pragma unroll
    for(int i=0;i<2;i++) 
    {
        int sel_x = (sel_ab >> (8*i)) & 0xFF;
        int shift = (sel_x >> (l_id * 2)) & 3;
        __global const float * frame = channel + row * stride + col;
        frame += shift * stride;
        int r=row + shift;

#endif

        if(r >= 0 && r < H) {
            if(col >= 0 && col + 3 < W) {
                my_a = vload4(0,frame);
            }
            else {
                int c=col;
                my_a.s0 = (c >= 0 && c < W) ? frame[0] : 0.0f;
                c++;
                my_a.s1 = (c >= 0 && c < W) ? frame[1] : 0.0f;
                c++;
                my_a.s2 = (c >= 0 && c < W) ? frame[2] : 0.0f;
                c++;
                my_a.s3 = (c >= 0 && c < W) ? frame[3] : 0.0f;
                c++;
            }
        }
        else {
            my_a = 0.0f;
        }
#ifndef shuffle_x4
        px[i] = my_a;
#endif
    }
    
    float4 p1,p2;
#ifdef shuffle_x4
    p1.s0 = shuffle_x4(my_a.s0,sel_a);
    p1.s1 = shuffle_x4(my_a.s1,sel_a);
    p1.s2 = shuffle_x4(my_a.s2,sel_a);
    p1.s3 = shuffle_x4(my_a.s3,sel_a);

    p2.s0 = shuffle_x4(my_a.s0,sel_b);
    p2.s1 = shuffle_x4(my_a.s1,sel_b);
    p2.s2 = shuffle_x4(my_a.s2,sel_b);
    p2.s3 = shuffle_x4(my_a.s3,sel_b);
#else
    p1 = px[0];
    p2 = px[1];
#endif

    const unsigned int neg_b = 1 + (0 << 1) + (1 << 2) + (1 << 3);
    float sign = (1 & (neg_b >> l_id)) ? -1.0 : 1.0;

    float4 bta = p1 + sign * p2;
    float4 btab;
    
    btab.s0 = bta.s0 - bta.s2;
    btab.s1 = bta.s1 + bta.s2;
    btab.s2 = bta.s2 - bta.s1;
    btab.s3 = bta.s1 - bta.s3;
   
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


__attribute__((reqd_work_group_size(4,8,8)))
__kernel void winconv_im2tile_4x4_x4(int BC,int H, int W,int pH,int pW,
                                    int h_tiles,int w_tiles,
                                    __global const float * restrict data,
                                    __global float4 *tiles)
{
    int bc  = get_global_id(0) / 4;
    int r = get_global_id(1);
    int c = get_global_id(2);
    int row = r * 2 - pH;
    int col = c * 2 - pW;

    if(bc >= BC || r >= h_tiles || c>= w_tiles)
        return;
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
#define TILES_IN_WG 16
#endif
#ifndef KERNELS_IN_WG
#define KERNELS_IN_WG 16
#endif

#define TILES_IN_BLOCK (TILES_IN_WG / WG_DIM_TILES)
#define KERNELS_IN_BLOCK (KERNELS_IN_WG / WG_DIM_KERNELS)

#define THREADS (WG_DIM_TILES * WG_DIM_KERNELS)


#define TILE_ITEMS_PER_THREAD (TILES_IN_WG * 16 / THREADS)
#define KERNEL_ITEMS_PER_THREAD (KERNELS_IN_WG * 16 / THREADS)


#define KERNEL_ITEMS_PER_WINMAT (16 / KERNEL_ITEMS_PER_THREAD)
#define TILE_ITEMS_PER_WINMAT (16 / TILE_ITEMS_PER_THREAD)

#if TILE_ITEMS_PER_THREAD == 16
#define vstoreT(x,y,z) vstore16(x,y,z)
#define vloadT(x,y) vload16(x,y)
#define floatT float16
#elif TILE_ITEMS_PER_THREAD == 8
#define vstoreT(x,y,z) vstore8(x,y,z)
#define vloadT(x,y) vload8(x,y)
#define floatT float8
#elif TILE_ITEMS_PER_THREAD == 4
#define vstoreT(x,y,z) vstore4(x,y,z)
#define vloadT(x,y) vload4(x,y)
#define floatT float4
#elif TILE_ITEMS_PER_THREAD == 2
#define vstoreT(x,y,z) vstore2(x,y,z)
#define vloadT(x,y) vload2(x,y)
#define floatT float2
#elif TILE_ITEMS_PER_THREAD == 1
#define vstoreT(x,y,z) z[(y)] = (x)
#define vloadT(x,y) y[x]
#define floatT float
#else
#error "Unsupported configuration"
#endif

#if KERNEL_ITEMS_PER_THREAD == 16
#define vstoreK(x,y,z) vstore16(x,y,z)
#define vloadK(x,y) vload16(x,y)
#define floatK float16
#elif KERNEL_ITEMS_PER_THREAD == 8
#define vstoreK(x,y,z) vstore8(x,y,z)
#define vloadK(x,y) vload8(x,y)
#define floatK float8
#elif KERNEL_ITEMS_PER_THREAD == 4
#define vstoreK(x,y,z) vstore4(x,y,z)
#define vloadK(x,y) vload4(x,y)
#define floatK float4
#elif KERNEL_ITEMS_PER_THREAD == 2
#define vstoreK(x,y,z) vstore2(x,y,z)
#define vloadK(x,y) vload2(x,y)
#define floatK float2
#elif KERNEL_ITEMS_PER_THREAD == 1
#define vstoreK(x,y,z) z[(y)] = (x)
#define vloadK(x,y) y[x]
#define floatK float
#else
#error "Unsupported configuration"
#endif


#define LOCAL_TILES_D1 (TILE_ITEMS_PER_THREAD + LOCAL_TMEM_PAD)
#define LOCAL_TILES_D2 (TILE_ITEMS_PER_WINMAT + LOCAL_TMEM_PAD0)
#define LOCAL_TILES_F2 (LOCAL_TILES_D1 * LOCAL_TILES_D2 + LOCAL_TMEM_PADX)
inline int swap_bits(int x)
{
    int bit_diff = ((x >> 3) ^ x) & 1;
    x ^= bit_diff << 3;
    x ^= bit_diff;
    return x;
}
inline float select(float4 x,int i)
{
    switch(i) {
    case 0: return x.s0;
    case 1: return x.s1;
    case 2: return x.s2;
    case 3: return x.s3;
    }
}

int4 get_selectm(int i)
{
    switch(i) {
    case 0: return (int4)(-1,0,0,0);
    case 1: return (int4)(0,-1,0,0);
    case 2: return (int4)(0,0,-1,0);
    case 3: return (int4)(0,0,0,-1);
    }

}
inline float selectm(float4 x,int4 mask)
{
    int4 tmp = as_int4(x) & mask;
    return as_float(tmp.s0 | tmp.s1 | tmp.s2 | tmp.s3);
}

#define ltiles(a,b,c) local_tiles[swap_bits(a) * LOCAL_TILES_F2 + (b) * LOCAL_TILES_D1 + (c)] 
//#define ltiles(a,b,c) local_tiles[(a) * LOCAL_TILES_F2 + (b) * LOCAL_TILES_D1 + (c)] 
#define LOCAL_TILES_SIZE (TILES_IN_WG * LOCAL_TILES_F2)

#if INTEL_PLATFORM == 1
#define permute(mp_per,line) intel_sub_group_shuffle(mp_per,(get_sub_group_local_id() & 0x1c) | line)
#define mad_permute(sum,mp1,mp_per,line) sum = mad(mp1,permute(mp_per,line),sum)
#elif AMD_PLATFORM == 1
#define permute(mp_per,line) as_float(__builtin_amdgcn_mov_dpp(as_int(mp_per),(line) | ((line) << 2) | ((line) << 4) | ((line) << 6) ,0xF,0xF,0))
#define mad_permute(sum,mp1,mp_per,line) \
    __asm__ volatile ("v_mac_f32_dpp %0,%1,%2 quad_perm:[" #line "," #line  "," #line ","#line "] row_mask:0xf bank_mask:0xf " \
                    : "+v" (sum) : "v" (mp_per) , "v" (mp1)   )
#elif NVIDIA_PLATFORM == 1
float permute(float v,int line)
{
        float temp_val;
        asm("shfl.idx.b32 %0,%1,%2,0x1C1F;" : "=r"(temp_val) : "r"(v),"r"(line) ); 
        return temp_val;
}
#define mad_permute(sum,mp1,mp_per,line) \
    do {        \
        float temp_val; \
        asm("shfl.idx.b32 %0,%1," #line ",0x1C1F;" : "=r"(temp_val) : "r"(mp_per) ); \
        sum=mad((temp_val),(mp1),sum); \
    } while(0) 
#endif


#define SUBSET_SIZE1D 4
#define SUBSET_SIZE (SUBSET_SIZE1D*SUBSET_SIZE1D)
#define SUBSETS (THREADS / SUBSET_SIZE)
#define SUBSETS_T (WG_DIM_TILES / SUBSET_SIZE1D)
#define SUBSETS_K (WG_DIM_KERNELS / SUBSET_SIZE1D)
#define TILES_IN_SUBSET   (TILES_IN_WG / SUBSETS_T)
#define KERNELS_IN_SUBSET (KERNELS_IN_WG / SUBSETS_K)

__kernel 
__attribute__((reqd_work_group_size(1,WG_DIM_TILES, WG_DIM_KERNELS)))
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
    __local float local_sum[TILES_IN_WG][KERNELS_IN_WG][SUBSET_SIZE];
    float ts[TILES_IN_SUBSET];
    float ks[KERNELS_IN_SUBSET];
    float s[TILES_IN_SUBSET][KERNELS_IN_SUBSET]={};

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

    int tile0 = get_local_id(1) / SUBSET_SIZE1D * TILES_IN_SUBSET;
    int kern0 = get_local_id(2) / SUBSET_SIZE1D * KERNELS_IN_SUBSET;

    int subset_local_id = get_local_id(1) % SUBSET_SIZE1D + (get_local_id(2) % SUBSET_SIZE1D) * SUBSET_SIZE1D;

    int tind = subset_local_id / TILE_ITEMS_PER_THREAD;
    int toff = subset_local_id % TILE_ITEMS_PER_THREAD;
    int kind = subset_local_id / KERNEL_ITEMS_PER_THREAD;
    int koff = subset_local_id % KERNEL_ITEMS_PER_THREAD;

    int tiles2d = tilesH * tilesW;
    __global float const * tile_base    =  tiles   + 16*(batch * tiles2d * C + tile_wg_id);
#if REV_KERNELS == 1
    __global float const * kernels_base = kernels + 16*kernel_wg_id;
    const int kern_step = oC * 16;
    const int kernels_stride = 16;
#else
    __global float const * kernels_base = kernels + 16*(C * kernel_wg_id);
    const int kern_step = 16;
    const int kernels_stride = 16 * C;
#endif

    local_tile_id = swap_bits(local_tile_id);
    __global float const *gtiles_ptr = tile_base + local_tile_id * 16 + local_tile_offset * TILE_ITEMS_PER_THREAD;
    __local  float *ltiles_ptr = &ltiles(local_tile_id,local_tile_offset,0);
    bool tile_load_flag =  (local_tile_id + tile_wg_id < tiles_no);

    __global float const *gkernels_ptr = kernels_base + local_kern_id * kernels_stride + local_kern_offset * KERNEL_ITEMS_PER_THREAD; 
    __local float *lkernels_ptr = local_kernels[local_kern_id][local_kern_offset];
    bool kern_load_flag = (local_kern_id + kernel_wg_id < oC);

    int my_offset = local_wg_id / 8 % 4; 
    int m0 = (my_offset + 0) % 4;
    int m1 = (my_offset + 1) % 4;
    int m2 = (my_offset + 2) % 4;
    int m3 = (my_offset + 3) % 4;

    int4 ms0 = get_selectm(m0);
    int4 ms1 = get_selectm(m1);
    int4 ms2 = get_selectm(m2);
    int4 ms3 = get_selectm(m3);

    __local float *ltiles_ptr0 = ltiles_ptr + m0;
    __local float *ltiles_ptr1 = ltiles_ptr + m1;
    __local float *ltiles_ptr2 = ltiles_ptr + m2;
    __local float *ltiles_ptr3 = ltiles_ptr + m3;

    for(int channel=0;channel<C;channel++) {
        
        {
                #if SIM==1
                volatile floatT tmpt = tile_load_flag ? 3.14f : 0.0f;
                volatile floatK tmpk = kern_load_flag ? 2.00f : 0.0f;
                #else
                
                #if 1
                floatT tmpt = tile_load_flag ? (float4)(gtiles_ptr[m0],gtiles_ptr[m1],gtiles_ptr[m2],gtiles_ptr[m3]) : 0.0f;
                #else
                floatT tmpt = tile_load_flag ? vloadT(0,gtiles_ptr) : 0.0f;
                #endif
                floatK tmpk = kern_load_flag ? vloadK(0,gkernels_ptr) : 0.0f;
                
                #endif  // SIM
                gtiles_ptr += tiles2d*16;
                gkernels_ptr += kern_step;

                #if 0
                vstoreT(tmpt,0,ltiles_ptr);
                #elif 1
                ltiles_ptr[m0] = tmpt.s0;
                ltiles_ptr[m1] = tmpt.s1;
                ltiles_ptr[m2] = tmpt.s2;
                ltiles_ptr[m3] = tmpt.s3;
                #else
                *ltiles_ptr0 = selectm(tmpt,ms0);
                *ltiles_ptr1 = selectm(tmpt,ms1);
                *ltiles_ptr2 = selectm(tmpt,ms2);
                *ltiles_ptr3 = selectm(tmpt,ms3);
                #endif
                vstoreK(tmpk,0,lkernels_ptr);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for(int dt = 0;dt < TILES_IN_SUBSET ; dt ++)
            ts[dt] = ltiles(tile0 + dt,tind,toff);
        #pragma unroll
        for(int dk = 0;dk < KERNELS_IN_SUBSET ; dk ++)
            ks[dk] = local_kernels[kern0 + dk][kind][koff];

        #pragma unroll
        for(int dt=0;dt<TILES_IN_SUBSET;dt++) {
            #pragma unroll
            for(int dk=0;dk<KERNELS_IN_SUBSET;dk++) {
                s[dt][dk] = mad(ts[dt],ks[dk],s[dt][dk]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    //return;
    
    
    //printf("k0=%d+%d t0=%d+%d sid=%d\n",kern0,KERNELS_IN_SUBSET,tile0,TILES_IN_SUBSET,subset_local_id);
    #pragma unroll
    for(int dt=0;dt < TILES_IN_SUBSET;dt++) {
        #pragma unroll
        for(int dk=0;dk < KERNELS_IN_SUBSET;dk++) {
            //if(kern0 + dk == 0 && tile0 + dt == 4)
            //    printf("%d:%f \n",subset_local_id,s[dt][dk]);
            //__local float * volatile ptr = &local_sum[tile0 + dt][kern0 + dk][subset_local_id];
            //*ptr = s[dt][dk];
            local_sum[tile0 + dt][kern0 + dk][subset_local_id] = s[dt][dk];
            //volatile float x=s[dt][dk];
        }
    }
    //return;
    barrier(CLK_LOCAL_MEM_FENCE);

    int oHW = oH*oW;
    int doff0 = oHW;
    int off0= batch*oC*oHW + oHW*kernel_id;
    #pragma unroll
    for(int dk=0;dk<KERNELS_IN_BLOCK;dk++,off0 += doff0) {
        int local_kern = kernel_local_base + dk;
        int kid = kernel_id + dk;
        if(kid >= oC)
            continue;
        int crow = tile_row;
        int ccol = tile_col;
        #pragma unroll
        for(int dt=0;dt<TILES_IN_BLOCK;dt++) {
            int local_tile = tile_local_base + dt;
            int tid = tile_row_col + dt;
            float16 tile16 = vload16(0,local_sum[local_tile][local_kern]);
            //printf("t=%d,k=%d %f %f %f %f\n",local_tile,local_kern,tile16.s0,tile16.s1,tile16.s2,tile16.s3);
            float4 tile_res = tile4x4_after_wingorad_to_2x2(tile16);
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

