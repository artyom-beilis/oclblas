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


__kernel void winconv_calc_gkgt_3x3(int N,__global const float * restrict gk3,__global float16 *k4)
{
    int kid = get_global_id(0);
    if(kid >= N)
        return;
    k4[kid] = load_3x3_kernel_and_transform(gk3 + kid*9);
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

#ifndef TILES_IN_WG
#define TILES_IN_WG 8
#endif
#ifndef KERNELS_IN_WG
#define KERNELS_IN_WG 8
#endif


__kernel 
__attribute__((reqd_work_group_size(1,TILES_IN_WG, KERNELS_IN_WG)))
void winconv_3x3(int B, int C,int oC, int oH, int oW,int tilesH,int tilesW,
                          __global float const *tiles,
                          __global float const *kernels,
                          __global float *result)
{
    // +1 to prevent bank collisions
    __local float local_tiles[TILES_IN_WG][16+1]; 
    __local float local_kernels[KERNELS_IN_WG][16+1];
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

    for(int channel=0;channel<C;channel++,tile_base += tiles2d*16,kernels_base +=16) {
        const int kstep = 16 / KERNELS_IN_WG;
        const int tstep = 16 / TILES_IN_WG;
        if(tile_row_col < tiles2d) {
            #pragma unroll
            for(int i=0;i<kstep;i++) {
                local_tiles[tile_wg_id][kernel_wg_id*kstep +i]  = tile_base[kernel_wg_id*kstep + i];
            }
        }
        else {
            #pragma unroll
            for(int i=0;i<kstep;i++) {
                local_tiles[tile_wg_id][kernel_wg_id*kstep +i]  = 0.0f;
            }
        }
        if(kernel_id < oC) {
            #pragma unroll
            for(int i=0;i<tstep;i++) {
                local_kernels[kernel_wg_id][tile_wg_id*tstep+i] = kernels_base[tile_wg_id*tstep+i];
            }
        }
        else {
            #pragma unroll
            for(int i=0;i<tstep;i++)
                local_kernels[kernel_wg_id][tile_wg_id*tstep+i] = 0.0f;
        }


        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for(int i=0;i<16;i++) {
            s[i] = mad(local_tiles[tile_wg_id][i],local_kernels[kernel_wg_id][i],s[i]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
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
