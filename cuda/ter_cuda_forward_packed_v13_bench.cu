// ter_cuda_forward_packed_v13_bench.cu  --  v11 + v12 + v13-{lut,cpasync,int4tc}
// standalone benchmark over the five Llama-1B projection shapes.
//
// CSV out: kernel,shape,ms_median,gb_per_s,bw_eff_percent   (25 rows total)
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_86 -Xptxas -O3 \
//        ter_cuda_forward_packed_v13_bench.cu -o ter_cuda_forward_packed_v13_bench
// Run:
//   ./ter_cuda_forward_packed_v13_bench [iters=200] [warmup=20]

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>

#define CK(call) do { cudaError_t e=(call); if(e){ \
    std::fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
    std::exit(1);} } while(0)

// =============================================================================
//  v11 reference (inlined, identical to mm_packed_v4 in the paper baseline)
// =============================================================================
__global__ void mm_packed_v11_ref(
    const int8_t*  X, const uint8_t* W_col, int K, int N,
    const float* scale_x_dev, float w_scale, __half* out)
{
    int warp_block_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    int j_base = warp_block_id * 4;
    if (j_base >= N) return;
    int j_byte = j_base / 4;
    const uint8_t* Wc = W_col + (size_t)j_byte * K;
    __shared__ float scale_smem;
    if (threadIdx.x == 0) scale_smem = scale_x_dev[0] * w_scale;
    int kpt = (K + 31) / 32;
    int kstart = lane * kpt, kend = min(K, kstart + kpt);
    int acc[4] = {0,0,0,0};
    int kk = kstart;
    for (; kk + 4 <= kend; kk += 4) {
        uint32_t w = *reinterpret_cast<const uint32_t*>(Wc + kk);
        uint8_t b0=w&0xff, b1=(w>>8)&0xff, b2=(w>>16)&0xff, b3=(w>>24)&0xff;
        int32_t X4 = *reinterpret_cast<const uint32_t*>(X + kk);
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            int sb = sub*2;
            int t0=(b0>>sb)&3, t1=(b1>>sb)&3, t2=(b2>>sb)&3, t3=(b3>>sb)&3;
            int wv0=(t0==1)-(t0==2), wv1=(t1==1)-(t1==2);
            int wv2=(t2==1)-(t2==2), wv3=(t3==1)-(t3==2);
            int32_t W4 = (uint8_t)wv0 | ((uint8_t)wv1<<8) |
                         ((uint8_t)wv2<<16) | ((uint8_t)wv3<<24);
            acc[sub] = __dp4a(X4, W4, acc[sub]);
        }
    }
    for (; kk < kend; ++kk) {
        int xv = X[kk]; if (xv == 0) continue;
        uint8_t b = Wc[kk];
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            int t = (b >> (sub*2)) & 3;
            acc[sub] += xv * ((t==1)-(t==2));
        }
    }
    #pragma unroll
    for (int sub = 0; sub < 4; ++sub)
        for (int o = 16; o > 0; o >>= 1)
            acc[sub] += __shfl_xor_sync(0xffffffff, acc[sub], o);
    if (lane == 0) {
        float scale = scale_smem;
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub)
            if (j_base + sub < N)
                out[j_base + sub] = __float2half((float)acc[sub] * scale);
    }
}
static void launch_v11(const int8_t* X, const uint8_t* W, int K, int N,
                       const float* scale, float ws, __half* out)
{
    int t = 256, warps = t/32;
    int blocks = (N/4 + warps - 1) / warps;
    mm_packed_v11_ref<<<blocks, t>>>(X, W, K, N, scale, ws, out);
}

// =============================================================================
//  v12 inlined (LDG.128 + smem X stage + 8 cols/warp arithmetic unpack)
// =============================================================================
#ifndef WARPS_PER_BLOCK_V12
#define WARPS_PER_BLOCK_V12 8
#endif
__global__ void mm_packed_v12_inl(
    const int8_t*  __restrict__ X,
    const uint8_t* __restrict__ W_col,
    int K, int N,
    const float* __restrict__ scale_x_dev,
    float w_scale,
    __half* __restrict__ out)
{
    extern __shared__ int8_t sX[];
    const int tid    = threadIdx.x;
    const int bs     = blockDim.x;
    const int warpId = tid >> 5;
    const int lane   = tid & 31;
    const int j_base = (blockIdx.x * WARPS_PER_BLOCK_V12 + warpId) * 8;
    {
        const int4* Xv = reinterpret_cast<const int4*>(X);
        int4*       sv = reinterpret_cast<int4*>(sX);
        int n_int4 = K >> 4;
        for (int i = tid; i < n_int4; i += bs) sv[i] = __ldg(Xv + i);
    }
    __shared__ float scale_smem;
    if (tid == 0) scale_smem = scale_x_dev[0] * w_scale;
    __syncthreads();
    if (j_base >= N) return;
    int j_byte0 = j_base >> 2;
    int j_byte1 = j_byte0 + 1;
    const uint8_t* Wc0 = W_col + (size_t)j_byte0 * K;
    const uint8_t* Wc1 = W_col + (size_t)j_byte1 * K;
    int acc[8] = {0,0,0,0,0,0,0,0};
    constexpr int LANE_STRIDE_BYTES = 32 * 16;
    int kk = lane * 16;
    for (; kk + 16 <= K; kk += LANE_STRIDE_BYTES) {
        int4 w0 = __ldg(reinterpret_cast<const int4*>(Wc0 + kk));
        int4 w1 = __ldg(reinterpret_cast<const int4*>(Wc1 + kk));
        int4 x4 = *reinterpret_cast<const int4*>(sX + kk);
        const uint32_t* w0u = reinterpret_cast<const uint32_t*>(&w0);
        const uint32_t* w1u = reinterpret_cast<const uint32_t*>(&w1);
        const int32_t*  xu  = reinterpret_cast<const int32_t*>(&x4);
        #pragma unroll
        for (int chunk = 0; chunk < 4; ++chunk) {
            uint32_t W0p = w0u[chunk], W1p = w1u[chunk];
            int32_t  X4  = xu[chunk];
            uint8_t a0=W0p&0xff,a1=(W0p>>8)&0xff,a2=(W0p>>16)&0xff,a3=(W0p>>24)&0xff;
            uint8_t b0=W1p&0xff,b1=(W1p>>8)&0xff,b2=(W1p>>16)&0xff,b3=(W1p>>24)&0xff;
            #pragma unroll
            for (int sub = 0; sub < 4; ++sub) {
                int sh = sub*2;
                int t0=(a0>>sh)&3,t1=(a1>>sh)&3,t2=(a2>>sh)&3,t3=(a3>>sh)&3;
                int v0=(t0==1)-(t0==2),v1=(t1==1)-(t1==2);
                int v2=(t2==1)-(t2==2),v3=(t3==1)-(t3==2);
                int32_t W4 = (uint8_t)v0|((uint8_t)v1<<8)|((uint8_t)v2<<16)|((uint8_t)v3<<24);
                acc[sub] = __dp4a(X4, W4, acc[sub]);
                int u0=(b0>>sh)&3,u1=(b1>>sh)&3,u2=(b2>>sh)&3,u3=(b3>>sh)&3;
                int p0=(u0==1)-(u0==2),p1=(u1==1)-(u1==2);
                int p2=(u2==1)-(u2==2),p3=(u3==1)-(u3==2);
                int32_t W4b = (uint8_t)p0|((uint8_t)p1<<8)|((uint8_t)p2<<16)|((uint8_t)p3<<24);
                acc[sub+4] = __dp4a(X4, W4b, acc[sub+4]);
            }
        }
    }
    #pragma unroll
    for (int s = 0; s < 8; ++s)
        for (int o = 16; o > 0; o >>= 1)
            acc[s] += __shfl_xor_sync(0xffffffff, acc[s], o);
    if (lane == 0) {
        float scale = scale_smem;
        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            int col = j_base + s;
            if (col < N) out[col] = __float2half((float)acc[s] * scale);
        }
    }
}
static void launch_v12(const int8_t* X, const uint8_t* W, int K, int N,
                       const float* scale, float ws, __half* out)
{
    assert((K & 15) == 0); assert((N & 7) == 0);
    int t = 32 * WARPS_PER_BLOCK_V12;
    int cpb = WARPS_PER_BLOCK_V12 * 8;
    int blocks = (N + cpb - 1) / cpb;
    mm_packed_v12_inl<<<blocks, t, K>>>(X, W, K, N, scale, ws, out);
}

// =============================================================================
//  v13_lut inlined  (constant-memory LUT unpack)
// =============================================================================
__constant__ int32_t kUnpackLUT_bench[256];
static void init_unpack_lut_bench() {
    int32_t h[256];
    for (int b = 0; b < 256; ++b) {
        int8_t v[4];
        for (int s = 0; s < 4; ++s) {
            int t = (b >> (s*2)) & 3;
            v[s] = (int8_t)((t == 1) - (t == 2));
        }
        h[b] = ((uint8_t)v[0]) | ((uint8_t)v[1]<<8) |
               ((uint8_t)v[2]<<16) | ((uint8_t)v[3]<<24);
    }
    CK(cudaMemcpyToSymbol(kUnpackLUT_bench, h, sizeof(h)));
}
__global__ void mm_packed_v13_lut_inl(
    const int8_t*  __restrict__ X,
    const uint8_t* __restrict__ W_col,
    int K, int N,
    const float* __restrict__ scale_x_dev,
    float w_scale,
    __half* __restrict__ out)
{
    int warp_block_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    int j_base = warp_block_id * 4;
    if (j_base >= N) return;
    int j_byte = j_base / 4;
    const uint8_t* Wc = W_col + (size_t)j_byte * K;
    __shared__ float scale_smem;
    if (threadIdx.x == 0) scale_smem = scale_x_dev[0] * w_scale;
    __syncthreads();
    int kpt = (K + 31) / 32;
    int kstart = lane * kpt, kend = min(K, kstart + kpt);
    int acc[4] = {0,0,0,0};
    int kk = kstart;
    for (; kk + 4 <= kend; kk += 4) {
        uint32_t w  = *reinterpret_cast<const uint32_t*>(Wc + kk);
        int32_t  X4 = *reinterpret_cast<const int32_t*>(X + kk);
        int32_t l0 = kUnpackLUT_bench[(w      ) & 0xff];
        int32_t l1 = kUnpackLUT_bench[(w >>  8) & 0xff];
        int32_t l2 = kUnpackLUT_bench[(w >> 16) & 0xff];
        int32_t l3 = kUnpackLUT_bench[(w >> 24) & 0xff];
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            int sh = sub * 8;
            uint8_t v0 = (l0 >> sh) & 0xff;
            uint8_t v1 = (l1 >> sh) & 0xff;
            uint8_t v2 = (l2 >> sh) & 0xff;
            uint8_t v3 = (l3 >> sh) & 0xff;
            int32_t W4 = v0 | (v1<<8) | (v2<<16) | (v3<<24);
            acc[sub] = __dp4a(X4, W4, acc[sub]);
        }
    }
    for (; kk < kend; ++kk) {
        int xv = X[kk]; if (xv == 0) continue;
        uint8_t b = Wc[kk];
        int32_t lut = kUnpackLUT_bench[b];
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            int8_t v = (int8_t)((lut >> (sub*8)) & 0xff);
            acc[sub] += xv * (int)v;
        }
    }
    #pragma unroll
    for (int sub = 0; sub < 4; ++sub)
        for (int o = 16; o > 0; o >>= 1)
            acc[sub] += __shfl_xor_sync(0xffffffff, acc[sub], o);
    if (lane == 0) {
        float scale = scale_smem;
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            int col = j_base + sub;
            if (col < N) out[col] = __float2half((float)acc[sub] * scale);
        }
    }
}
static void launch_v13_lut(const int8_t* X, const uint8_t* W, int K, int N,
                           const float* scale, float ws, __half* out)
{
    int t = 256, warps = t/32;
    int blocks = (N/4 + warps - 1) / warps;
    mm_packed_v13_lut_inl<<<blocks, t>>>(X, W, K, N, scale, ws, out);
}

// =============================================================================
//  v13_cpasync inlined  (cp.async X-stage; same v11 compute)
// =============================================================================
__device__ __forceinline__ void cp_async_16_b(void* sp, const void* gp) {
    unsigned s = __cvta_generic_to_shared(sp);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gp));
}
__device__ __forceinline__ void cp_async_commit_b() {
    asm volatile("cp.async.commit_group;\n" ::);
}
__device__ __forceinline__ void cp_async_wait_all_b() {
    asm volatile("cp.async.wait_all;\n" ::);
}
__global__ void mm_packed_v13_cpasync_inl(
    const int8_t*  __restrict__ X,
    const uint8_t* __restrict__ W_col,
    int K, int N,
    const float* __restrict__ scale_x_dev,
    float w_scale,
    __half* __restrict__ out)
{
    extern __shared__ int8_t sX[];
    const int tid = threadIdx.x;
    const int bs  = blockDim.x;
    int n_int16 = K >> 4;
    for (int i = tid; i < n_int16; i += bs)
        cp_async_16_b(sX + i*16, X + i*16);
    cp_async_commit_b();
    cp_async_wait_all_b();
    __shared__ float scale_smem;
    if (tid == 0) scale_smem = scale_x_dev[0] * w_scale;
    __syncthreads();

    int warp_block_id = (blockIdx.x * blockDim.x + tid) / 32;
    int lane = tid & 31;
    int j_base = warp_block_id * 4;
    if (j_base >= N) return;
    int j_byte = j_base / 4;
    const uint8_t* Wc = W_col + (size_t)j_byte * K;

    int kpt = (K + 31) / 32;
    int kstart = lane * kpt, kend = min(K, kstart + kpt);
    int acc[4] = {0,0,0,0};
    int kk = kstart;
    for (; kk + 4 <= kend; kk += 4) {
        uint32_t w  = *reinterpret_cast<const uint32_t*>(Wc + kk);
        int32_t  X4 = *reinterpret_cast<const int32_t*>(sX + kk);
        uint8_t b0=w&0xff,b1=(w>>8)&0xff,b2=(w>>16)&0xff,b3=(w>>24)&0xff;
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            int sb = sub*2;
            int t0=(b0>>sb)&3,t1=(b1>>sb)&3,t2=(b2>>sb)&3,t3=(b3>>sb)&3;
            int wv0=(t0==1)-(t0==2),wv1=(t1==1)-(t1==2);
            int wv2=(t2==1)-(t2==2),wv3=(t3==1)-(t3==2);
            int32_t W4 = (uint8_t)wv0|((uint8_t)wv1<<8)|((uint8_t)wv2<<16)|((uint8_t)wv3<<24);
            acc[sub] = __dp4a(X4, W4, acc[sub]);
        }
    }
    for (; kk < kend; ++kk) {
        int xv = sX[kk]; if (xv == 0) continue;
        uint8_t b = Wc[kk];
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            int t = (b >> (sub*2)) & 3;
            acc[sub] += xv * ((t==1)-(t==2));
        }
    }
    #pragma unroll
    for (int sub = 0; sub < 4; ++sub)
        for (int o = 16; o > 0; o >>= 1)
            acc[sub] += __shfl_xor_sync(0xffffffff, acc[sub], o);
    if (lane == 0) {
        float scale = scale_smem;
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            int col = j_base + sub;
            if (col < N) out[col] = __float2half((float)acc[sub] * scale);
        }
    }
}
static void launch_v13_cpasync(const int8_t* X, const uint8_t* W, int K, int N,
                               const float* scale, float ws, __half* out)
{
    int t = 256, warps = t/32;
    int blocks = (N/4 + warps - 1) / warps;
    mm_packed_v13_cpasync_inl<<<blocks, t, K>>>(X, W, K, N, scale, ws, out);
}

// =============================================================================
//  v13_int4tc  (SCAFFOLD — no-op kernel; documented in its own .cu)
// =============================================================================
__global__ void mm_packed_v13_int4tc_inl(
    const int8_t* /*X*/, const uint8_t* /*W*/, int /*K*/, int N,
    const float* /*scale*/, float /*ws*/, __half* out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out[idx] = __float2half(0.0f);
}
static void launch_v13_int4tc(const int8_t* X, const uint8_t* W, int K, int N,
                              const float* scale, float ws, __half* out)
{
    int t = 256;
    int blocks = (N + t - 1) / t;
    mm_packed_v13_int4tc_inl<<<blocks, t>>>(X, W, K, N, scale, ws, out);
}

// =============================================================================
//  Bench harness
// =============================================================================
struct Shape { const char* name; int K, N; };

static void bench_one(const char* tag, int K, int N,
                      void (*launch)(const int8_t*, const uint8_t*, int, int,
                                     const float*, float, __half*),
                      int iters, int warmup,
                      double peak_gbs,
                      const int8_t* dX, const uint8_t* dW, __half* dY,
                      const float* dScale)
{
    for (int i = 0; i < warmup; ++i) launch(dX, dW, K, N, dScale, 0.05f, dY);
    CK(cudaDeviceSynchronize());
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    std::vector<double> times; times.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(s);
        launch(dX, dW, K, N, dScale, 0.05f, dY);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms = 0; cudaEventElapsedTime(&ms, s, e);
        times.push_back(ms);
    }
    std::sort(times.begin(), times.end());
    double med = times[times.size()/2];
    double bytes = (double)K * (N/4);
    double gbs   = bytes / (med * 1e-3) / 1e9;
    double effp  = gbs / peak_gbs * 100.0;
    std::printf("%s,K=%d_N=%d,%.4f,%.1f,%.2f\n", tag, K, N, med, gbs, effp);
    cudaEventDestroy(s); cudaEventDestroy(e);
}

int main(int argc, char** argv) {
    int iters  = (argc > 1) ? std::atoi(argv[1]) : 200;
    int warmup = (argc > 2) ? std::atoi(argv[2]) : 20;

    init_unpack_lut_bench();

    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    int mem_clk_khz = 0, mem_bus_bits = 0;
    cudaError_t e1 = cudaDeviceGetAttribute(&mem_clk_khz,  cudaDevAttrMemoryClockRate, 0);
    cudaError_t e2 = cudaDeviceGetAttribute(&mem_bus_bits, cudaDevAttrGlobalMemoryBusWidth, 0);
    double peak_gbs = (e1 == cudaSuccess && e2 == cudaSuccess && mem_clk_khz > 0 && mem_bus_bits > 0)
        ? 2.0 * static_cast<double>(mem_clk_khz) * 1e3 * (static_cast<double>(mem_bus_bits) / 8.0) / 1e9
        : 936.0;
    std::fprintf(stderr, "Device: %s   peak HBM ~%.0f GB/s\n", p.name, peak_gbs);
    std::printf("kernel,shape,ms_median,gb_per_s,bw_eff_percent\n");

    Shape shapes[] = {
        {"Wq_Wo",   2048,   2048},
        {"Wk_Wv",   2048,    512},
        {"Wgate_up",2048,   8192},
        {"Wdown",   8192,   2048},
        {"lm_head", 2048, 128256},
    };

    // Find the maximum K and N to allocate big-enough scratch buffers once.
    int Kmax = 0, Nmax = 0;
    for (auto& s : shapes) {
        if (s.K > Kmax) Kmax = s.K;
        if (s.N > Nmax) Nmax = s.N;
    }
    // We allocate per-shape inside the loop (simpler; matches v12 bench).
    std::mt19937 rng(0xdeadbeef);
    std::uniform_int_distribution<int> dx(-127, 127);
    std::uniform_int_distribution<int> dw(0, 255);

    using LaunchFn = void(*)(const int8_t*, const uint8_t*, int, int,
                             const float*, float, __half*);
    struct Variant { const char* tag; LaunchFn fn; };
    Variant variants[] = {
        {"v11",          launch_v11},
        {"v12",          launch_v12},
        {"v13_lut",      launch_v13_lut},
        {"v13_cpasync",  launch_v13_cpasync},
        {"v13_int4tc",   launch_v13_int4tc},
    };

    for (auto& sh : shapes) {
        int K = sh.K, N = sh.N;
        int8_t*  dX = nullptr; uint8_t* dW = nullptr;
        __half*  dY = nullptr; float*   dS = nullptr;
        CK(cudaMalloc(&dX, K));
        CK(cudaMalloc(&dW, (size_t)K * (N/4)));
        CK(cudaMalloc(&dY, N * sizeof(__half)));
        CK(cudaMalloc(&dS, sizeof(float)));
        std::vector<int8_t>  hX(K);
        std::vector<uint8_t> hW((size_t)K * (N/4));
        for (auto& v : hX) v = (int8_t)dx(rng);
        for (auto& v : hW) v = (uint8_t)dw(rng);
        CK(cudaMemcpy(dX, hX.data(), K, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dW, hW.data(), hW.size(), cudaMemcpyHostToDevice));
        float scale = 1.0f / 127.f;
        CK(cudaMemcpy(dS, &scale, sizeof(float), cudaMemcpyHostToDevice));
        for (auto& v : variants) {
            bench_one(v.tag, K, N, v.fn, iters, warmup, peak_gbs,
                      dX, dW, dY, dS);
        }
        cudaFree(dX); cudaFree(dW); cudaFree(dY); cudaFree(dS);
    }
    return 0;
}
