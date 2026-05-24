// ter_cuda_hybrid_dispatch_v13.cu
//
// Hybrid dispatch wrapper combining:
//   * v11 (dp4a packed-trit)        -> wins for small N and K-heavy shapes
//   * v13 INT4 Tensor Cores         -> wins for large N (gate/up, lm_head)
//
// Empirical crossover (RTX 3090, 2026-05-17):
//   shape                    v11_ms    v13_int4tc_ms   winner
//   K=2048_N=2048   (Wq/Wo)  0.0092    0.0113          v11
//   K=2048_N=512    (Wk/Wv)  0.0092    0.0113          v11
//   K=2048_N=8192   (Wgate)  0.0213    0.0184          v13
//   K=8192_N=2048   (Wdown)  0.0244    0.0305          v11
//   K=2048_N=128256 (lmh)    0.1843    0.1587          v13
//
// Heuristic: dispatch v13 INT4 TC when N >= 4096, else v11.
// (Wdown has K=8192 N=2048 -> N<4096 -> v11. Correct decision.)
//
// This file is *header-only style* — include it once in a bench/forward TU.
// It expects ter_cuda_forward_packed_v13_int4tc.cu to be already included
// with V13_INT4TC_NO_MAIN defined, so the v13 kernel + prepack helpers
// (mm_packed_v13_int4tc, launch_mm_packed_v13_int4tc,
// prepack_w_int4_v13_device, repack_trits_to_int4_host) are visible.

#ifndef TER_CUDA_HYBRID_DISPATCH_V13_CU
#define TER_CUDA_HYBRID_DISPATCH_V13_CU

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef HCK
#define HCK(call) do { cudaError_t e=(call); if(e){ \
    std::fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
    std::exit(1);} } while(0)
#endif

// ---------------------------------------------------------------------------
// Inlined v11 (dp4a packed-trit) kernel + launcher, lifted from
// ter_cuda_forward_packed_v12.cu (mm_packed_v11_ref). Renamed to
// mm_packed_v11_hyb / launch_v11_hyb to avoid symbol collisions if the caller
// also pulls in v12.
// ---------------------------------------------------------------------------
__global__ void mm_packed_v11_hyb(
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

static inline void launch_v11_hyb(const int8_t* X, const uint8_t* W,
                                  int K, int N, const float* scale, float ws,
                                  __half* out, cudaStream_t s = 0)
{
    int t = 256, warps = t/32;
    int blocks = (N/4 + warps - 1) / warps;
    mm_packed_v11_hyb<<<blocks, t, 0, s>>>(X, W, K, N, scale, ws, out);
}

// ---------------------------------------------------------------------------
// Dispatch decision.
// ---------------------------------------------------------------------------
enum class HybridKernel { V11_DP4A, V13_INT4TC, V13_INT4TC_P };

// Global flag selecting the int4 variant for the hybrid dispatch:
//   0 = lossy v13_int4tc (default, original)
//   1 = precise v13_int4tc_p (bit-decomposition, full int8)
static int g_hybrid_int4_precise = 0;
static inline void set_hybrid_int4_precise(int v) { g_hybrid_int4_precise = v; }

static inline HybridKernel pick_kernel(int /*K*/, int N) {
    if (N >= 4096) {
        return g_hybrid_int4_precise ? HybridKernel::V13_INT4TC_P
                                     : HybridKernel::V13_INT4TC;
    }
    return HybridKernel::V11_DP4A;
}

// ---------------------------------------------------------------------------
// Pre-packed weight bundle. We ALWAYS allocate both layouts so the runtime
// dispatch is a plain branch with no fallback work. Extra memory cost is
// modest: INT4 layout is K*N/2 bytes ≈ 2x the packed-trit footprint.
// ---------------------------------------------------------------------------
struct HybridWeights {
    uint8_t*  W_packed;   // v11 dp4a: col-major 4 trits/byte, [N/4 * K] bytes
    uint32_t* W_int4;     // v13 int4tc: per-lane preformatted, [N/8 * K/32 * 32] u32
    int32_t*  W_sum;      // v13 int4tc_p: per-N column weight sum, [N] int32
    int K, N;
};

// Build both layouts on the device from a host-side packed-trit buffer.
//   W_packed_host:  [N/4 * K] bytes (col-major 4 trits/byte)
//   K, N           : dims (K%32==0, N%8==0 required for v13 path)
static inline HybridWeights prepack_hybrid_weights(
    const uint8_t* W_packed_host, int K, int N)
{
    HybridWeights H{};
    H.K = K; H.N = N;
    size_t pbytes = (size_t)K * (N / 4);

    HCK(cudaMalloc(&H.W_packed, pbytes));
    HCK(cudaMemcpy(H.W_packed, W_packed_host, pbytes, cudaMemcpyHostToDevice));

    if (K % 32 == 0 && N % 8 == 0) {
        size_t out_words = (size_t)(N/8) * (K/32) * 32;
        std::vector<uint32_t> hWi(out_words);
        repack_trits_to_int4_host(W_packed_host, K, N, hWi.data());
        HCK(cudaMalloc(&H.W_int4, out_words * sizeof(uint32_t)));
        HCK(cudaMemcpy(H.W_int4, hWi.data(),
                       out_words * sizeof(uint32_t), cudaMemcpyHostToDevice));
        // Precompute W_sum[n] = sum_k W[k,n] on host (cheap, one-shot).
        std::vector<int32_t> hWsum(N, 0);
        for (int n = 0; n < N; ++n) {
            int j_byte = n / 4;
            int sub    = n % 4;
            int32_t acc = 0;
            for (int k = 0; k < K; ++k) {
                int t = (W_packed_host[(size_t)j_byte * K + k] >> (sub * 2)) & 3;
                acc += (t == 1) - (t == 2);
            }
            hWsum[n] = acc;
        }
        HCK(cudaMalloc(&H.W_sum, (size_t)N * sizeof(int32_t)));
        HCK(cudaMemcpy(H.W_sum, hWsum.data(),
                       (size_t)N * sizeof(int32_t), cudaMemcpyHostToDevice));
    } else {
        H.W_int4 = nullptr;
        H.W_sum  = nullptr;
    }
    return H;
}

static inline void free_hybrid_weights(HybridWeights& H) {
    if (H.W_packed) { cudaFree(H.W_packed); H.W_packed = nullptr; }
    if (H.W_int4)   { cudaFree(H.W_int4);   H.W_int4   = nullptr; }
    if (H.W_sum)    { cudaFree(H.W_sum);    H.W_sum    = nullptr; }
    H.K = H.N = 0;
}

// ---------------------------------------------------------------------------
// Unified launcher.
// ---------------------------------------------------------------------------
static inline void launch_hybrid_matmul(
    const int8_t* X, const HybridWeights& W,
    const float* scale, float ws, __half* out, cudaStream_t s = 0)
{
    HybridKernel k = pick_kernel(W.K, W.N);
    if (k == HybridKernel::V13_INT4TC_P && W.W_int4 != nullptr && W.W_sum != nullptr) {
        launch_mm_packed_v13_int4tc_p(
            X, reinterpret_cast<const uint8_t*>(W.W_int4), W.W_sum,
            W.K, W.N, scale, ws, out, s);
    } else if (k == HybridKernel::V13_INT4TC && W.W_int4 != nullptr) {
        launch_mm_packed_v13_int4tc(
            X, reinterpret_cast<const uint8_t*>(W.W_int4),
            W.K, W.N, scale, ws, out, s);
    } else {
        launch_v11_hyb(X, W.W_packed, W.K, W.N, scale, ws, out, s);
    }
}

#endif  // TER_CUDA_HYBRID_DISPATCH_V13_CU
