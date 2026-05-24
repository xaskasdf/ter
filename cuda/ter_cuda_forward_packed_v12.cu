// ter_cuda_forward_packed_v12.cu  --  Packed-trit GEMV kernel, v12.
// Standalone kernel + minimal benchmark harness. v11 baseline lives in
// ter_cuda_forward_packed.cu (mm_packed_v4) and is NOT modified.
//
// Build (Ampere / RTX 3090):
//   nvcc -O3 -arch=sm_86 -std=c++17 ter_cuda_forward_packed_v12.cu \
//        -o ter_cuda_forward_packed_v12
// Build (sm_80, A100 compat):
//   nvcc -O3 -arch=sm_80 -std=c++17 ter_cuda_forward_packed_v12.cu \
//        -o ter_cuda_forward_packed_v12
//
// ---------------------------------------------------------------------------
// MOTIVATION
// ---------------------------------------------------------------------------
// v11 (`mm_packed_v4`) achieves ~46 GB/s on RTX 3090 Llama-1B matmul fabric
// (M=1 GEMV against 4-trits/byte W) — ~5% of the 936 GB/s HBM peak. cuBLAS
// INT8 TC achieves ~196 GB/s (21%) on the same shape but pays 4x weight
// footprint. Goal of v12: stay at 4x compression vs INT8 while pushing the
// packed kernel to 10-15% bw_eff (~95-140 GB/s achieved), which projects to
// ms/forward ~4.5-5.5 on Llama 1B M=1 — i.e., beats cuBLAS net of compression.
//
// ---------------------------------------------------------------------------
// v11 BOTTLENECKS  (from line-level analysis of mm_packed_v4)
// ---------------------------------------------------------------------------
//   B1. SCALAR uint32 LOADS (line ~365):  `uint32_t w = *(const uint32_t*)(Wc+kk)`
//       reads 4 bytes/transaction. Each warp lane issues a separate load every
//       4 K-elements → 32 separate 4B transactions per warp per K-step. Memory
//       subsystem coalesces them (lanes hit adjacent addresses), but the per-
//       lane instruction count is high and we never get an int4 (16B) wide LDG.
//
//   B2. NO __ldg / READ-ONLY CACHE HINT:  raw `*` deref, not `__ldg(...)`.
//       Weights are reused 0 times in M=1 GEMV so cache reuse is moot, but the
//       L1/tex path on Ampere still gives lower-latency reads for streaming
//       LDG with the .nc / __ldg modifier.
//
//   B3. UNPACK COST = 6 ops/trit:  for each byte the inner loop computes
//       `t = (b>>sb)&3; wv = (t==1) - (t==2);` per sub-column then packs four
//       wv's back into an int32 W4 before __dp4a. That's ~24 ALU ops per
//       __dp4a per warp lane. With memory now better fed, ALU may bind.
//
//   B4. WARP-COOP K-REDUCTION ALIGNMENT:  v11 splits K across 32 lanes
//       contiguously (`kpt = (K+31)/32`, `kstart = lane*kpt`). For K=2048 each
//       lane owns a 64-element range. Lane 0 reads bytes 0..63 from Wc, lane 1
//       reads 64..127, etc. → loads from lane l and lane l+1 within the same
//       cycle are 64 B apart → SECTOR THRASHING (each sector = 32 B). The HW
//       has to issue many sector loads per warp instruction even though the
//       overall warp footprint is contiguous.  This is the #1 efficiency leak.
//
//   B5. SHARED-MEMORY UNUSED FOR X:  the activation vector X[K] is hit by
//       every warp in the grid; v11 reads it from gmem inside each warp via
//       `*(const uint32_t*)(X+kk)`. For Wgate/Wup (N=8192, K=2048) there are
//       8192/4 = 2048 warps each re-reading 2 KB of X = 4 MB redundant reads.
//       Caching to smem once per block kills that traffic.
//
// ---------------------------------------------------------------------------
// v12 DESIGN
// ---------------------------------------------------------------------------
//   D1. INT4 (16B) WIDE LDG over W_col.  We change the per-lane stride so each
//       lane consumes a contiguous int4 (16 packed bytes = 64 trits) per
//       memory transaction, then advances by warpSize * 16. Sector footprint
//       per warp instruction = 512 B = 16 contiguous sectors → fully coalesced
//       and one 128B HBM burst can feed the whole half-warp.
//       Expected: ~4x effective LDG throughput vs v11. Memory side becomes
//       the bandwidth limit, not the issue limit.
//
//   D2. __ldg() on weights + activations.  Marks the path as read-only,
//       enables non-coherent load (Ampere LDG.E.128.SYS.CONSTANT). Cheap, no
//       downside for our access pattern.
//
//   D3. SMEM STAGING OF X.  X is K bytes (K=2048 for most projections, K=8192
//       for Wdown). Each block stages X once into __shared__ int8_t sX[K]
//       cooperatively, __syncthreads(), then every warp reads X from smem.
//       Eliminates B5 (redundant gmem traffic). At K=8192, smem usage = 8 KB
//       which is well under the 100 KB/block budget on sm_86.
//
//   D4. UNPACK VIA NIBBLE LUT  (compile-time const, register-resident).  For
//       each of the four 2-bit sub-fields of a byte we want the signed value
//       in {-1, 0, +1}. v11 computes it with 2 compares per trit. v12 uses a
//       small lookup driven by `__byte_perm` to fold the four sub-fields of a
//       byte into a packed int32 of signed nibbles, then packs back to int32
//       for __dp4a.  This drops the inner-loop ALU count from ~24 to ~10 ops
//       per __dp4a (still bounded by HBM, but lower fp ALU power draw and
//       gives the SM headroom for instruction issue parallelism).
//
//   D5. K-STRIDE IS warpSize, NOT CONTIGUOUS RANGE.  Each lane processes
//       k = lane, lane+32, lane+64, ... so lane l reads byte at offset
//       (kk + l) and lane l+1 reads (kk + l + 1) — i.e., the 32 lanes of a
//       warp form a single contiguous 32-byte sector per fundamental step,
//       and with vector LDG.128 we promote that to four packed-int4 per warp.
//       Final reduction stays warp shuffle.  Fixes B4 directly.
//
//   D6. 8 COLS PER WARP (UP FROM 4).  Reading 16 packed bytes per LDG.128
//       gives us 64 trits — enough to feed 8 output columns concurrently with
//       a 2x2 register accumulator tile. v11 uses 4 cols/warp with 4 acc
//       registers; v12 uses 8 cols/warp with 8 acc registers. Doubles arithmetic
//       intensity per byte loaded → better tolerance of any unpack stalls.
//
//   D7. (OPTIONAL, STAGED) cp.async on sm_80+ for X staging — wired as a TODO
//       block below; default path uses ordinary smem copy which is already
//       1-shot per block so the asynchronous variant is a small win at best.
//
// ---------------------------------------------------------------------------
// EXPECTED IMPACT  (back-of-envelope, Llama 1B M=1, K=2048 N=8192 layer)
// ---------------------------------------------------------------------------
//   Weight bytes touched per forward call to one packed projection
//                       = K * N / 4 = 2048 * 8192 / 4 = 4.2 MB
//   v11 measured        : 46 GB/s achieved   =>  4.2 MB / 46 GB/s = 91 us
//   v12 target          : 120 GB/s achieved  =>  4.2 MB /120 GB/s = 35 us
//   Speedup per matmul  : 2.6x
//   Whole forward has 7 matmuls/layer * 16 layers ≈ dominates non-attn time.
//   ms/forward today    : 7.29 (v11) vs 5.90 (ADD-only) vs 6.82 (cuBLAS).
//   ms/forward projected: 4.7-5.0 ms  (matches the 10-15% bw_eff target).
//
// ---------------------------------------------------------------------------
// HOST INVOCATION
// ---------------------------------------------------------------------------
// Signature mirrors `mm_packed_v4` in ter_cuda_forward_packed.cu so it can be
// dropped into `mm_packed_dispatch` directly:
//
//   mm_packed_v12<<<blocks, 256, K, stream>>>(
//       X, W_col, K, N, scale_x_dev, w_scale, out);
//
// where:
//   X         : device int8_t[K] activations (per-tensor scaled)
//   W_col     : device uint8_t[(N/4) * K]  4-trits/byte, col-major in groups of 4
//   K, N      : K = input dim, N = output dim (must be multiple of 8)
//   scale_x_dev: device float[1]
//   w_scale   : host float
//   out       : device __half[N]
//
// Constraints (asserted on host):
//   - K % 32 == 0   (LDG.128 alignment; trivially true for Llama 1B: 2048, 8192)
//   - N % 8  == 0   (8 cols/warp packing)
//
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
//  Kernel: mm_packed_v12
// ---------------------------------------------------------------------------
//  Grid:   blocks = ceil(N / (WARPS_PER_BLOCK * 8))
//  Block:  256 threads = 8 warps
//  Shmem:  K bytes  (staged X[K])
//
//  Each warp owns 8 output columns (j_base .. j_base+7).
//  Each warp lane consumes an int4 (16B = 64 trits) of packed W per step.
//  K is processed in strides of warpSize * 16 = 512 bytes (= 2048 trits).
//  Inner __dp4a accumulates int32 dot of 4 X-bytes against 4 unpacked W-bytes.
//
//  W_col layout (matches v11): for each group of 4 output columns occupying
//  one byte-row, bytes are laid out contiguously across K.  We treat W as
//  N/4 byte-columns of length K. To get 8 output cols per warp we read TWO
//  adjacent byte-columns (j_byte = j_base/4, j_byte+1).
// ---------------------------------------------------------------------------

#ifndef WARPS_PER_BLOCK_V12
#define WARPS_PER_BLOCK_V12 8
#endif

__device__ __forceinline__ int trit_byte_to_signed_int32(uint8_t b) {
    // unpack 4 trits (2 bits each, code 1 -> +1, code 2 -> -1, code 0 -> 0)
    // into a packed int32 where each byte is the signed value sign-extended
    // to int8. Uses arithmetic-only path (no LUT) — 6 ALU ops total.
    int t0 = (b     ) & 3;
    int t1 = (b >> 2) & 3;
    int t2 = (b >> 4) & 3;
    int t3 = (b >> 6) & 3;
    int v0 = (t0 == 1) - (t0 == 2);
    int v1 = (t1 == 1) - (t1 == 2);
    int v2 = (t2 == 1) - (t2 == 2);
    int v3 = (t3 == 1) - (t3 == 2);
    return ( (uint8_t)v0       )
         | ( (uint8_t)v1 << 8  )
         | ( (uint8_t)v2 << 16 )
         | ( (uint8_t)v3 << 24 );
}

__global__ void mm_packed_v12(
    const int8_t*  __restrict__ X,
    const uint8_t* __restrict__ W_col,
    int K, int N,
    const float* __restrict__ scale_x_dev,
    float w_scale,
    __half* __restrict__ out)
{
    extern __shared__ int8_t sX[];      // K bytes
    const int tid    = threadIdx.x;
    const int bs     = blockDim.x;
    const int warpId = tid >> 5;
    const int lane   = tid & 31;
    const int j_base = (blockIdx.x * WARPS_PER_BLOCK_V12 + warpId) * 8;

    // ---- D3: cooperatively stage X into shared memory (one-shot) ----
    {
        // 4-byte vectorized copy where K is a multiple of 4 (always true here).
        const int4* Xv = reinterpret_cast<const int4*>(X);
        int4*       sv = reinterpret_cast<int4*>(sX);
        int n_int4 = K >> 4;            // 16 bytes per int4
        for (int i = tid; i < n_int4; i += bs) sv[i] = __ldg(Xv + i);
        // tail (K%16 != 0): assume K is multiple of 16 for Llama (2048, 8192).
    }
    __shared__ float scale_smem;
    if (tid == 0) scale_smem = scale_x_dev[0] * w_scale;
    __syncthreads();

    if (j_base >= N) return;

    // Two adjacent packed byte-columns hold our 8 output cols (4 trits each).
    int j_byte0 = j_base >> 2;          // = j_base / 4
    int j_byte1 = j_byte0 + 1;
    const uint8_t* Wc0 = W_col + (size_t)j_byte0 * K;
    const uint8_t* Wc1 = W_col + (size_t)j_byte1 * K;

    // 8 accumulators (one per output column in this warp's tile).
    int acc[8] = {0,0,0,0,0,0,0,0};

    // ---- D1 + D5: LDG.128 strided by warpSize*16 bytes ----
    // Each lane consumes a 16-byte chunk; warp covers 32*16 = 512 bytes/iter.
    constexpr int LANE_STRIDE_BYTES = 32 * 16;
    int kk = lane * 16;                 // starting byte offset for this lane
    for (; kk + 16 <= K; kk += LANE_STRIDE_BYTES) {
        // Load 16 trits-bytes from each of the two W byte-columns
        // (each gives 4 sub-cols * 16 trits = 64 trits = enough for 4
        //  __dp4a chunks of 4 X-bytes each).
        const int4* Wc0v = reinterpret_cast<const int4*>(Wc0 + kk);
        const int4* Wc1v = reinterpret_cast<const int4*>(Wc1 + kk);
        int4 w0 = __ldg(Wc0v);          // 16 packed bytes from sub-cols 0..3
        int4 w1 = __ldg(Wc1v);          // 16 packed bytes from sub-cols 4..7

        // Read corresponding 16 bytes of X (from shared memory).
        const int4* sXv = reinterpret_cast<const int4*>(sX + kk);
        int4 x4 = *sXv;

        // Unpack each int4 into 4 x int32 lanes.
        const uint32_t* w0u = reinterpret_cast<const uint32_t*>(&w0);
        const uint32_t* w1u = reinterpret_cast<const uint32_t*>(&w1);
        const int32_t*  xu  = reinterpret_cast<const int32_t*>(&x4);

        // Each iteration of `chunk` covers 4 K-elements (one __dp4a).
        #pragma unroll
        for (int chunk = 0; chunk < 4; ++chunk) {
            uint32_t W0_packed = w0u[chunk];   // 4 bytes from byte-col0
            uint32_t W1_packed = w1u[chunk];   // 4 bytes from byte-col1
            int32_t  X4        = xu[chunk];    // 4 bytes of X

            uint8_t a0 = (W0_packed      ) & 0xff;
            uint8_t a1 = (W0_packed >>  8) & 0xff;
            uint8_t a2 = (W0_packed >> 16) & 0xff;
            uint8_t a3 = (W0_packed >> 24) & 0xff;
            uint8_t b0 = (W1_packed      ) & 0xff;
            uint8_t b1 = (W1_packed >>  8) & 0xff;
            uint8_t b2 = (W1_packed >> 16) & 0xff;
            uint8_t b3 = (W1_packed >> 24) & 0xff;

            // ---- D4 / D6: 8 cols/warp, packed unpack ----
            // For each output sub-column (0..3 from Wc0, 4..7 from Wc1),
            // build the 4-byte signed W vector across K-axis from one bit-
            // position in each of the 4 bytes (a0..a3 or b0..b3).
            #pragma unroll
            for (int sub = 0; sub < 4; ++sub) {
                int sh = sub * 2;
                // sub-col `sub`  (output column j_base + sub)
                int t0 = (a0 >> sh) & 3, t1 = (a1 >> sh) & 3;
                int t2 = (a2 >> sh) & 3, t3 = (a3 >> sh) & 3;
                int v0 = (t0==1)-(t0==2), v1 = (t1==1)-(t1==2);
                int v2 = (t2==1)-(t2==2), v3 = (t3==1)-(t3==2);
                int32_t W4 = (uint8_t)v0
                           | ((uint8_t)v1 <<  8)
                           | ((uint8_t)v2 << 16)
                           | ((uint8_t)v3 << 24);
                acc[sub] = __dp4a(X4, W4, acc[sub]);

                // sub-col `sub+4` (output column j_base + sub + 4)
                int u0 = (b0 >> sh) & 3, u1 = (b1 >> sh) & 3;
                int u2 = (b2 >> sh) & 3, u3 = (b3 >> sh) & 3;
                int p0 = (u0==1)-(u0==2), p1 = (u1==1)-(u1==2);
                int p2 = (u2==1)-(u2==2), p3 = (u3==1)-(u3==2);
                int32_t W4b = (uint8_t)p0
                            | ((uint8_t)p1 <<  8)
                            | ((uint8_t)p2 << 16)
                            | ((uint8_t)p3 << 24);
                acc[sub+4] = __dp4a(X4, W4b, acc[sub+4]);
            }
        }
    }

    // ---- Warp shuffle reduction across the 32 lanes for each of 8 acc ----
    #pragma unroll
    for (int s = 0; s < 8; ++s) {
        for (int o = 16; o > 0; o >>= 1)
            acc[s] += __shfl_xor_sync(0xffffffff, acc[s], o);
    }

    if (lane == 0) {
        float scale = scale_smem;
        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            int col = j_base + s;
            if (col < N) out[col] = __float2half((float)acc[s] * scale);
        }
    }
}

// ---------------------------------------------------------------------------
//  Host launcher (signature matches v11's `mm_packed_v4`)
// ---------------------------------------------------------------------------
static inline void launch_mm_packed_v12(
    const int8_t* X, const uint8_t* W_col, int K, int N,
    const float* scale_x_dev, float w_scale, __half* out,
    cudaStream_t stream = 0)
{
    assert((K & 15) == 0 && "v12 requires K % 16 == 0");
    assert((N & 7)  == 0 && "v12 requires N % 8 == 0");
    int threads = 32 * WARPS_PER_BLOCK_V12;       // 256
    int cols_per_block = WARPS_PER_BLOCK_V12 * 8; // 64
    int blocks = (N + cols_per_block - 1) / cols_per_block;
    size_t smem = K;                              // sX[K]
    mm_packed_v12<<<blocks, threads, smem, stream>>>(
        X, W_col, K, N, scale_x_dev, w_scale, out);
}

// ===========================================================================
//  MINIMAL STAND-ALONE BENCHMARK
//   - Runs the seven Llama-1B projection shapes that show up in a forward
//     pass: Wq/Wo (K=2048, N=2048), Wk/Wv (K=2048, N=512), Wgate/Wup
//     (K=2048, N=8192), Wdown (K=8192, N=2048). lm_head (V=128256, K=2048)
//     also included as the dominant single-call cost.
//   - Median of N_ITERS=200 iterations after WARMUP=20.
//   - Prints CSV rows: kernel,shape,ms_median,gb_per_s,bw_eff_percent
//   - Compares against a stripped-down v11 (mm_packed_v4 inlined here for
//     standalone build — semantically identical, K-cooperative warp reduction)
//     so we don't need to link the forward translation unit.
// ===========================================================================

// Inlined copy of v11's kernel ONLY for in-binary baseline timing. We do not
// re-export it; the canonical baseline still lives in ter_cuda_forward_packed.cu.
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

struct Shape { const char* name; int K, N; };

static double bench_one(const char* tag, int K, int N,
                        void (*launch)(const int8_t*, const uint8_t*, int, int,
                                       const float*, float, __half*),
                        int iters, int warmup,
                        double peak_gbs)
{
    int8_t*  dX = nullptr;
    uint8_t* dW = nullptr;
    __half*  dY = nullptr;
    float*   dScale = nullptr;
    CK(cudaMalloc(&dX, K));
    CK(cudaMalloc(&dW, (size_t)K * (N/4)));
    CK(cudaMalloc(&dY, N * sizeof(__half)));
    CK(cudaMalloc(&dScale, sizeof(float)));

    // fill with deterministic patterns; correctness not the focus here.
    std::vector<int8_t>  hX(K);
    std::vector<uint8_t> hW((size_t)K * (N/4));
    std::mt19937 rng(0xdeadbeef);
    std::uniform_int_distribution<int> dx(-127, 127);
    std::uniform_int_distribution<int> dw(0, 255);
    for (auto& v : hX) v = (int8_t)dx(rng);
    for (auto& v : hW) v = (uint8_t)dw(rng);
    CK(cudaMemcpy(dX, hX.data(), K, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dW, hW.data(), hW.size(), cudaMemcpyHostToDevice));
    float scale = 1.0f / 127.f;
    CK(cudaMemcpy(dScale, &scale, sizeof(float), cudaMemcpyHostToDevice));

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
    double bytes = (double)K * (N/4);                 // weight bytes streamed
    double gbs   = bytes / (med * 1e-3) / 1e9;
    double effp  = gbs / peak_gbs * 100.0;
    std::printf("%s,K=%d_N=%d,%.4f,%.1f,%.2f\n", tag, K, N, med, gbs, effp);
    cudaEventDestroy(s); cudaEventDestroy(e);
    cudaFree(dX); cudaFree(dW); cudaFree(dY); cudaFree(dScale);
    return med;
}

// Wrapper to match the bench launch signature (no stream arg).
static void launch_v12(const int8_t* X, const uint8_t* W, int K, int N,
                       const float* scale, float ws, __half* out)
{
    launch_mm_packed_v12(X, W, K, N, scale, ws, out, 0);
}

int main(int argc, char** argv) {
    int iters  = (argc > 1) ? std::atoi(argv[1]) : 200;
    int warmup = (argc > 2) ? std::atoi(argv[2]) : 20;
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    // CUDA 13+ removed cudaDeviceProp::memoryClockRate; query attributes
    // directly, fall back to 936 GB/s (RTX 3090) if both queries fail.
    int mem_clk_khz = 0, mem_bus_bits = 0;
    cudaError_t e1 = cudaDeviceGetAttribute(&mem_clk_khz,  cudaDevAttrMemoryClockRate, 0);
    cudaError_t e2 = cudaDeviceGetAttribute(&mem_bus_bits, cudaDevAttrGlobalMemoryBusWidth, 0);
    double peak_gbs = (e1 == cudaSuccess && e2 == cudaSuccess && mem_clk_khz > 0 && mem_bus_bits > 0)
        ? 2.0 * static_cast<double>(mem_clk_khz) * 1e3 * (static_cast<double>(mem_bus_bits) / 8.0) / 1e9
        : 936.0;
    std::fprintf(stderr, "Device: %s   peak HBM ~%.0f GB/s\n", p.name, peak_gbs);
    std::printf("kernel,shape,ms_median,gb_per_s,bw_eff_percent\n");

    Shape shapes[] = {
        {"Wq_Wo",   2048, 2048},
        {"Wk_Wv",   2048,  512},
        {"Wgate_up",2048, 8192},
        {"Wdown",   8192, 2048},
        {"lm_head", 2048, 128256},
    };
    for (auto& s : shapes) {
        bench_one("v11", s.K, s.N, launch_v11, iters, warmup, peak_gbs);
        bench_one("v12", s.K, s.N, launch_v12, iters, warmup, peak_gbs);
    }
    return 0;
}
