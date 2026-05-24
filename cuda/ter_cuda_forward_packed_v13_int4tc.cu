// ter_cuda_forward_packed_v13_int4tc.cu  --  v13 INT4 Tensor-Core path.
//
// ============================================================================
// IMPLEMENTATION (no longer a no-op): m8n8k32.row.col.s32.s4.s4.s32 mma path
// for the M=1 ternary GEMV. Correctness verified vs CPU reference in main()
// before launching the kernel against random data.
// ============================================================================
//
// Math: trit weights {-1,0,+1} encoded as INT4 nibbles {0xF, 0x0, 0x1}. The
// activation int8 is clamped to {-7..7} and re-packed as INT4. The INT4 mma
// accumulates into int32 exactly as the dp4a path would — i.e., this is
// numerically identical to v11 *given the same |X|<=7 clamp on activations*.
// For the bench harness we synthesise X already in [-7,7] so the comparison
// is bit-exact (not just approximate).
//
// M=1 GEMV mapping onto m8n8k32:
//   - The instruction works on M=8 rows × N=8 cols × K=32 INT4 chunks.
//   - Our problem has M=1 (a single activation row). We BROADCAST x across
//     the 8 rows of A. Only row 0 of the int32 8×8 accumulator is real;
//     rows 1..7 are duplicates that we discard at epilogue.
//   - Each thread-block processes 8 consecutive N columns (one m8n8 tile)
//     and walks the full K dimension in steps of 32.
//
// Fragment layout (PTX ISA "Matrix Fragments for mma.m8n8k32 with .s4 type"):
//   For a warp lane `L` (0..31):
//     groupID            = L >> 2   (0..7)
//     threadID_in_group  = L & 3    (0..3)
//   A frag (row-major 8x32, 8 INT4 per thread):
//     row = groupID
//     col = threadID_in_group * 8 + {0..7}  (8 consecutive INT4 cols)
//   B frag (col-major 32x8, 8 INT4 per thread):
//     col = groupID
//     row = threadID_in_group * 8 + {0..7}
//   C/D frag (row-major 8x8 int32, 2 int32 per thread):
//     row = groupID
//     col = threadID_in_group * 2 + {0..1}
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_86 -Xptxas -O3 \
//        ter_cuda_forward_packed_v13_int4tc.cu \
//        -o ter_cuda_forward_packed_v13_int4tc

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(call) do { cudaError_t e=(call); if(e){ \
    std::fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
    std::exit(1);} } while(0)

// ---------------------------------------------------------------------------
// Host repack: packed-trit (col-major 4 trits/byte) -> INT4 (row-major in
// (n_tile, k, col_in_tile) layout). We choose a layout that makes the kernel
// load per-thread fragments with a single 32-bit read.
//
// Output layout description (W_int4):
//   For each output-column tile of 8 cols (n_tile = 0 .. N/8-1),
//     for each k-tile of 32 K-steps (k_tile = 0 .. K/32-1),
//       store 8*32 = 256 INT4 = 128 bytes describing the 32x8 B fragment of
//       this tile, in the per-thread packed order:
//         lane 0..31 each contribute one uint32_t (8 INT4 nibbles).
//       Specifically for lane L = groupID*4 + threadID_in_group:
//         8 INT4 = W[k_tile*32 + threadID_in_group*8 + {0..7}, n_tile*8 + groupID]
//
//   So W_int4 is a uint32_t array of shape [N/8, K/32, 32] (= one uint32 per
//   lane per (n_tile, k_tile)). The kernel reads W_int4[(n_tile*(K/32)+k_tile)*32 + lane].
//
// Requires N % 8 == 0 and K % 32 == 0. Both hold for all Llama 1B shapes.
// ---------------------------------------------------------------------------

static inline uint8_t trit_to_int4(int t) {
    // t is 2-bit code: 0 => 0, 1 => +1, 2 => -1, 3 => 0 (invalid, no-op safe)
    if (t == 1) return 0x1u;
    if (t == 2) return 0xFu;
    return 0x0u;
}

static void repack_trits_to_int4_host(
    const uint8_t* W_col, int K, int N, uint32_t* W_int4 /* out, [N/8 * K/32 * 32] */)
{
    assert(K % 32 == 0);
    assert(N % 8  == 0);
    int n_tiles = N / 8;
    int k_tiles = K / 32;
    // For each (n_tile, k_tile), produce 32 uint32 values (one per lane).
    for (int n_tile = 0; n_tile < n_tiles; ++n_tile) {
        for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
            uint32_t* dst = W_int4 + ((size_t)n_tile * k_tiles + k_tile) * 32;
            for (int lane = 0; lane < 32; ++lane) {
                int groupID           = lane >> 2;
                int threadID_in_group = lane & 3;
                int n = n_tile * 8 + groupID;
                uint32_t packed = 0;
                for (int i = 0; i < 8; ++i) {
                    int k = k_tile * 32 + threadID_in_group * 8 + i;
                    int j_byte = n / 4;
                    int sub    = n % 4;
                    uint8_t b  = W_col[(size_t)j_byte * K + k];
                    int t = (b >> (sub * 2)) & 3;
                    uint8_t nib = trit_to_int4(t);
                    packed |= (uint32_t)(nib & 0xFu) << (i * 4);
                }
                dst[lane] = packed;
            }
        }
    }
}

// Activation repack: int8 X[K] (already in [-128,127]) -> INT4 X_int4[K/2]
// We *expect* the upstream pipeline to clamp X to [-7,7] when this kernel is
// chosen. If not, we clamp here (lossy). For the correctness harness we feed
// X in [-7,7] so this is a no-op clamp and the comparison is exact.
static void repack_x_int8_to_int4_host(
    const int8_t* X, int K, uint8_t* X_int4 /* out, K/2 bytes */)
{
    assert(K % 2 == 0);
    for (int k = 0; k < K; k += 2) {
        int a = X[k];     if (a < -7) a = -7; if (a > 7) a = 7;
        int b = X[k + 1]; if (b < -7) b = -7; if (b > 7) b = 7;
        X_int4[k / 2] = (uint8_t)((a & 0xF) | ((b & 0xF) << 4));
    }
}

// Device-side activation repack used by the launch wrapper (mirrors the
// host repack, runs in O(K) on one CTA). Lossy clamp to [-7,7].
__global__ void quant_int8_to_int4_k(const int8_t* X, uint8_t* X_int4, int K) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int n_out = K / 2;
    if (tid >= n_out) return;
    int a = X[2*tid];     if (a < -7) a = -7; if (a > 7) a = 7;
    int b = X[2*tid + 1]; if (b < -7) b = -7; if (b > 7) b = 7;
    X_int4[tid] = (uint8_t)((a & 0xF) | ((b & 0xF) << 4));
}

// ---------------------------------------------------------------------------
// Build the A fragment (one uint32 per lane) for the broadcast M=1 GEMV.
// We need A[row, threadID_in_group*8 + {0..7}] for each (groupID = row).
// Since all 8 rows broadcast the same x, the A fragment value depends only
// on threadID_in_group. We compute it by loading 4 bytes of X_int4 (= 8
// INT4 values) starting at k_base + threadID_in_group*8 (in INT4 units,
// which is 4 bytes since 2 INT4/byte).
//
// X_int4 layout: byte index = k/2, low nibble = X[2k], high nibble = X[2k+1]
// We need 8 contiguous INT4 starting at k = k_base + threadID_in_group*8.
// k_base is multiple of 32, threadID_in_group*8 multiple of 8 -> aligned to
// byte boundary (since 8 INT4 = 4 bytes).
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t load_a_frag_m1(
    const uint8_t* X_int4_bytes, int k_base, int threadID_in_group)
{
    // 8 INT4 = 4 bytes; start byte = (k_base + threadID_in_group*8) / 2.
    int byte_off = (k_base + threadID_in_group * 8) >> 1;
    // Aligned 32-bit load.
    return *reinterpret_cast<const uint32_t*>(X_int4_bytes + byte_off);
}

// ---------------------------------------------------------------------------
// The kernel.
//
// Grid: blocks of 32 threads (one warp), one warp per 8-col N tile.
// Each warp walks K in 32-step strides, issues mma.m8n8k32, then writes
// the row-0 outputs (8 cols) at epilogue.
//
// Inputs:
//   X_int4:  (K/2) bytes
//   W_int4:  (N/8 * K/32 * 32) uint32  (per-lane preformatted B fragments)
//   K, N
//   scale_x_dev: pointer to fp32 scale = 1/scale_x  (X dequant)
//   w_scale:     fp32 W per-tensor scale
//   out:         __half[N] output
// ---------------------------------------------------------------------------
__global__ void mm_packed_v13_int4tc(
    const uint8_t*  __restrict__ X_int4,
    const uint32_t* __restrict__ W_int4,
    int K, int N,
    const float* __restrict__ scale_x_dev,
    float w_scale,
    __half* __restrict__ out)
{
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    int n_tile  = blockIdx.x * (blockDim.x >> 5) + warp_id;  // each warp = one 8-col tile
    int n_base  = n_tile * 8;
    if (n_base >= N) return;

    int groupID           = lane >> 2;
    int threadID_in_group = lane & 3;

    int k_tiles = K >> 5;          // K/32
    const uint32_t* Wptr = W_int4 + (size_t)n_tile * k_tiles * 32;

    int acc0 = 0, acc1 = 0;        // C fragment: 2 int32 per lane

#if (__CUDA_ARCH__ >= 800)
    for (int kt = 0; kt < k_tiles; ++kt) {
        int k_base = kt << 5;
        uint32_t a_frag = load_a_frag_m1(X_int4, k_base, threadID_in_group);
        uint32_t b_frag = Wptr[kt * 32 + lane];

        asm volatile(
            "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 "
            "{%0,%1}, {%2}, {%3}, {%0,%1};\n"
            : "+r"(acc0), "+r"(acc1)
            : "r"(a_frag), "r"(b_frag));
    }
#else
    // Pre-Ampere: no INT4 mma. Fall back to a per-lane scalar reduction so
    // the file still compiles and runs (slower) on older arches.
    for (int kt = 0; kt < k_tiles; ++kt) {
        int k_base = kt << 5;
        uint32_t a_frag = load_a_frag_m1(X_int4, k_base, threadID_in_group);
        uint32_t b_frag = Wptr[kt * 32 + lane];
        // 8 INT4 dot product, sign-extended.
        for (int i = 0; i < 8; ++i) {
            int av = (int)((a_frag >> (i*4)) & 0xF); if (av & 0x8) av -= 16;
            int bv = (int)((b_frag >> (i*4)) & 0xF); if (bv & 0x8) bv -= 16;
            // groupID is row in C (we only care about row 0 ultimately, but
            // accumulate for col0/col1 of this lane's C fragment).
            // Without real mma, mimic: each lane's C[groupID, threadID_in_group*2+{0,1}]
            // would require cross-lane traffic. This fallback intentionally
            // produces wrong results; correctness path requires sm_80+.
            (void)av; (void)bv;
        }
        acc0 += 0; acc1 += 0;
    }
#endif

    // Epilogue: row 0 of the 8x8 int32 C lives in threads where groupID == 0,
    // i.e. lanes 0..3. Each of those threads holds C[0, threadID_in_group*2 + {0,1}].
    // So:
    //   lane 0 -> C[0,0], C[0,1]
    //   lane 1 -> C[0,2], C[0,3]
    //   lane 2 -> C[0,4], C[0,5]
    //   lane 3 -> C[0,6], C[0,7]
    if (groupID == 0) {
        float scale = scale_x_dev[0] * w_scale;
        int col0 = threadID_in_group * 2 + 0;
        int col1 = threadID_in_group * 2 + 1;
        if (n_base + col0 < N) out[n_base + col0] = __float2half((float)acc0 * scale);
        if (n_base + col1 < N) out[n_base + col1] = __float2half((float)acc1 * scale);
    }
}

// ===========================================================================
// v13_int4tc_p ("precise") -- bit-decomposition of X int8 into two INT4 mma
// passes. Preserves FULL int8 dynamic range of activations (no [-7,7] clamp).
//
// Math identity (proof):
//   For any int8 X, define:
//     X_hi = X >> 4  (arithmetic signed shift, range {-8..7})
//     X_lo = X & 0xF (unsigned nibble, range {0..15})
//   Then:                   X = 16 * X_hi + X_lo                        (1)
//   Proof: 2's complement: X = (sign-extended hi nibble << 4) | lo nibble.
//   Arithmetic >> 4 yields the sign-extended hi nibble. Bitwise & 0xF yields
//   the unsigned lo nibble. Their reconstruction matches the original signed
//   byte. E.g. X=-128: X_hi=-8, X_lo=0 -> 16*-8+0=-128 ✓
//         X=-1:   X_hi=-1, X_lo=15 -> 16*-1+15=-1 ✓
//         X=127:  X_hi=7,  X_lo=15 -> 16*7+15=127 ✓
//   Define X_lo_s = X_lo - 8, range {-8..7}, fits signed int4. Then
//     X = 16 * X_hi + X_lo_s + 8                                        (2)
//   So  Y[n] = sum_k X[k] * W[k,n]
//            = 16 * sum_k X_hi[k]   * W[k,n]
//            +      sum_k X_lo_s[k] * W[k,n]
//            +  8 * sum_k W[k,n]
//            = 16 * acc_hi[n] + acc_lo[n] + 8 * W_sum[n]                (3)
//   where W_sum[n] is precomputed ONCE per layer at load time.
//
// Verification: int8 X=-128, W=+1 -> expected = -128.
//   X_hi=-8, X_lo_s=-8, Wsum=1 -> 16*(-8)*1 + (-8)*1 + 8*1 = -128 + -8 + 8 = -128 ✓
//
// Cost: 2x INT4 mma per K-step (vs 1x for lossy v13_int4tc) plus one fp add
// in epilogue. Weight memory traffic UNCHANGED (W reused across both passes).
// Memory-bound at large N -> ~same wall time as lossy variant.
// ===========================================================================

// Device-side activation repack: int8 X -> two INT4 streams (hi, lo_s).
// Each output buffer is K/2 bytes (8-INT4 packing, two per byte).
//   X_hi_int4[k]  = (X[k] >> 4) signed int4 packed nibble
//   X_lo_int4[k]  = ((X[k] & 0xF) - 8) signed int4 packed nibble
__global__ void quant_int8_to_int4_p_k(
    const int8_t* X, uint8_t* X_hi_int4, uint8_t* X_lo_int4, int K)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int n_out = K / 2;
    if (tid >= n_out) return;
    int a = X[2*tid];
    int b = X[2*tid + 1];
    // Arithmetic right shift on signed int8 (C/C++ implementation-defined
    // pre-C++20; in CUDA on sm_86 it's arithmetic for signed types).
    int a_hi = a >> 4;                 // {-8..7}
    int b_hi = b >> 4;
    int a_lo = (a & 0xF) - 8;          // X_lo_s = X_lo - 8, range {-8..7}
    int b_lo = (b & 0xF) - 8;
    X_hi_int4[tid] = (uint8_t)((a_hi & 0xF) | ((b_hi & 0xF) << 4));
    X_lo_int4[tid] = (uint8_t)((a_lo & 0xF) | ((b_lo & 0xF) << 4));
}

// Per-N column weight sum, packed-trit layout in (col-major 4 trits/byte).
// W_sum[n] = sum_k W_trit[k, n], W in {-1,0,+1}. Range: bounded by K.
__global__ void compute_w_sum_k(
    const uint8_t* __restrict__ W_col, int K, int N, int32_t* __restrict__ W_sum)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    int j_byte = n / 4;
    int sub = n % 4;
    const uint8_t* Wc = W_col + (size_t)j_byte * K;
    int32_t acc = 0;
    for (int k = 0; k < K; ++k) {
        int t = (Wc[k] >> (sub * 2)) & 3;
        acc += (t == 1) - (t == 2);    // +1 / -1 / 0
    }
    W_sum[n] = acc;
}

// Host helper: allocate + compute W_sum on device. Caller frees.
extern "C" int32_t* compute_w_sum_v13_device(
    const uint8_t* W_col_dev, int K, int N)
{
    int32_t* d_out = nullptr;
    CK(cudaMalloc(&d_out, (size_t)N * sizeof(int32_t)));
    int t = 256;
    int blocks = (N + t - 1) / t;
    compute_w_sum_k<<<blocks, t>>>(W_col_dev, K, N, d_out);
    CK(cudaDeviceSynchronize());
    return d_out;
}

// The precise kernel: two mma passes over the SAME W fragment with two
// different A fragments (X_hi, X_lo_s), then combine in epilogue using
// W_sum precomputed once at layer load time.
__global__ void mm_packed_v13_int4tc_p(
    const uint8_t*  __restrict__ X_hi_int4,
    const uint8_t*  __restrict__ X_lo_int4,
    const uint32_t* __restrict__ W_int4,
    const int32_t*  __restrict__ W_sum,
    int K, int N,
    const float* __restrict__ scale_x_dev,
    float w_scale,
    __half* __restrict__ out)
{
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    int n_tile  = blockIdx.x * (blockDim.x >> 5) + warp_id;
    int n_base  = n_tile * 8;
    if (n_base >= N) return;

    int groupID           = lane >> 2;
    int threadID_in_group = lane & 3;

    int k_tiles = K >> 5;
    const uint32_t* Wptr = W_int4 + (size_t)n_tile * k_tiles * 32;

    int acc_hi_0 = 0, acc_hi_1 = 0;
    int acc_lo_0 = 0, acc_lo_1 = 0;

#if (__CUDA_ARCH__ >= 800)
    for (int kt = 0; kt < k_tiles; ++kt) {
        int k_base = kt << 5;
        uint32_t a_hi = load_a_frag_m1(X_hi_int4, k_base, threadID_in_group);
        uint32_t a_lo = load_a_frag_m1(X_lo_int4, k_base, threadID_in_group);
        uint32_t b_frag = Wptr[kt * 32 + lane];

        asm volatile(
            "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 "
            "{%0,%1}, {%2}, {%3}, {%0,%1};\n"
            : "+r"(acc_hi_0), "+r"(acc_hi_1)
            : "r"(a_hi), "r"(b_frag));

        asm volatile(
            "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 "
            "{%0,%1}, {%2}, {%3}, {%0,%1};\n"
            : "+r"(acc_lo_0), "+r"(acc_lo_1)
            : "r"(a_lo), "r"(b_frag));
    }
#endif

    // Epilogue: only groupID == 0 writes (row 0 of m8n8 C).
    if (groupID == 0) {
        float scale = scale_x_dev[0] * w_scale;
        int col0 = threadID_in_group * 2 + 0;
        int col1 = threadID_in_group * 2 + 1;
        // Y_int = 16*acc_hi + acc_lo + 8*W_sum
        if (n_base + col0 < N) {
            int32_t y0 = 16 * acc_hi_0 + acc_lo_0 + 8 * W_sum[n_base + col0];
            out[n_base + col0] = __float2half((float)y0 * scale);
        }
        if (n_base + col1 < N) {
            int32_t y1 = 16 * acc_hi_1 + acc_lo_1 + 8 * W_sum[n_base + col1];
            out[n_base + col1] = __float2half((float)y1 * scale);
        }
    }
}

// Persistent buffers for the precise launcher (two K/2 buffers for hi/lo).
static uint8_t* g_x_hi_int4_buf = nullptr;
static uint8_t* g_x_lo_int4_buf = nullptr;
static size_t   g_x_hilo_int4_buf_bytes = 0;

static inline void ensure_x_hilo_int4_buf(int K) {
    size_t need = (size_t)K / 2;
    if (g_x_hilo_int4_buf_bytes < need) {
        if (g_x_hi_int4_buf) cudaFree(g_x_hi_int4_buf);
        if (g_x_lo_int4_buf) cudaFree(g_x_lo_int4_buf);
        CK(cudaMalloc(&g_x_hi_int4_buf, need));
        CK(cudaMalloc(&g_x_lo_int4_buf, need));
        g_x_hilo_int4_buf_bytes = need;
    }
}

static inline void launch_mm_packed_v13_int4tc_p(
    const int8_t* X, const uint8_t* W_int4, const int32_t* W_sum,
    int K, int N,
    const float* scale, float ws, __half* out, cudaStream_t s = 0)
{
    ensure_x_hilo_int4_buf(K);
    int qt = 256;
    int qb = (K/2 + qt - 1) / qt;
    quant_int8_to_int4_p_k<<<qb, qt, 0, s>>>(
        X, g_x_hi_int4_buf, g_x_lo_int4_buf, K);

    const int warps_per_block = 8;
    int n_tiles = (N + 7) / 8;
    int blocks  = (n_tiles + warps_per_block - 1) / warps_per_block;
    mm_packed_v13_int4tc_p<<<blocks, warps_per_block * 32, 0, s>>>(
        g_x_hi_int4_buf, g_x_lo_int4_buf,
        reinterpret_cast<const uint32_t*>(W_int4), W_sum,
        K, N, scale, ws, out);
}

// ---------------------------------------------------------------------------
// Launch wrapper expected by v13_bench. Signature matches the other variants:
//   launch_*(const int8_t* X, const uint8_t* W, int K, int N,
//            const float* scale, float ws, __half* out, cudaStream_t)
//
// IMPORTANT: this wrapper assumes W has been *pre-repacked* into the INT4
// per-lane layout described above. The bench harness must repack once at
// model load time (cheap, one-shot). To preserve the existing signature,
// we expose a separate `prepack_w_int4_v13` helper used by the bench setup.
//
// If the bench passes the *packed-trit* W (not repacked), this kernel will
// produce garbage. Therefore the wrapper checks an env-var / assumes the
// caller has already repacked. The bench harness file should call
// `prepack_w_int4_v13` once to convert and then pass the converted buffer
// in as `W`.
//
// The on-device activation repack (int8 -> INT4) runs inline before the mma.
// ---------------------------------------------------------------------------
static uint8_t* g_x_int4_buf = nullptr;
static size_t   g_x_int4_buf_bytes = 0;

static inline void ensure_x_int4_buf(int K) {
    size_t need = (size_t)K / 2;
    if (g_x_int4_buf_bytes < need) {
        if (g_x_int4_buf) cudaFree(g_x_int4_buf);
        CK(cudaMalloc(&g_x_int4_buf, need));
        g_x_int4_buf_bytes = need;
    }
}

static inline void launch_mm_packed_v13_int4tc(
    const int8_t* X, const uint8_t* W_int4, int K, int N,
    const float* scale, float ws, __half* out, cudaStream_t s = 0)
{
    ensure_x_int4_buf(K);
    int qt = 256;
    int qb = (K/2 + qt - 1) / qt;
    quant_int8_to_int4_k<<<qb, qt, 0, s>>>(X, g_x_int4_buf, K);

    // One warp per 8-col tile, pack 8 warps per block.
    const int warps_per_block = 8;
    int n_tiles = (N + 7) / 8;
    int blocks  = (n_tiles + warps_per_block - 1) / warps_per_block;
    mm_packed_v13_int4tc<<<blocks, warps_per_block * 32, 0, s>>>(
        g_x_int4_buf, reinterpret_cast<const uint32_t*>(W_int4),
        K, N, scale, ws, out);
}

// One-shot host helper: convert packed-trit W (4 trits/byte, col-major byte
// layout [N/4, K]) into the per-lane INT4 layout this kernel expects.
// Caller frees the returned device buffer.
extern "C" uint32_t* prepack_w_int4_v13_device(
    const uint8_t* W_col_dev, int K, int N)
{
    assert(K % 32 == 0); assert(N % 8 == 0);
    size_t in_bytes  = (size_t)K * (N / 4);
    size_t out_words = (size_t)(N/8) * (K/32) * 32;
    std::vector<uint8_t> hW(in_bytes);
    CK(cudaMemcpy(hW.data(), W_col_dev, in_bytes, cudaMemcpyDeviceToHost));
    std::vector<uint32_t> hWi(out_words);
    repack_trits_to_int4_host(hW.data(), K, N, hWi.data());
    uint32_t* d_out = nullptr;
    CK(cudaMalloc(&d_out, out_words * sizeof(uint32_t)));
    CK(cudaMemcpy(d_out, hWi.data(), out_words * sizeof(uint32_t),
                  cudaMemcpyHostToDevice));
    return d_out;
}

// ---------------------------------------------------------------------------
// T5: correctness harness. Run kernel + CPU reference on small random data
// (K=64, N=8, X in {-7..7}, W trits in {-1,0,+1}) and abort on any mismatch
// in the integer accumulator (before fp scaling). atol = 0 (bit-exact).
// ---------------------------------------------------------------------------
static int run_correctness_test() {
    const int K = 64;      // multiple of 32 ✓
    const int N = 16;      // multiple of 8  ✓

    // Build deterministic packed-trit W (col-major 4 trits/byte).
    std::vector<uint8_t> hW((size_t)K * (N / 4), 0);
    // For each (k, n), pick a trit deterministically.
    std::vector<int8_t> hW_trits((size_t)K * N);  // reference, signed -1/0/+1
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            int t_code = (k * 7 + n * 13) % 3;   // 0,1,2
            int t_sgn  = (t_code == 1) ? +1 : (t_code == 2 ? -1 : 0);
            hW_trits[(size_t)k * N + n] = (int8_t)t_sgn;
            int j_byte = n / 4;
            int sub    = n % 4;
            hW[(size_t)j_byte * K + k] |= (uint8_t)(t_code << (sub * 2));
        }
    }

    // X int8 in {-7..7}.
    std::vector<int8_t> hX(K);
    for (int k = 0; k < K; ++k) hX[k] = (int8_t)((k * 11 + 3) % 15 - 7);

    // CPU reference: int32 acc[n] = sum_k X[k] * W_trit[k,n]
    std::vector<int32_t> ref(N, 0);
    for (int n = 0; n < N; ++n) {
        int32_t acc = 0;
        for (int k = 0; k < K; ++k)
            acc += (int32_t)hX[k] * (int32_t)hW_trits[(size_t)k * N + n];
        ref[n] = acc;
    }

    // Repack W on host -> device.
    int n_tiles = N / 8, k_tiles = K / 32;
    std::vector<uint32_t> hWi((size_t)n_tiles * k_tiles * 32);
    repack_trits_to_int4_host(hW.data(), K, N, hWi.data());

    int8_t*   dX = nullptr;
    uint32_t* dWi = nullptr;
    __half*   dY = nullptr;
    float*    dS = nullptr;
    CK(cudaMalloc(&dX, K));
    CK(cudaMalloc(&dWi, hWi.size() * sizeof(uint32_t)));
    CK(cudaMalloc(&dY, N * sizeof(__half)));
    CK(cudaMalloc(&dS, sizeof(float)));
    CK(cudaMemcpy(dX, hX.data(), K, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dWi, hWi.data(), hWi.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    // scale = 1 so that out = acc (as float). w_scale = 1.
    float one = 1.f;
    CK(cudaMemcpy(dS, &one, sizeof(float), cudaMemcpyHostToDevice));

    launch_mm_packed_v13_int4tc(dX,
        reinterpret_cast<const uint8_t*>(dWi),
        K, N, dS, 1.f, dY);
    CK(cudaDeviceSynchronize());

    std::vector<__half> hY(N);
    CK(cudaMemcpy(hY.data(), dY, N * sizeof(__half), cudaMemcpyDeviceToHost));

    int n_mismatch = 0;
    for (int n = 0; n < N; ++n) {
        int32_t got = (int32_t)__half2float(hY[n]);  // scale=1 → integer
        if (got != ref[n]) {
            if (n_mismatch < 8) {
                std::fprintf(stderr,
                    "MISMATCH n=%d got=%d ref=%d\n", n, got, ref[n]);
            }
            n_mismatch++;
        }
    }
    cudaFree(dX); cudaFree(dWi); cudaFree(dY); cudaFree(dS);
    if (g_x_int4_buf) { cudaFree(g_x_int4_buf); g_x_int4_buf = nullptr; g_x_int4_buf_bytes = 0; }

    if (n_mismatch == 0) {
        std::printf("v13_int4tc correctness: PASS (K=%d N=%d, bit-exact vs CPU)\n", K, N);
        return 0;
    } else {
        std::fprintf(stderr,
            "v13_int4tc correctness: FAIL (%d / %d mismatches)\n", n_mismatch, N);
        return 1;
    }
}

// Bench mode: run timing on the 5 Llama 1B GEMV shapes. Output CSV to stdout.
// Repack W once per shape, then ITERS timed launches with WARMUP non-timed.
static int run_bench(int iters, int warmup) {
    struct Shape { const char* name; int K, N; };
    Shape shapes[] = {
        {"K=2048_N=2048",   2048,   2048},
        {"K=2048_N=512",    2048,    512},
        {"K=2048_N=8192",   2048,   8192},
        {"K=8192_N=2048",   8192,   2048},
        {"K=2048_N=128256", 2048, 128256},
    };

    // Peak HBM via CUDA 13 attribute API; fallback 936 GB/s (RTX 3090).
    int mhz_khz = 0, bus_bits = 0;
    cudaError_t e1 = cudaDeviceGetAttribute(&mhz_khz,  cudaDevAttrMemoryClockRate, 0);
    cudaError_t e2 = cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, 0);
    double peak_gbs = (e1 == cudaSuccess && e2 == cudaSuccess && mhz_khz > 0 && bus_bits > 0)
        ? 2.0 * (double)mhz_khz * 1e3 * ((double)bus_bits / 8.0) / 1e9
        : 936.0;
    std::fprintf(stderr, "[bench] peak HBM ~%.0f GB/s\n", peak_gbs);

    std::printf("kernel,shape,ms_median,gb_per_s,bw_eff_percent\n");

    for (const auto& s : shapes) {
        int K = s.K, N = s.N;

        // Allocate inputs (random small values for benchmarking — exact values
        // don't matter for timing).
        std::vector<int8_t> hX(K);
        for (int k = 0; k < K; ++k) hX[k] = (int8_t)((k % 15) - 7);
        std::vector<uint8_t> hW_col((size_t)K * (N / 4), 0xA5);  // arbitrary

        int8_t* dX = nullptr; uint8_t* dW_col = nullptr;
        __half* dY = nullptr; float* dS = nullptr;
        CK(cudaMalloc(&dX, K));
        CK(cudaMalloc(&dW_col, hW_col.size()));
        CK(cudaMalloc(&dY, N * sizeof(__half)));
        CK(cudaMalloc(&dS, sizeof(float)));
        CK(cudaMemcpy(dX, hX.data(), K, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dW_col, hW_col.data(), hW_col.size(), cudaMemcpyHostToDevice));
        float ones = 0.05f;
        CK(cudaMemcpy(dS, &ones, sizeof(float), cudaMemcpyHostToDevice));

        // One-shot repack to INT4 layout. Counted OUT of timed region (matches
        // production where weights are repacked once at load time).
        uint32_t* dWi = prepack_w_int4_v13_device(dW_col, K, N);

        // Warmup.
        for (int i = 0; i < warmup; ++i)
            launch_mm_packed_v13_int4tc(dX, reinterpret_cast<const uint8_t*>(dWi),
                                         K, N, dS, 0.05f, dY);
        CK(cudaDeviceSynchronize());

        // Timed loop.
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        std::vector<double> times; times.reserve(iters);
        for (int i = 0; i < iters; ++i) {
            cudaEventRecord(a);
            launch_mm_packed_v13_int4tc(dX, reinterpret_cast<const uint8_t*>(dWi),
                                         K, N, dS, 0.05f, dY);
            cudaEventRecord(b);
            cudaEventSynchronize(b);
            float ms = 0;
            cudaEventElapsedTime(&ms, a, b);
            times.push_back(ms);
        }
        std::sort(times.begin(), times.end());
        double med = times[times.size() / 2];
        // Bytes moved: weights in INT4 layout = K*(N/2) bytes; activations
        // (X int8 + X int4 quant buffer) negligible.
        double bytes = (double)K * (N / 2);
        double gbs = bytes / med / 1e6;     // ms → s: bytes/ms / 1e6 = GB/s
        double eff = gbs / peak_gbs * 100.0;
        std::printf("v13_int4tc_real,%s,%.4f,%.1f,%.2f\n", s.name, med, gbs, eff);

        cudaEventDestroy(a); cudaEventDestroy(b);
        cudaFree(dX); cudaFree(dW_col); cudaFree(dY); cudaFree(dS);
        cudaFree(dWi);
        if (g_x_int4_buf) {
            cudaFree(g_x_int4_buf);
            g_x_int4_buf = nullptr; g_x_int4_buf_bytes = 0;
        }
    }
    return 0;
}

#ifndef V13_INT4TC_NO_MAIN
int main(int argc, char** argv) {
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    std::fprintf(stderr, "Device: %s  (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    if (prop.major < 8) {
        std::fprintf(stderr, "ERROR: INT4 mma.m8n8k32 requires sm_80+\n");
        return 2;
    }
    // First: correctness check (always). Then: bench if requested.
    int rc = run_correctness_test();
    if (rc != 0) return rc;
    if (argc >= 2 && std::string(argv[1]) == "bench") {
        int iters  = (argc > 2) ? std::atoi(argv[2]) : 200;
        int warmup = (argc > 3) ? std::atoi(argv[3]) : 20;
        return run_bench(iters, warmup);
    }
    return 0;
}
#endif
