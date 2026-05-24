// ter_cuda_forward_packed_v13_lut.cu  --  v13 LUT-unpack variant.
// Standalone kernel + host launcher. Hypothesis: v11's bottleneck is ALU
// pressure from the arithmetic trit unpack (~24 ops per __dp4a). Replace
// the entire 6-ops/byte arithmetic unpack with a 256-entry constant-memory
// LUT keyed on the packed byte. The 4 unpacked signed-int8 trits per entry
// live contiguous in __constant__ memory; the warp reads the entry as a
// single int32 (4 int8 lanes ready for __dp4a immediately).
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_86 -Xptxas -O3 \
//        ter_cuda_forward_packed_v13_lut.cu -o ter_cuda_forward_packed_v13_lut
//
// Design (A/B clean — only the unpack changes vs v11):
//   * Same per-warp K split, same 4 output cols/warp, same warp shuffle
//     reduction, same scalar uint32 weight load. Just no arithmetic unpack.
//   * kUnpackLUT[256] : int32, each int32 holds 4 int8 trits in {-1,0,+1}.
//     Initialised on host via cudaMemcpyToSymbol before any kernel launch.
//   * Per __dp4a we issue 4 LUT loads (one per byte of the 4-byte W chunk)
//     and 4 byte-shifts of these int32s to align the per-sub-column trit
//     stream — but each load is constant-cache broadcast-friendly when many
//     lanes hit the same byte, and falls back to LDC + L1 hits otherwise.
//   * Net ALU per __dp4a: ~10 ops (load index + shifts + pack) vs ~24 in
//     v11. Predicted gain biggest on Wq/Wo (2048x2048) and Wk/Wv where
//     v11's bw_eff is 3-12% (compute-bound regime).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <vector>

#define CK(call) do { cudaError_t e=(call); if(e){ \
    std::fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
    std::exit(1);} } while(0)

// ---- constant LUT: 256 entries x 4 signed int8 trits = 1024 bytes ----
// Layout: int32 per entry, byte i (i=0..3) = signed trit for bit-pair i.
// trit codes: 0 -> 0, 1 -> +1, 2 -> -1, 3 -> 0 (invalid sentinel).
__constant__ int32_t kUnpackLUT[256];

static void init_unpack_lut_host() {
    int32_t h[256];
    for (int b = 0; b < 256; ++b) {
        int8_t v[4];
        for (int s = 0; s < 4; ++s) {
            int t = (b >> (s * 2)) & 3;
            v[s] = (int8_t)((t == 1) - (t == 2));  // 0->0, 1->+1, 2->-1, 3->0
        }
        h[b] = ((uint8_t)v[0])
             | ((uint8_t)v[1] << 8)
             | ((uint8_t)v[2] << 16)
             | ((uint8_t)v[3] << 24);
    }
    CK(cudaMemcpyToSymbol(kUnpackLUT, h, sizeof(h)));
}

// ---------------------------------------------------------------------------
// Kernel: same skeleton as v11 (mm_packed_v4) — single change is LUT unpack.
// Each warp owns 4 output columns starting at j_base = warp_id * 4.
// ---------------------------------------------------------------------------
__global__ void mm_packed_v13_lut(
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
    int kstart = lane * kpt;
    int kend   = min(K, kstart + kpt);

    int acc[4] = {0, 0, 0, 0};

    int kk = kstart;
    for (; kk + 4 <= kend; kk += 4) {
        uint32_t w  = *reinterpret_cast<const uint32_t*>(Wc + kk);
        int32_t  X4 = *reinterpret_cast<const int32_t*>(X + kk);

        // ---- LUT unpack: 4 const-cache loads, one per packed byte ----
        // Each entry is an int32 of 4 signed trits {byte0=bit-pair0,
        // byte1=bit-pair1, byte2=bit-pair2, byte3=bit-pair3}.
        int32_t l0 = kUnpackLUT[(w      ) & 0xff];  // byte 0  of K-window
        int32_t l1 = kUnpackLUT[(w >>  8) & 0xff];  // byte 1
        int32_t l2 = kUnpackLUT[(w >> 16) & 0xff];  // byte 2
        int32_t l3 = kUnpackLUT[(w >> 24) & 0xff];  // byte 3

        // For sub-col `sub` (output col j_base+sub), the corresponding signed
        // trit comes from BYTE `sub` of l0,l1,l2,l3 respectively. We assemble
        // the 4-byte signed W vector (one byte per K-step) for the __dp4a.
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            int sh = sub * 8;
            uint8_t v0 = (l0 >> sh) & 0xff;
            uint8_t v1 = (l1 >> sh) & 0xff;
            uint8_t v2 = (l2 >> sh) & 0xff;
            uint8_t v3 = (l3 >> sh) & 0xff;
            int32_t W4 = v0 | (v1 << 8) | (v2 << 16) | (v3 << 24);
            acc[sub] = __dp4a(X4, W4, acc[sub]);
        }
    }
    // tail
    for (; kk < kend; ++kk) {
        int xv = X[kk]; if (xv == 0) continue;
        uint8_t b = Wc[kk];
        int32_t lut = kUnpackLUT[b];
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            int8_t v = (int8_t)((lut >> (sub * 8)) & 0xff);
            acc[sub] += xv * (int)v;
        }
    }

    // warp shuffle reduction
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

static inline void launch_mm_packed_v13_lut(
    const int8_t* X, const uint8_t* W, int K, int N,
    const float* scale, float ws, __half* out, cudaStream_t s = 0)
{
    int t = 256;
    int warps = t / 32;
    int blocks = (N / 4 + warps - 1) / warps;
    mm_packed_v13_lut<<<blocks, t, 0, s>>>(X, W, K, N, scale, ws, out);
}

// ---------------------------------------------------------------------------
// Tiny self-bench so this .cu compiles + runs standalone.
// ---------------------------------------------------------------------------
int main() {
    init_unpack_lut_host();
    int K = 2048, N = 2048;
    int8_t*  dX; uint8_t* dW; __half* dY; float* dS;
    CK(cudaMalloc(&dX, K));
    CK(cudaMalloc(&dW, (size_t)K * (N / 4)));
    CK(cudaMalloc(&dY, N * sizeof(__half)));
    CK(cudaMalloc(&dS, sizeof(float)));
    std::vector<int8_t>  hX(K, 1);
    std::vector<uint8_t> hW((size_t)K * (N / 4), 0x55);
    float scale = 1.0f / 127.f;
    CK(cudaMemcpy(dX, hX.data(), K, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dW, hW.data(), hW.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dS, &scale, sizeof(float), cudaMemcpyHostToDevice));
    for (int i = 0; i < 5; ++i)
        launch_mm_packed_v13_lut(dX, dW, K, N, dS, 0.05f, dY);
    CK(cudaDeviceSynchronize());
    std::printf("v13_lut OK (K=%d N=%d)\n", K, N);
    cudaFree(dX); cudaFree(dW); cudaFree(dY); cudaFree(dS);
    return 0;
}
