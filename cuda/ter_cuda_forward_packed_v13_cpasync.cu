// ter_cuda_forward_packed_v13_cpasync.cu  --  v13 cp.async X-staging variant.
// Same v11 inner loop (arithmetic unpack, 4 cols/warp, scalar uint32 LDG of W),
// but X[K] is staged into shared memory using cp.async (sm_80+) so the LDG
// of X is decoupled from the compute pipeline. A single async-copy stage
// (no double buffer) — the win comes from overlapping the X gmem load with
// the first __dp4a iterations and from freeing L1 from X traffic.
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_86 -Xptxas -O3 \
//        ter_cuda_forward_packed_v13_cpasync.cu \
//        -o ter_cuda_forward_packed_v13_cpasync
//
// Notes:
//   * Uses the PTX `cp.async.cg.shared.global` directly (16-byte path) to
//     avoid a hard dep on <cuda_pipeline.h> headers across CUDA versions.
//   * `cg` cache-global hint matches the streaming pattern (X is touched
//     exactly once per kernel invocation).
//   * After all lanes issue cp.async, a `cp.async.commit_group;` then
//     `cp.async.wait_all;` plus __syncthreads makes sX[K] visible.

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

// 16-byte cp.async from global -> shared. Requires sm_80+.
__device__ __forceinline__ void cp_async_16(void* smem_ptr, const void* gmem_ptr) {
    unsigned smem_int = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_int), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::);
}

// ---------------------------------------------------------------------------
// Kernel: identical compute to v11; only X-staging path changes.
// ---------------------------------------------------------------------------
__global__ void mm_packed_v13_cpasync(
    const int8_t*  __restrict__ X,
    const uint8_t* __restrict__ W_col,
    int K, int N,
    const float* __restrict__ scale_x_dev,
    float w_scale,
    __half* __restrict__ out)
{
    extern __shared__ int8_t sX[];   // K bytes

    const int tid = threadIdx.x;
    const int bs  = blockDim.x;

    // ---- cp.async stage of X[K] into shared memory, 16 bytes per thread ----
    // K is a multiple of 16 for all Llama-1B shapes (2048, 8192).
    int n_int16 = K >> 4;                       // number of 16-byte chunks
    for (int i = tid; i < n_int16; i += bs) {
        cp_async_16(sX + i * 16, X + i * 16);
    }
    cp_async_commit();
    cp_async_wait_all();

    __shared__ float scale_smem;
    if (tid == 0) scale_smem = scale_x_dev[0] * w_scale;
    __syncthreads();

    int warp_block_id = (blockIdx.x * blockDim.x + tid) / 32;
    int lane   = tid & 31;
    int j_base = warp_block_id * 4;
    if (j_base >= N) return;
    int j_byte = j_base / 4;
    const uint8_t* Wc = W_col + (size_t)j_byte * K;

    int kpt = (K + 31) / 32;
    int kstart = lane * kpt;
    int kend   = min(K, kstart + kpt);

    int acc[4] = {0, 0, 0, 0};

    int kk = kstart;
    for (; kk + 4 <= kend; kk += 4) {
        uint32_t w  = *reinterpret_cast<const uint32_t*>(Wc + kk);
        int32_t  X4 = *reinterpret_cast<const int32_t*>(sX + kk);
        uint8_t b0 = w & 0xff, b1 = (w >> 8) & 0xff,
                b2 = (w >> 16) & 0xff, b3 = (w >> 24) & 0xff;
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            int sb = sub * 2;
            int t0 = (b0 >> sb) & 3, t1 = (b1 >> sb) & 3;
            int t2 = (b2 >> sb) & 3, t3 = (b3 >> sb) & 3;
            int wv0 = (t0 == 1) - (t0 == 2);
            int wv1 = (t1 == 1) - (t1 == 2);
            int wv2 = (t2 == 1) - (t2 == 2);
            int wv3 = (t3 == 1) - (t3 == 2);
            int32_t W4 = (uint8_t)wv0
                       | ((uint8_t)wv1 << 8)
                       | ((uint8_t)wv2 << 16)
                       | ((uint8_t)wv3 << 24);
            acc[sub] = __dp4a(X4, W4, acc[sub]);
        }
    }
    for (; kk < kend; ++kk) {
        int xv = sX[kk]; if (xv == 0) continue;
        uint8_t b = Wc[kk];
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            int t = (b >> (sub * 2)) & 3;
            acc[sub] += xv * ((t == 1) - (t == 2));
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

static inline void launch_mm_packed_v13_cpasync(
    const int8_t* X, const uint8_t* W, int K, int N,
    const float* scale, float ws, __half* out, cudaStream_t s = 0)
{
    int t = 256;
    int warps = t / 32;
    int blocks = (N / 4 + warps - 1) / warps;
    size_t smem = K;
    mm_packed_v13_cpasync<<<blocks, t, smem, s>>>(X, W, K, N, scale, ws, out);
}

int main() {
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
        launch_mm_packed_v13_cpasync(dX, dW, K, N, dS, 0.05f, dY);
    CK(cudaDeviceSynchronize());
    std::printf("v13_cpasync OK (K=%d N=%d)\n", K, N);
    cudaFree(dX); cudaFree(dW); cudaFree(dY); cudaFree(dS);
    return 0;
}
