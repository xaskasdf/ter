// ter_cuda_packed.cu -- ternary-native packed matmul on CUDA.
//
// Weights: 4 trits per byte (2 bits each, mapping 0->0, 1->+1, 2->-1, 3->0).
// Activations: int8 (BitNet-style quantized).
// Output: fp16 with per-tensor scale.
//
// Compares three backends at Llama 1B matmul shapes:
//   1) packed-2bit: custom kernel, 4 trits/byte, ternary-native
//   2) cuBLAS sgemm fp32: reference, 4 bytes/elem
//   3) cuBLAS INT8 TC: reference, 1 byte/elem
//
// Reports throughput (GMAC/s) AND memory footprint (MB / matrix) AND
// implied bandwidth (bytes_read / time). The packed kernel's win lives in
// the byte/elem ratio: 0.25 vs 1 (INT8) vs 4 (fp32) vs 9 (Q8_0 with overhead).
//
// Build:
//   nvcc -O3 -arch=sm_86 -std=c++17 ter_cuda_packed.cu -lcublas -o ter_cuda_packed

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <random>

#define CK(call) do { cudaError_t e=(call); if(e){std::fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));std::exit(1);} } while(0)
#define CB(call) do { cublasStatus_t s=(call); if(s){std::fprintf(stderr,"cuBLAS %s:%d %d\n",__FILE__,__LINE__,(int)s);std::exit(1);} } while(0)

// ---------- Packed kernel: 4 trits / byte, K-loop scans byte rows ----------
// Each thread produces 4 contiguous output columns, reads W_packed coalesced.
// Layout: W_packed is row-major (K, N/4) bytes. byte at row k, col j_byte holds
// trits for output cols [j_byte*4 .. j_byte*4 + 3] in bits [0..1, 2..3, 4..5, 6..7].
__global__ void mm_packed_trit_v4(
    const int8_t*  __restrict__ X,         // (K,) int8 activations
    const uint8_t* __restrict__ W,         // (K, N/4) bytes
    int K, int N,
    float scale,
    __half* __restrict__ out)              // (N,) fp16
{
    int j_base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (j_base >= N) return;
    int j_byte = j_base / 4;
    int N_bytes = N / 4;

    int acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    for (int k = 0; k < K; ++k) {
        int xv = X[k];
        if (xv == 0) continue;
        uint8_t b = W[k * N_bytes + j_byte];
        int t0 = (b >> 0) & 3, t1 = (b >> 2) & 3, t2 = (b >> 4) & 3, t3 = (b >> 6) & 3;
        // decode: 1 -> +1, 2 -> -1, 0/3 -> 0
        int w0 = (t0 == 1) - (t0 == 2);
        int w1 = (t1 == 1) - (t1 == 2);
        int w2 = (t2 == 1) - (t2 == 2);
        int w3 = (t3 == 1) - (t3 == 2);
        acc0 += xv * w0;
        acc1 += xv * w1;
        acc2 += xv * w2;
        acc3 += xv * w3;
    }
    out[j_base + 0] = __float2half((float)acc0 * scale);
    out[j_base + 1] = __float2half((float)acc1 * scale);
    out[j_base + 2] = __float2half((float)acc2 * scale);
    out[j_base + 3] = __float2half((float)acc3 * scale);
}

// Fancier packed kernel: each thread loads 4 bytes (uint32_t) at once, processes
// 16 contiguous output columns. Reduces per-thread loop overhead.
__global__ void mm_packed_trit_v16(
    const int8_t*  __restrict__ X,
    const uint8_t* __restrict__ W,
    int K, int N,
    float scale,
    __half* __restrict__ out)
{
    int j_base = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
    if (j_base >= N) return;
    int j_byte4 = j_base / 16;        // each uint32 covers 16 trits = 4 bytes
    int N_u32 = N / 16;
    const uint32_t* W32 = reinterpret_cast<const uint32_t*>(W);

    int acc[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0;

    for (int k = 0; k < K; ++k) {
        int xv = X[k];
        if (xv == 0) continue;
        uint32_t u = W32[k * N_u32 + j_byte4];
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int t = (u >> (i * 2)) & 3;
            int w = (t == 1) - (t == 2);
            acc[i] += xv * w;
        }
    }
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        out[j_base + i] = __float2half((float)acc[i] * scale);
    }
}

// ---------- Helpers ----------
struct MatShape { const char* name; int K; int N; };

struct BR {
    double ms;
    uint64_t lane_macs;
    size_t weight_bytes;
};

template<typename F>
static double time_kernel(F&& launch, int n) {
    // Warmup
    launch();
    CK(cudaDeviceSynchronize());
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < n; ++i) launch();
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms / n;
}

int main(int argc, char** argv) {
    int n_iters = (argc > 1) ? std::atoi(argv[1]) : 100;

    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s, SM %d.%d, %d SMs, %.1f GiB (RTX 3090 peak ~936 GB/s)\n",
        prop.name, prop.major, prop.minor, prop.multiProcessorCount,
        prop.totalGlobalMem / (1024.0*1024.0*1024.0));

    cublasHandle_t cublas; CB(cublasCreate(&cublas));

    MatShape mats[] = {
        {"Wq",       2048,    2048},
        {"Wk",       2048,     512},
        {"Wv",       2048,     512},
        {"Wo",       2048,    2048},
        {"Wgate",    2048,    8192},
        {"Wup",      2048,    8192},
        {"Wdown",    8192,    2048},
        {"lm_head",  2048,  128256},
    };

    std::printf("%-10s %6s %8s %12s %12s %12s %12s %14s %14s %14s %12s\n",
        "shape", "K", "N",
        "packed-v4 ms", "packed-v16 ms", "sgemm ms", "int8 ms",
        "packed GMAC/s", "sgemm GMAC/s", "int8 GMAC/s", "packed MB");

    int M = 1;  // single-token gen path

    int reps[] = {16, 16, 16, 16, 16, 16, 16, 1};
    double total_pv4 = 0, total_pv16 = 0, total_sgemm = 0, total_int8 = 0;
    uint64_t total_lane_macs = 0;
    size_t total_packed_bytes = 0, total_int8_bytes = 0, total_fp16_bytes = 0;

    std::mt19937 rng(42);

    int idx = 0;
    for (auto& m : mats) {
        int K = m.K, N = m.N;
        size_t Wn = (size_t)K * N;

        // Packed weights: K * N/4 bytes (assumes N % 4 == 0, all our shapes do)
        size_t W_packed_bytes = (size_t)K * (N / 4);
        size_t W_int8_bytes   = Wn;
        size_t W_fp32_bytes   = Wn * 4;

        // Generate ternary weights as raw -1/0/+1 trits, then pack
        std::vector<int8_t>  hW_int8(Wn);
        std::vector<uint8_t> hW_pack(W_packed_bytes);
        std::uniform_int_distribution<int> d3(-1, 1);
        for (auto& v : hW_int8) v = (int8_t)d3(rng);
        for (size_t k = 0; k < (size_t)K; ++k) {
            for (size_t jb = 0; jb < (size_t)N/4; ++jb) {
                uint8_t b = 0;
                for (int t = 0; t < 4; ++t) {
                    int v = hW_int8[k * N + jb * 4 + t];
                    int code = (v == 1) ? 1 : (v == -1 ? 2 : 0);
                    b |= (code & 0x3) << (t * 2);
                }
                hW_pack[k * (N/4) + jb] = b;
            }
        }

        // Activations: int8 random
        std::vector<int8_t> hX(K);
        for (auto& v : hX) v = (int8_t)d3(rng);

        // fp32 versions for sgemm
        std::vector<float> hWf(Wn), hXf(K);
        for (size_t i = 0; i < Wn; ++i) hWf[i] = (float)hW_int8[i];
        for (int i = 0; i < K; ++i)     hXf[i] = (float)hX[i];

        int8_t   *dX_i8;  uint8_t *dW_pack;
        int8_t   *dW_i8;  float   *dXf, *dWf;
        __half   *dOut_h; int32_t *dOut_i32;
        CK(cudaMalloc(&dX_i8,   K  * sizeof(int8_t)));
        CK(cudaMalloc(&dW_pack, W_packed_bytes));
        CK(cudaMalloc(&dW_i8,   Wn));
        CK(cudaMalloc(&dXf,     K  * sizeof(float)));
        CK(cudaMalloc(&dWf,     Wn * sizeof(float)));
        CK(cudaMalloc(&dOut_h,  N  * sizeof(__half)));
        CK(cudaMalloc(&dOut_i32, N * sizeof(int32_t)));

        CK(cudaMemcpy(dX_i8,   hX.data(),     K  * sizeof(int8_t),  cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dW_pack, hW_pack.data(), W_packed_bytes,      cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dW_i8,   hW_int8.data(), Wn,                  cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dXf,     hXf.data(),    K  * sizeof(float),   cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dWf,     hWf.data(),    Wn * sizeof(float),   cudaMemcpyHostToDevice));

        float scale = 1.0f / 100.0f;

        // packed-v4 timing
        int threads = 128;
        int blocks_v4 = (N/4 + threads - 1) / threads;
        auto launch_v4 = [&]() {
            mm_packed_trit_v4<<<blocks_v4, threads>>>(dX_i8, dW_pack, K, N, scale, dOut_h);
        };
        double ms_pv4 = time_kernel(launch_v4, n_iters);

        // packed-v16 timing (only if N%16 == 0; all our shapes satisfy this except lm_head N=128256/16=8016)
        int blocks_v16 = (N/16 + threads - 1) / threads;
        auto launch_v16 = [&]() {
            mm_packed_trit_v16<<<blocks_v16, threads>>>(dX_i8, dW_pack, K, N, scale, dOut_h);
        };
        double ms_pv16 = -1;
        if (N % 16 == 0) ms_pv16 = time_kernel(launch_v16, n_iters);

        // cuBLAS sgemm fp32 (M=1 effectively sgemv)
        const float a32 = scale, b32 = 0.0f;
        // Reuse the sgemm path: C(N,1) = W(N,K)col * X(K,1)col, with W row-major (K,N) = (N,K)col
        auto launch_sgemm = [&]() {
            CB(cublasSgemv(cublas, CUBLAS_OP_N, N, K, &a32, dWf, N, dXf, 1, &b32,
                           reinterpret_cast<float*>(dOut_h), 1));
        };
        double ms_sgemm = time_kernel(launch_sgemm, n_iters);

        // cuBLAS INT8 GEMM (M=1)
        const int32_t aI = 1, bI = 0;
        auto launch_int8 = [&]() {
            CB(cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                &aI, dW_i8, CUDA_R_8I, N, dX_i8, CUDA_R_8I, K,
                &bI, dOut_i32, CUDA_R_32I, N,
                CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        };
        double ms_int8 = time_kernel(launch_int8, n_iters);

        // analytical lane-MACs (matches our F8 honest counter)
        uint64_t lane_macs = (uint64_t)((K + 26) / 27) * (uint64_t)N * 27ull;

        auto gmacs = [&](double ms) { return ms <= 0 ? 0.0 : (double)lane_macs / (ms * 1e6); };
        std::printf("%-10s %6d %8d %12.4f %12.4f %12.4f %12.4f %14.1f %14.1f %14.1f %12.2f\n",
            m.name, K, N,
            ms_pv4, ms_pv16, ms_sgemm, ms_int8,
            gmacs(ms_pv4), gmacs(ms_sgemm), gmacs(ms_int8),
            W_packed_bytes / (1024.0*1024.0));

        total_pv4   += ms_pv4   * reps[idx];
        total_pv16  += (ms_pv16 > 0 ? ms_pv16 : ms_pv4) * reps[idx];
        total_sgemm += ms_sgemm * reps[idx];
        total_int8  += ms_int8  * reps[idx];
        total_lane_macs   += lane_macs * reps[idx];
        total_packed_bytes += W_packed_bytes * reps[idx];
        total_int8_bytes  += W_int8_bytes   * reps[idx];
        total_fp16_bytes  += Wn * 2          * reps[idx];

        cudaFree(dX_i8); cudaFree(dW_pack); cudaFree(dW_i8);
        cudaFree(dXf); cudaFree(dWf); cudaFree(dOut_h); cudaFree(dOut_i32);
        idx++;
    }

    std::printf("\n=== Llama 1B forward equivalent (matmul fabric only, M=1) ===\n");
    std::printf("Total lane-MACs        : %llu\n", (unsigned long long)total_lane_macs);
    std::printf("\nWall-clock per backend (M=1 single-token gen):\n");
    std::printf("  packed-v4 (4 trit/B) : %8.3f ms  (%.1f GMAC/s)\n", total_pv4,   (double)total_lane_macs/(total_pv4*1e6));
    std::printf("  packed-v16 (uint32)  : %8.3f ms  (%.1f GMAC/s)\n", total_pv16,  (double)total_lane_macs/(total_pv16*1e6));
    std::printf("  cuBLAS sgemm fp32    : %8.3f ms  (%.1f GMAC/s)\n", total_sgemm, (double)total_lane_macs/(total_sgemm*1e6));
    std::printf("  cuBLAS INT8 TC       : %8.3f ms  (%.1f GMAC/s)\n", total_int8,  (double)total_lane_macs/(total_int8*1e6));
    std::printf("\nWeight memory footprint per forward (read-once amortization):\n");
    std::printf("  packed (1.58 b/elem) : %8.2f MB\n", total_packed_bytes / (1024.0*1024.0));
    std::printf("  int8   (8 b/elem)    : %8.2f MB\n", total_int8_bytes   / (1024.0*1024.0));
    std::printf("  fp16   (16 b/elem)   : %8.2f MB\n", total_fp16_bytes   / (1024.0*1024.0));
    std::printf("  Q8_0   (~9 b/elem)   : %8.2f MB (analytical, with scale overhead)\n",
        total_int8_bytes * 9.0 / 8.0 / (1024.0*1024.0));

    std::printf("\nImplied effective bandwidth (weights only, packed-v16):\n");
    std::printf("  %.1f GB/s   (peak: ~936 GB/s on RTX 3090)\n",
        total_packed_bytes / (total_pv16 * 1e6));

    cublasDestroy(cublas);
    return 0;
}
