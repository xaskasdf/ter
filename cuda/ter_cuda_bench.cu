// ter_cuda_bench.cu -- standalone microbench of the ter sim's mm_row on CUDA.
//
// Mirrors the AVX2 GEMV in src/tx/forward.cpp: int32 trit payload (Format B
// 9-trit, range [-9841..+9841]) for both X (1, K) and W (K, N) row-major,
// k-outer/j-inner with X-broadcast and zero-skip. Accumulates int64 then
// multiplies by per-tensor scale.
//
// Two backends:
//   1) naive: custom kernel mirroring the AVX2 algorithm exactly
//   2) cublas: convert payloads to fp32, run sgemm (upper-bound throughput)
//
// Drives Llama 3.2 1B forward shapes (Q/K/V/O/gate/up/down per layer + lm_head).
//
// Build:
//   nvcc -O3 -arch=sm_86 ter_cuda_bench.cu -lcublas -o ter_cuda_bench

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <vector>
#include <random>
#include <string>

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        std::exit(1); \
    } \
} while (0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t s = (call); \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        std::fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)s); \
        std::exit(1); \
    } \
} while (0)

// ---------------- Naive mirror kernel ----------------
// j-parallel: each thread computes one output column accumulating across K.
__global__ void mm_row_kernel_naive(
    const int32_t* __restrict__ X,
    const int32_t* __restrict__ W,
    int K, int N,
    float scale,
    float* __restrict__ out)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    long long acc = 0;
    for (int k = 0; k < K; ++k) {
        int xv = X[k];
        if (xv == 0) continue;
        acc += (long long)xv * (long long)W[(long long)k * N + j];
    }
    out[j] = (float)acc * scale;
}

// ---------------- Bench harness ----------------
struct MatShape { const char* name; int K; int N; int reps_per_forward; };

struct BenchResult {
    double ms_per_call_naive  = 0.0;
    double ms_per_call_cublas = 0.0;
    uint64_t lane_macs_per_call = 0;   // ceil(K/27) * N * 27
};

BenchResult bench_shape(const MatShape& m, cublasHandle_t cublas, int n_iters)
{
    int K = m.K, N = m.N;
    size_t Wn = (size_t)K * (size_t)N;

    // Allocate + fill random {-1, 0, +1} amplified to [-9841..+9841]
    std::mt19937 rng(42 + K * 7919 + N);
    std::uniform_int_distribution<int> dist(-9841, 9841);

    std::vector<int32_t> hX(K), hW(Wn);
    for (auto& v : hX) v = dist(rng);
    for (auto& v : hW) v = dist(rng);

    int32_t *dX, *dW;
    float   *dOut, *dXf, *dWf;
    CUDA_CHECK(cudaMalloc(&dX,   K   * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&dW,   Wn  * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&dOut, N   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dXf,  K   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dWf,  Wn  * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dX, hX.data(), K  * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW, hW.data(), Wn * sizeof(int32_t), cudaMemcpyHostToDevice));

    // Pre-convert int32 to fp32 for cuBLAS path (one-shot, cost not in timing).
    std::vector<float> hXf(K), hWf(Wn);
    for (int i = 0; i < K; ++i) hXf[i] = (float)hX[i];
    for (size_t i = 0; i < Wn; ++i) hWf[i] = (float)hW[i];
    CUDA_CHECK(cudaMemcpy(dXf, hXf.data(), K  * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dWf, hWf.data(), Wn * sizeof(float), cudaMemcpyHostToDevice));

    float scale = 1.0f / (9841.0f * 9841.0f);
    BenchResult br;
    br.lane_macs_per_call = (uint64_t)((K + 26) / 27) * (uint64_t)N * 27ull;

    // Warmup
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    mm_row_kernel_naive<<<blocks, threads>>>(dX, dW, K, N, scale, dOut);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Naive timing ----
    {
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);
        for (int i = 0; i < n_iters; ++i)
            mm_row_kernel_naive<<<blocks, threads>>>(dX, dW, K, N, scale, dOut);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms = 0;
        cudaEventElapsedTime(&ms, t0, t1);
        br.ms_per_call_naive = ms / n_iters;
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }

    // ---- cuBLAS sgemm timing (vector-matrix product) ----
    // X (1, K) * W (K, N) -> Y (1, N).  cuBLAS is column-major; treat W^T as (N, K).
    {
        const float alpha = scale, beta = 0.0f;
        // sgemv: y = alpha * A^T * x + beta * y, where A is (K, N) col-major.
        // We have W row-major (K, N) which equals (N, K) col-major; sgemv with
        // op=N over (N, K)col gives length-N result from length-K x. Perfect.
        // Warmup
        CUBLAS_CHECK(cublasSgemv(cublas, CUBLAS_OP_N, N, K, &alpha,
                                 dWf, N, dXf, 1, &beta, dOut, 1));
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);
        for (int i = 0; i < n_iters; ++i)
            CUBLAS_CHECK(cublasSgemv(cublas, CUBLAS_OP_N, N, K, &alpha,
                                     dWf, N, dXf, 1, &beta, dOut, 1));
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms = 0;
        cudaEventElapsedTime(&ms, t0, t1);
        br.ms_per_call_cublas = ms / n_iters;
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }

    cudaFree(dX); cudaFree(dW); cudaFree(dOut); cudaFree(dXf); cudaFree(dWf);
    return br;
}

int main(int argc, char** argv)
{
    int n_iters = (argc > 1) ? std::atoi(argv[1]) : 100;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s, SM %d.%d, %d SMs, %.1f GiB\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));

    // Llama 3.2 1B Q8_0 forward shapes (16 layers; counts already include layer factor)
    MatShape mats[] = {
        {"Wq",      2048, 2048,  16},
        {"Wk",      2048,  512,  16},
        {"Wv",      2048,  512,  16},
        {"Wo",      2048, 2048,  16},
        {"Wgate",   2048, 8192,  16},
        {"Wup",     2048, 8192,  16},
        {"Wdown",   8192, 2048,  16},
        {"lm_head", 2048, 128256, 1},
    };

    std::printf("%-10s %6s %8s %5s %12s %12s %14s %14s\n",
                "shape", "K", "N", "reps", "naive ms/c", "cublas ms/c",
                "naive GMAC/s", "cublas GMAC/s");

    double total_naive_ms = 0, total_cublas_ms = 0;
    uint64_t total_lane_macs = 0;
    for (auto& m : mats) {
        BenchResult r = bench_shape(m, cublas, n_iters);
        double naive_gmacs  = (double)r.lane_macs_per_call / (r.ms_per_call_naive  * 1e6);
        double cublas_gmacs = (double)r.lane_macs_per_call / (r.ms_per_call_cublas * 1e6);
        std::printf("%-10s %6d %8d %5d %12.4f %12.4f %14.1f %14.1f\n",
                    m.name, m.K, m.N, m.reps_per_forward,
                    r.ms_per_call_naive, r.ms_per_call_cublas,
                    naive_gmacs, cublas_gmacs);
        total_naive_ms  += r.ms_per_call_naive  * m.reps_per_forward;
        total_cublas_ms += r.ms_per_call_cublas * m.reps_per_forward;
        total_lane_macs += r.lane_macs_per_call * m.reps_per_forward;
    }

    double naive_total_gmacs  = (double)total_lane_macs / (total_naive_ms  * 1e6);
    double cublas_total_gmacs = (double)total_lane_macs / (total_cublas_ms * 1e6);
    std::printf("\n=== Full Llama 1B forward equivalent ===\n");
    std::printf("Total lane-MACs    : %llu\n", (unsigned long long)total_lane_macs);
    std::printf("naive  forward time: %8.3f ms  (%.1f GMAC/s)\n", total_naive_ms,  naive_total_gmacs);
    std::printf("cublas forward time: %8.3f ms  (%.1f GMAC/s)\n", total_cublas_ms, cublas_total_gmacs);
    std::printf("Reference (Mac AVX2 sim): 25600 ms / forward (~38 GMAC/s)\n");

    cublasDestroy(cublas);
    return 0;
}
