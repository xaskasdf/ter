// ter_cuda_bench2.cu -- microbench v2 of the ter sim's mm_row on CUDA, now with
// INT8 tensor cores via cublasGemmEx. Demonstrates ternary substrate running
// at production silicon speed on Ampere INT8 path (~285 TOPS peak).
//
// Three backends per shape, at M = {1, 16, 64} batch sizes:
//   1) naive: custom int32 kernel, mirrors AVX2 mm_row exactly
//   2) sgemm fp32: cublasGemmEx CUDA_R_32F -> 32F (no tensor cores)
//   3) INT8 tensor cores: cublasGemmEx CUDA_R_8I in / CUDA_R_32I acc (TC path)
//
// The INT8 path is the H3 "substrate-data alignment" story: trit values
// (BitNet) or 9-trit values (Format B clipped) fit naturally in int8, and
// the Ampere INT8 tensor cores deliver ~285 TOPS peak (8x the fp32 peak).

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
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

// Naive mm_row kernel, j-parallel. M=1 case only (matches single-token gen).
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

struct MatShape { const char* name; int K; int N; int reps_per_forward; };

static double bench_naive(const int32_t* dX, const int32_t* dW, float scale,
                          float* dOut, int K, int N, int n_iters)
{
    int threads = 256, blocks = (N + threads - 1) / threads;
    // warmup
    mm_row_kernel_naive<<<blocks, threads>>>(dX, dW, K, N, scale, dOut);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < n_iters; ++i)
        mm_row_kernel_naive<<<blocks, threads>>>(dX, dW, K, N, scale, dOut);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms / n_iters;
}

// cublasGemmEx FP32 GEMM, M tokens batched. C(N,M) = W(N,K)col * X(K,M)col
static double bench_sgemm(cublasHandle_t cublas,
                          const float* dWf, const float* dXf, float* dOut,
                          int M, int K, int N, float scale, int n_iters)
{
    const float alpha = scale, beta = 0.0f;
    auto call = [&]() {
        return cublasGemmEx(cublas,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            dWf, CUDA_R_32F, N,
            dXf, CUDA_R_32F, K,
            &beta,
            dOut, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    };
    CUBLAS_CHECK(call());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < n_iters; ++i) CUBLAS_CHECK(call());
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms / n_iters;
}

// cublasGemmEx INT8 GEMM with INT32 accumulator -- tensor core path on Ampere.
// alpha/beta are INT32 here. After GEMM we'd multiply by per-tensor scale on
// host or in a fused kernel; we skip that for the throughput measurement.
static double bench_int8(cublasHandle_t cublas,
                         const int8_t* dWi, const int8_t* dXi, int32_t* dOuti,
                         int M, int K, int N, int n_iters)
{
    const int32_t alpha = 1, beta = 0;
    auto call = [&]() {
        return cublasGemmEx(cublas,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            dWi, CUDA_R_8I, N,
            dXi, CUDA_R_8I, K,
            &beta,
            dOuti, CUDA_R_32I, N,
            CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    };
    cublasStatus_t s = call();
    if (s != CUBLAS_STATUS_SUCCESS) return -1.0;
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < n_iters; ++i) CUBLAS_CHECK(call());
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms / n_iters;
}

int main(int argc, char** argv)
{
    int n_iters = (argc > 1) ? std::atoi(argv[1]) : 100;
    int M_max   = 64;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s, SM %d.%d, %d SMs, %.1f GiB\n\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));

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

    // Single allocation pool, sized to the largest shape and max M.
    int K_max = 8192, N_max = 128256;
    size_t Wmax = (size_t)K_max * N_max;
    int32_t *dX, *dW;       float   *dXf, *dWf, *dOutF;
    int8_t  *dXi, *dWi;     int32_t *dOutI;
    CUDA_CHECK(cudaMalloc(&dX,    K_max * M_max * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&dW,    Wmax           * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&dXf,   K_max * M_max * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dWf,   Wmax           * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOutF, N_max * M_max * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dXi,   K_max * M_max * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&dWi,   Wmax           * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&dOutI, N_max * M_max * sizeof(int32_t)));

    int M_list[] = {1, 16, 64};
    int n_M = sizeof(M_list) / sizeof(M_list[0]);

    std::printf("%-10s %5s %6s %8s %12s %12s %12s %12s %12s %12s\n",
                "shape", "M", "K", "N",
                "naive ms/c", "sgemm ms/c", "int8 ms/c",
                "naive GMAC/s", "sgemm GMAC/s", "int8 GMAC/s");

    // Per-M totals to summarize a "1 forward" cost
    std::vector<double> total_naive_ms(n_M, 0), total_sgemm_ms(n_M, 0), total_int8_ms(n_M, 0);
    std::vector<uint64_t> total_lane_macs(n_M, 0);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist9(-9841, 9841);
    std::uniform_int_distribution<int> dist1(-1, 1);     // BitNet-style

    for (auto& m : mats) {
        int K = m.K, N = m.N;
        size_t Wn = (size_t)K * N;

        // Fill data each shape
        std::vector<int32_t> hX(K * M_max), hW(Wn);
        std::vector<int8_t>  hXi(K * M_max), hWi(Wn);
        for (auto& v : hX) v = dist9(rng);
        for (auto& v : hW) v = dist9(rng);
        // INT8 versions: BitNet-style ternary {-1,0,+1} for both X and W
        // (this is the H3 alignment scenario; activations also quantized).
        for (auto& v : hXi) v = (int8_t)dist1(rng);
        for (auto& v : hWi) v = (int8_t)dist1(rng);

        std::vector<float> hXf(K * M_max), hWf(Wn);
        for (size_t i = 0; i < hX.size(); ++i) hXf[i] = (float)hX[i];
        for (size_t i = 0; i < hW.size(); ++i) hWf[i] = (float)hW[i];

        CUDA_CHECK(cudaMemcpy(dX,  hX.data(),  hX.size()  * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dW,  hW.data(),  hW.size()  * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dXf, hXf.data(), hXf.size() * sizeof(float),   cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dWf, hWf.data(), hWf.size() * sizeof(float),   cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dXi, hXi.data(), hXi.size() * sizeof(int8_t),  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dWi, hWi.data(), hWi.size() * sizeof(int8_t),  cudaMemcpyHostToDevice));

        float scale = 1.0f / (9841.0f * 9841.0f);

        for (int mi = 0; mi < n_M; ++mi) {
            int M = M_list[mi];
            uint64_t lane_macs = (uint64_t)((K + 26) / 27) * (uint64_t)N * 27ull * (uint64_t)M;

            double naive_ms = (M == 1) ? bench_naive(dX, dW, scale, dOutF, K, N, n_iters) : -1.0;
            double sgemm_ms = bench_sgemm(cublas, dWf, dXf, dOutF, M, K, N, scale, n_iters);
            double int8_ms  = bench_int8 (cublas, dWi, dXi, dOutI, M, K, N, n_iters);

            auto gmacs = [&](double ms) {
                return (ms <= 0) ? 0.0 : (double)lane_macs / (ms * 1e6);
            };

            std::printf("%-10s %5d %6d %8d %12.4f %12.4f %12.4f %12.1f %12.1f %12.1f\n",
                        m.name, M, K, N,
                        naive_ms, sgemm_ms, int8_ms,
                        gmacs(naive_ms), gmacs(sgemm_ms), gmacs(int8_ms));

            if (naive_ms > 0) total_naive_ms[mi] += naive_ms * m.reps_per_forward;
            total_sgemm_ms[mi] += sgemm_ms * m.reps_per_forward;
            total_int8_ms [mi] += int8_ms  * m.reps_per_forward;
            total_lane_macs[mi] += lane_macs * m.reps_per_forward;
        }
    }

    std::printf("\n=== Llama 1B forward equivalent at each M (matmul fabric only) ===\n");
    std::printf("%5s %14s %14s %14s %14s %14s %14s\n",
                "M", "naive ms/fwd", "sgemm ms/fwd", "int8 ms/fwd",
                "naive GMAC/s", "sgemm GMAC/s", "int8 GMAC/s");
    for (int mi = 0; mi < n_M; ++mi) {
        int M = M_list[mi];
        auto gmacs = [&](double ms) {
            return (ms <= 0) ? 0.0 : (double)total_lane_macs[mi] / (ms * 1e6);
        };
        std::printf("%5d %14.3f %14.3f %14.3f %14.1f %14.1f %14.1f\n",
                    M, total_naive_ms[mi], total_sgemm_ms[mi], total_int8_ms[mi],
                    gmacs(total_naive_ms[mi]), gmacs(total_sgemm_ms[mi]), gmacs(total_int8_ms[mi]));
    }
    std::printf("\nReference (Mac AVX2 sim, M=1): 25600 ms / forward (~38 GMAC/s)\n");
    std::printf("RTX 3090 INT8 tensor core peak: ~285 TOPS = 285000 GMAC/s\n");

    cublasDestroy(cublas);
    return 0;
}
