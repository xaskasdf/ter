// ter_cuda_packed_batched.cu -- v11 warp4 extended to M >= 1 (batched).
//
// Single-token gen (M=1) hides our 16:1 byte->MAC density advantage because
// INT8 TC isn't saturated either. Production uses M >= 64 (prefill batch).
// Our density extends to 16M MACs per byte read; INT8 TC's effective
// utilization should saturate but we keep the constant density advantage.
//
// Bench at M in {1, 16, 64, 256, 512} on Llama 1B matmul fabric.

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

// Batched packed kernel: M tokens batched. X is (K, M) col-major in int8.
// W_col is (N/4 byte-cols, K rows) packed bytes.
// Each warp computes 4 output cols across all M tokens.
// Per K-iter: 1 byte (4 trits) + M int8 reads from X = 4M MACs.
__global__ void mm_packed_batched(
    const int8_t* X,       // (K, M) col-major
    const uint8_t* W_col,  // packed col-major byte
    int K, int N, int M,
    float scale,
    __half* out)           // (N, M) col-major
{
    int warp_block_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    int j_base = warp_block_id * 4;
    if (j_base >= N) return;
    int j_byte = j_base / 4;
    const uint8_t* Wc = W_col + (size_t)j_byte * K;
    int kpt = (K + 31) / 32;
    int kstart = lane * kpt, kend = min(K, kstart + kpt);

    // acc[col_sub][token_m]. Cap M at 16 for register pressure.
    constexpr int M_MAX = 16;
    int acc[4][M_MAX];
    #pragma unroll
    for (int sub = 0; sub < 4; ++sub)
        for (int mi = 0; mi < M_MAX; ++mi) acc[sub][mi] = 0;

    for (int kk = kstart; kk < kend; ++kk) {
        uint8_t b = Wc[kk];
        // Decode 4 trits at this K position
        int t[4];
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) t[sub] = (b >> (sub * 2)) & 3;
        // For each token, accumulate
        for (int mi = 0; mi < M; ++mi) {
            int xv = X[mi * K + kk];  // X col-major: token mi at K=kk
            if (xv == 0) continue;
            #pragma unroll
            for (int sub = 0; sub < 4; ++sub) {
                int sign = (t[sub] == 1) - (t[sub] == 2);
                acc[sub][mi] += sign * xv;
            }
        }
    }

    // Warp-shuffle reduce per (col, token)
    #pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
        for (int mi = 0; mi < M; ++mi) {
            int v = acc[sub][mi];
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
            acc[sub][mi] = v;
        }
    }
    if (lane == 0) {
        for (int mi = 0; mi < M; ++mi) {
            #pragma unroll
            for (int sub = 0; sub < 4; ++sub)
                out[mi * N + j_base + sub] = __float2half((float)acc[sub][mi] * scale);
        }
    }
}

// ----------- Bench harness -----------
struct MatShape { const char* name; int K; int N; int reps; };

template<typename F>
static double time_it(F&& launch, int n) {
    launch(); CK(cudaDeviceSynchronize());
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < n; ++i) launch();
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms / n;
}

int main(int argc, char** argv) {
    int n_iters = (argc > 1) ? std::atoi(argv[1]) : 30;
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s, %d SMs (n_iters=%d)\n\n", prop.name, prop.multiProcessorCount, n_iters);
    cublasHandle_t cublas; CB(cublasCreate(&cublas));

    MatShape mats[] = {
        {"Wq",       2048,    2048, 16},
        {"Wk",       2048,     512, 16},
        {"Wv",       2048,     512, 16},
        {"Wo",       2048,    2048, 16},
        {"Wgate",    2048,    8192, 16},
        {"Wup",      2048,    8192, 16},
        {"Wdown",    8192,    2048, 16},
        {"lm_head",  2048,  128256,  1},
    };

    int M_list[] = {1, 16};  // packed kernel capped at M_MAX=16; INT8 TC tested wider separately
    int n_M = sizeof(M_list) / sizeof(M_list[0]);

    std::printf("%-10s %5s %6s %8s %12s %12s %12s %12s\n",
        "shape", "M", "K", "N",
        "packed ms", "INT8 ms", "packed GM/s", "INT8 GM/s");

    std::vector<double> tot_pkd(n_M, 0), tot_int8(n_M, 0);
    std::vector<uint64_t> tot_lane_macs(n_M, 0);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dt(0, 2);
    std::uniform_int_distribution<int> dx(-1, 1);

    int M_max = M_list[n_M - 1];

    for (auto& m : mats) {
        int K = m.K, N = m.N;
        size_t Wn = (size_t)K * N, Wp = (size_t)K * (N / 4);
        std::vector<int8_t> hX(K * M_max), hW_int8(Wn);
        for (auto& v : hX) v = (int8_t)dx(rng);
        std::vector<int> hW_codes(Wn);
        for (auto& c : hW_codes) c = dt(rng);
        for (size_t i = 0; i < Wn; ++i) hW_int8[i] = (hW_codes[i] == 1) ? 1 : (hW_codes[i] == 2 ? -1 : 0);
        std::vector<uint8_t> hW_row(Wp), hW_col(Wp);
        for (size_t k = 0; k < (size_t)K; ++k)
            for (size_t jb = 0; jb < (size_t)N/4; ++jb) {
                uint8_t b = 0;
                for (int t = 0; t < 4; ++t) {
                    int c = hW_codes[k * N + jb * 4 + t];
                    b |= (c & 0x3) << (t * 2);
                }
                hW_row[k * (N/4) + jb] = b;
            }
        for (size_t jb = 0; jb < (size_t)N/4; ++jb)
            for (size_t k = 0; k < (size_t)K; ++k)
                hW_col[jb * K + k] = hW_row[k * (N/4) + jb];

        int8_t *dX, *dW_i8;  uint8_t *dW_col;
        __half *dOut; int32_t *dOut_i32;
        CK(cudaMalloc(&dX, K * M_max));
        CK(cudaMalloc(&dW_col, Wp)); CK(cudaMalloc(&dW_i8, Wn));
        CK(cudaMalloc(&dOut, (size_t)N * M_max * sizeof(__half)));
        CK(cudaMalloc(&dOut_i32, (size_t)N * M_max * sizeof(int32_t)));
        CK(cudaMemcpy(dX, hX.data(), K * M_max, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dW_col, hW_col.data(), Wp, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dW_i8, hW_int8.data(), Wn, cudaMemcpyHostToDevice));

        float scale = 1.0f / 100.0f;

        for (int mi = 0; mi < n_M; ++mi) {
            int M = M_list[mi];
            int t256 = 256;
            int blocks = (N/4 + 8 - 1) / 8;
            auto l_pkd = [&]() { mm_packed_batched<<<blocks, t256>>>(dX, dW_col, K, N, M, scale, dOut); };
            const int32_t aI = 1, bI = 0;
            auto l_int8 = [&]() {
                CB(cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                    &aI, dW_i8, CUDA_R_8I, N, dX, CUDA_R_8I, K,
                    &bI, dOut_i32, CUDA_R_32I, N,
                    CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            };

            double ms_pkd = time_it(l_pkd, n_iters);
            double ms_int8 = time_it(l_int8, n_iters);

            uint64_t lane_macs = (uint64_t)((K + 26) / 27) * (uint64_t)N * 27ull * (uint64_t)M;
            auto g = [&](double ms) { return ms <= 0 ? 0.0 : (double)lane_macs / (ms * 1e6); };

            std::printf("%-10s %5d %6d %8d %12.4f %12.4f %12.1f %12.1f\n",
                m.name, M, K, N, ms_pkd, ms_int8, g(ms_pkd), g(ms_int8));

            tot_pkd[mi] += ms_pkd * m.reps;
            tot_int8[mi] += ms_int8 * m.reps;
            tot_lane_macs[mi] += lane_macs * m.reps;
        }
        cudaFree(dX); cudaFree(dW_col); cudaFree(dW_i8); cudaFree(dOut); cudaFree(dOut_i32);
    }

    std::printf("\n=== Llama 1B forward equivalent at each batch size M ===\n");
    std::printf("%5s %15s %15s %15s %15s %12s\n",
        "M", "packed ms/fwd", "INT8 ms/fwd", "packed GMAC/s", "INT8 GMAC/s", "INT8/packed");
    for (int mi = 0; mi < n_M; ++mi) {
        int M = M_list[mi];
        auto g = [&](double ms) { return (double)tot_lane_macs[mi] / (ms * 1e6); };
        std::printf("%5d %15.3f %15.3f %15.1f %15.1f %12.2f\n",
            M, tot_pkd[mi], tot_int8[mi], g(tot_pkd[mi]), g(tot_int8[mi]),
            tot_int8[mi] / tot_pkd[mi]);
    }
    std::printf("\n(>1.0 in last column means packed kernel is FASTER than cuBLAS INT8 TC)\n");

    cublasDestroy(cublas);
    return 0;
}
