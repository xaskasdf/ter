// ter_cuda_packed_v2.cu -- packed ternary matmul kernels v3+: tiled, larger
// blocks, warp-cooperative. Goal: close the gap from naive packed (1.8 GMAC/s)
// to cuBLAS INT8 TC (39.6 GMAC/s) WITHOUT using tensor cores -- validates
// that the win is architectural, not "thanks to TC".
//
// Build:
//   nvcc -O3 -arch=sm_86 -std=c++17 ter_cuda_packed_v2.cu -lcublas -o ter_cuda_packed_v2

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

// v3: shared-memory K-tile, 4 output cols/thread.
__global__ void mm_packed_v3_tiled(
    const int8_t*  __restrict__ X,
    const uint8_t* __restrict__ W,
    int K, int N, float scale, __half* __restrict__ out)
{
    constexpr int K_TILE = 256;
    constexpr int OUTS = 4;
    __shared__ int8_t X_smem[K_TILE];

    int tid = threadIdx.x;
    int j_base = (blockIdx.x * blockDim.x + tid) * OUTS;
    if (j_base >= N) return;
    int j_byte = j_base / 4;
    int N_bytes = N / 4;

    int acc0=0, acc1=0, acc2=0, acc3=0;

    for (int k0 = 0; k0 < K; k0 += K_TILE) {
        // Cooperative load of X tile
        for (int i = tid; i < K_TILE && k0 + i < K; i += blockDim.x)
            X_smem[i] = X[k0 + i];
        __syncthreads();

        int kmax = min(K_TILE, K - k0);
        for (int kk = 0; kk < kmax; ++kk) {
            int xv = X_smem[kk];
            if (xv == 0) continue;
            uint8_t b = W[(k0 + kk) * N_bytes + j_byte];
            int t0 = (b >> 0) & 3, t1 = (b >> 2) & 3, t2 = (b >> 4) & 3, t3 = (b >> 6) & 3;
            acc0 += xv * ((t0 == 1) - (t0 == 2));
            acc1 += xv * ((t1 == 1) - (t1 == 2));
            acc2 += xv * ((t2 == 1) - (t2 == 2));
            acc3 += xv * ((t3 == 1) - (t3 == 2));
        }
        __syncthreads();
    }

    out[j_base + 0] = __float2half((float)acc0 * scale);
    out[j_base + 1] = __float2half((float)acc1 * scale);
    out[j_base + 2] = __float2half((float)acc2 * scale);
    out[j_base + 3] = __float2half((float)acc3 * scale);
}

// v4: 1 output col/thread, big blocks (256 threads), shared X tile, no smem needed for W
__global__ void mm_packed_v4_wide(
    const int8_t*  __restrict__ X,
    const uint8_t* __restrict__ W,
    int K, int N, float scale, __half* __restrict__ out)
{
    constexpr int K_TILE = 512;
    __shared__ int8_t X_smem[K_TILE];

    int tid = threadIdx.x;
    int j = blockIdx.x * blockDim.x + tid;
    if (j >= N) return;
    int N_bytes = N / 4;
    int j_byte = j / 4;
    int sub_bit = (j & 3) * 2;

    int acc = 0;

    for (int k0 = 0; k0 < K; k0 += K_TILE) {
        for (int i = tid; i < K_TILE && k0 + i < K; i += blockDim.x)
            X_smem[i] = X[k0 + i];
        __syncthreads();

        int kmax = min(K_TILE, K - k0);
        for (int kk = 0; kk < kmax; ++kk) {
            int xv = X_smem[kk];
            if (xv == 0) continue;
            uint8_t b = W[(k0 + kk) * N_bytes + j_byte];
            int t = (b >> sub_bit) & 3;
            acc += xv * ((t == 1) - (t == 2));
        }
        __syncthreads();
    }
    out[j] = __float2half((float)acc * scale);
}

// v5: 16 output cols/thread via uint32 loads, shared X tile.
// Combines v16-style fat compute with v3-style cooperative X load.
__global__ void mm_packed_v5_fat(
    const int8_t*  __restrict__ X,
    const uint8_t* __restrict__ W,
    int K, int N, float scale, __half* __restrict__ out)
{
    constexpr int K_TILE = 256;
    constexpr int OUTS = 16;
    __shared__ int8_t X_smem[K_TILE];

    int tid = threadIdx.x;
    int j_base = (blockIdx.x * blockDim.x + tid) * OUTS;
    if (j_base >= N) return;
    int j_u32 = j_base / 16;
    int N_u32 = N / 16;
    const uint32_t* W32 = reinterpret_cast<const uint32_t*>(W);

    int acc[OUTS];
    #pragma unroll
    for (int i = 0; i < OUTS; ++i) acc[i] = 0;

    for (int k0 = 0; k0 < K; k0 += K_TILE) {
        for (int i = tid; i < K_TILE && k0 + i < K; i += blockDim.x)
            X_smem[i] = X[k0 + i];
        __syncthreads();

        int kmax = min(K_TILE, K - k0);
        for (int kk = 0; kk < kmax; ++kk) {
            int xv = X_smem[kk];
            if (xv == 0) continue;
            uint32_t u = W32[(k0 + kk) * N_u32 + j_u32];
            #pragma unroll
            for (int i = 0; i < OUTS; ++i) {
                int t = (u >> (i * 2)) & 3;
                acc[i] += xv * ((t == 1) - (t == 2));
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < OUTS; ++i)
        out[j_base + i] = __float2half((float)acc[i] * scale);
}

// ---- Bench harness ----
struct MatShape { const char* name; int K; int N; int reps; };

template<typename F>
static double time_it(F&& launch, int n) {
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
    int n_iters = (argc > 1) ? std::atoi(argv[1]) : 50;

    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s, %d SMs (RTX 3090 peak ~936 GB/s, INT8 TC ~285 TOPS)\n\n",
        prop.name, prop.multiProcessorCount);

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

    std::printf("%-10s %6s %8s %10s %10s %10s %10s %10s %10s %10s %10s %10s\n",
        "shape", "K", "N",
        "v3 ms", "v4 ms", "v5 ms", "INT8 ms",
        "v3 GMAC/s", "v4 GMAC/s", "v5 GMAC/s", "INT8 GMAC/s", "v5/INT8");

    double tot_v3 = 0, tot_v4 = 0, tot_v5 = 0, tot_int8 = 0;
    uint64_t tot_lane_macs = 0;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> d3(-1, 1);

    for (auto& m : mats) {
        int K = m.K, N = m.N;
        size_t Wn = (size_t)K * N;
        size_t Wp = (size_t)K * (N / 4);

        std::vector<int8_t> hX(K), hW_int8(Wn);
        for (auto& v : hX)      v = (int8_t)d3(rng);
        for (auto& v : hW_int8) v = (int8_t)d3(rng);
        std::vector<uint8_t> hW_pack(Wp);
        for (size_t k = 0; k < (size_t)K; ++k)
            for (size_t jb = 0; jb < (size_t)N/4; ++jb) {
                uint8_t b = 0;
                for (int t = 0; t < 4; ++t) {
                    int v = hW_int8[k * N + jb * 4 + t];
                    int code = (v == 1) ? 1 : (v == -1 ? 2 : 0);
                    b |= (code & 0x3) << (t * 2);
                }
                hW_pack[k * (N/4) + jb] = b;
            }

        int8_t  *dX, *dW_i8;  uint8_t *dW;
        __half  *dOut;        int32_t *dOut_i32;
        CK(cudaMalloc(&dX,  K));
        CK(cudaMalloc(&dW,  Wp));
        CK(cudaMalloc(&dW_i8, Wn));
        CK(cudaMalloc(&dOut, N * sizeof(__half)));
        CK(cudaMalloc(&dOut_i32, N * sizeof(int32_t)));
        CK(cudaMemcpy(dX,  hX.data(),     K,  cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dW,  hW_pack.data(), Wp, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dW_i8, hW_int8.data(), Wn, cudaMemcpyHostToDevice));

        float scale = 1.0f / 100.0f;

        int t128 = 128, t256 = 256;
        // v3: 4 outs/thread
        int b_v3 = (N/4 + t128 - 1) / t128;
        auto l_v3 = [&]() { mm_packed_v3_tiled<<<b_v3, t128>>>(dX, dW, K, N, scale, dOut); };
        // v4: 1 out/thread, big block
        int b_v4 = (N + t256 - 1) / t256;
        auto l_v4 = [&]() { mm_packed_v4_wide<<<b_v4, t256>>>(dX, dW, K, N, scale, dOut); };
        // v5: 16 outs/thread, only if N divisible by 16
        int b_v5 = (N/16 + t128 - 1) / t128;
        auto l_v5 = [&]() { mm_packed_v5_fat<<<b_v5, t128>>>(dX, dW, K, N, scale, dOut); };

        // INT8 TC
        const int32_t aI = 1, bI = 0;
        auto l_int8 = [&]() {
            CB(cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, 1, K,
                &aI, dW_i8, CUDA_R_8I, N, dX, CUDA_R_8I, K,
                &bI, dOut_i32, CUDA_R_32I, N,
                CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        };

        double ms_v3 = time_it(l_v3, n_iters);
        double ms_v4 = time_it(l_v4, n_iters);
        double ms_v5 = (N % 16 == 0) ? time_it(l_v5, n_iters) : -1.0;
        double ms_int8 = time_it(l_int8, n_iters);

        uint64_t lane_macs = (uint64_t)((K + 26) / 27) * (uint64_t)N * 27ull;
        auto g = [&](double ms) { return ms <= 0 ? 0.0 : (double)lane_macs / (ms * 1e6); };
        double ratio_v5 = (ms_v5 > 0 && ms_int8 > 0) ? (ms_int8 / ms_v5) : 0;

        std::printf("%-10s %6d %8d %10.4f %10.4f %10.4f %10.4f %10.1f %10.1f %10.1f %10.1f %10.2f\n",
            m.name, K, N, ms_v3, ms_v4, ms_v5, ms_int8,
            g(ms_v3), g(ms_v4), g(ms_v5), g(ms_int8), ratio_v5);

        tot_v3 += ms_v3 * m.reps;
        tot_v4 += ms_v4 * m.reps;
        tot_v5 += (ms_v5 > 0 ? ms_v5 : ms_v3) * m.reps;
        tot_int8 += ms_int8 * m.reps;
        tot_lane_macs += lane_macs * m.reps;

        cudaFree(dX); cudaFree(dW); cudaFree(dW_i8);
        cudaFree(dOut); cudaFree(dOut_i32);
    }

    auto g = [&](double ms) { return (double)tot_lane_macs / (ms * 1e6); };
    std::printf("\n=== Llama 1B forward equivalent (matmul fabric only, M=1) ===\n");
    std::printf("packed v3 tiled  : %8.3f ms / forward  (%.1f GMAC/s)\n",  tot_v3,  g(tot_v3));
    std::printf("packed v4 wide   : %8.3f ms / forward  (%.1f GMAC/s)\n",  tot_v4,  g(tot_v4));
    std::printf("packed v5 fat    : %8.3f ms / forward  (%.1f GMAC/s)\n",  tot_v5,  g(tot_v5));
    std::printf("cuBLAS INT8 TC   : %8.3f ms / forward  (%.1f GMAC/s)\n",  tot_int8, g(tot_int8));
    std::printf("Best packed / INT8 TC ratio : %.2fx\n",
        std::min({tot_v3, tot_v4, tot_v5}) / tot_int8);

    cublasDestroy(cublas);
    return 0;
}
