// ter_cuda_addonly_batched.cu -- BitNet ADD-only kernel batched over M tokens.
//
// Generalizes mm_bitnet_addonly to handle M > 1 inputs. Same packed-trit
// W_col layout, same TVMAC=0 architectural property (no __dp4a, just
// conditional add/sub), now with batch dim.
//
// Layout:
//   X[M, K] row-major int8 (M tokens × K features each)
//   X_scale[M] per-row activation scale
//   W_col[N/4 × K] col-major packed-trit (4 trits per byte)
//   out[M, N] fp16
//
// Each warp computes 4 output columns × M tokens. Shfl-xor reduces across
// K-coop lanes. Register pressure: 4×M ints/thread; for M=16 = 64 ints
// (fits within 256-reg limit on Ampere).

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

// M-batched ADD-only matmul. Each warp processes 4 output cols across M tokens.
// Templated on M for compile-time register allocation (acc[4][M]).
template<int M_FIXED>
__global__ void mm_bitnet_addonly_batched_k(
    const int8_t* __restrict__ X,        // [M, K] int8
    const float* __restrict__ X_scale,   // [M] per-row scale
    const uint8_t* __restrict__ W_col,
    int K, int N,
    float w_scale,
    __half* __restrict__ out)            // [M, N] fp16
{
    int warp_block_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    int j_base = warp_block_id * 4;
    if (j_base >= N) return;
    int j_byte = j_base / 4;
    const uint8_t* Wc = W_col + (size_t)j_byte * K;

    int kpt = (K + 31) / 32;
    int kstart = lane * kpt, kend = min(K, kstart + kpt);

    // 4 output cols × M tokens accumulators
    int acc[4][M_FIXED];
    #pragma unroll
    for (int sub = 0; sub < 4; ++sub)
        #pragma unroll
        for (int m = 0; m < M_FIXED; ++m) acc[sub][m] = 0;

    int kk = kstart;
    for (; kk + 4 <= kend; kk += 4) {
        // Load 4 W bytes (one per K position kk..kk+3)
        uint32_t w = *reinterpret_cast<const uint32_t*>(Wc + kk);
        uint8_t b0 = w & 0xff, b1 = (w>>8)&0xff, b2 = (w>>16)&0xff, b3 = (w>>24)&0xff;

        // Decode 4 codes per byte for each of the 4 output cols
        // For each sub, get the 4-K codes
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            int sb = sub * 2;
            int t0=(b0>>sb)&3, t1=(b1>>sb)&3, t2=(b2>>sb)&3, t3=(b3>>sb)&3;
            // For each of M tokens, accumulate the 4 K-element contributions
            #pragma unroll
            for (int m = 0; m < M_FIXED; ++m) {
                int32_t X4 = *reinterpret_cast<const uint32_t*>(X + (size_t)m * K + kk);
                int x0 = (int8_t)(X4 & 0xff);
                int x1 = (int8_t)((X4 >> 8) & 0xff);
                int x2 = (int8_t)((X4 >> 16) & 0xff);
                int x3 = (int8_t)((X4 >> 24) & 0xff);
                if (t0 == 1)      acc[sub][m] += x0;
                else if (t0 == 2) acc[sub][m] -= x0;
                if (t1 == 1)      acc[sub][m] += x1;
                else if (t1 == 2) acc[sub][m] -= x1;
                if (t2 == 1)      acc[sub][m] += x2;
                else if (t2 == 2) acc[sub][m] -= x2;
                if (t3 == 1)      acc[sub][m] += x3;
                else if (t3 == 2) acc[sub][m] -= x3;
            }
        }
    }
    // Tail
    for (; kk < kend; ++kk) {
        uint8_t b = Wc[kk];
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            int t = (b >> (sub * 2)) & 3;
            if (t == 0) continue;
            #pragma unroll
            for (int m = 0; m < M_FIXED; ++m) {
                int xv = X[(size_t)m * K + kk];
                if (xv == 0) continue;
                if (t == 1)      acc[sub][m] += xv;
                else /* t==2 */  acc[sub][m] -= xv;
            }
        }
    }
    // Reduce across 32 K-coop lanes for each (sub, m)
    #pragma unroll
    for (int sub = 0; sub < 4; ++sub)
        #pragma unroll
        for (int m = 0; m < M_FIXED; ++m)
            for (int o = 16; o > 0; o >>= 1)
                acc[sub][m] += __shfl_xor_sync(0xffffffff, acc[sub][m], o);

    // Write outputs
    if (lane == 0) {
        #pragma unroll
        for (int m = 0; m < M_FIXED; ++m) {
            float scale = X_scale[m] * w_scale;
            #pragma unroll
            for (int sub = 0; sub < 4; ++sub)
                if (j_base + sub < N)
                    out[(size_t)m * N + j_base + sub] = __float2half((float)acc[sub][m] * scale);
        }
    }
}

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

struct Shape { const char* name; int K; int N; };

int main(int argc, char** argv) {
    int n_iters = (argc > 1) ? std::atoi(argv[1]) : 100;
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s, %d SMs (n_iters=%d)\n", prop.name, prop.multiProcessorCount, n_iters);
    cublasHandle_t cublas; CB(cublasCreate(&cublas));

    // BitNet 2B-4T shapes
    Shape shapes[] = {
        {"Wq",     2560,  2560},
        {"Wk",     2560,   640},
        {"Wv",     2560,   640},
        {"Wo",     2560,  2560},
        {"Wgate",  2560,  6912},
        {"Wup",    2560,  6912},
        {"Wdown",  6912,  2560},
    };
    int n_shapes = sizeof(shapes) / sizeof(shapes[0]);

    constexpr int M = 16;
    std::printf("\nBatched ADD-only kernel @ M=%d (BitNet 2B-4T shapes):\n", M);
    std::printf("%-8s %5s %6s  %14s  %14s  %12s\n",
        "shape", "K", "N", "ADD-batched ms", "INT8 TC ms", "ratio (TVMAC=0)");

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dx(-127, 127);
    std::uniform_int_distribution<int> dt(0, 2);

    double sum_addb = 0, sum_int8 = 0;

    for (int si = 0; si < n_shapes; ++si) {
        int K = shapes[si].K, N = shapes[si].N;

        std::vector<int8_t> hX((size_t)M*K), hW((size_t)K*N);
        for (auto& v : hX) v = (int8_t)dx(rng);
        std::vector<int> codes((size_t)K*N);
        for (auto& c : codes) c = dt(rng);
        for (size_t z = 0; z < (size_t)K*N; ++z) hW[z] = (codes[z]==1)?1:(codes[z]==2?-1:0);

        // Pack W col-major
        std::vector<uint8_t> hW_col((size_t)K * (N/4), 0);
        for (int jb = 0; jb < N/4; ++jb)
            for (int k = 0; k < K; ++k) {
                uint8_t b = 0;
                for (int t = 0; t < 4; ++t) b |= (codes[k*N + jb*4 + t] & 3) << (t * 2);
                hW_col[(size_t)jb * K + k] = b;
            }

        std::vector<float> hX_scale(M, 1.0f / 100.0f);

        int8_t *dX, *dW;  uint8_t *dW_col;  float *dX_scale;
        __half *dOut;  int32_t *dOut_i32;
        CK(cudaMalloc(&dX, (size_t)M*K));
        CK(cudaMalloc(&dW, (size_t)K*N));
        CK(cudaMalloc(&dW_col, (size_t)K * (N/4)));
        CK(cudaMalloc(&dX_scale, M*sizeof(float)));
        CK(cudaMalloc(&dOut, (size_t)M*N*sizeof(__half)));
        CK(cudaMalloc(&dOut_i32, (size_t)M*N*sizeof(int32_t)));
        CK(cudaMemcpy(dX, hX.data(), (size_t)M*K, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dW, hW.data(), (size_t)K*N, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dW_col, hW_col.data(), (size_t)K*(N/4), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dX_scale, hX_scale.data(), M*sizeof(float), cudaMemcpyHostToDevice));

        int t = 256;
        int blocks = (N/4 + 8 - 1) / 8;
        float w_scale = 1.5f;
        auto l_addb = [&]() {
            mm_bitnet_addonly_batched_k<M><<<blocks, t>>>(dX, dX_scale, dW_col, K, N, w_scale, dOut);
        };

        const int32_t aI = 1, bI = 0;
        auto l_int8 = [&]() {
            CB(cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                &aI, dW, CUDA_R_8I, N, dX, CUDA_R_8I, K,
                &bI, dOut_i32, CUDA_R_32I, N,
                CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        };

        double ms_addb = time_it(l_addb, n_iters);
        double ms_int8 = time_it(l_int8, n_iters);

        std::printf("%-8s %6d %6d  %14.4f  %14.4f  %12.2fx\n",
            shapes[si].name, K, N, ms_addb, ms_int8, ms_int8 / ms_addb);

        sum_addb += ms_addb;
        sum_int8 += ms_int8;

        cudaFree(dX); cudaFree(dW); cudaFree(dW_col); cudaFree(dX_scale);
        cudaFree(dOut); cudaFree(dOut_i32);
    }

    std::printf("\nBitNet 2B-4T matmul fabric @ M=%d (one layer = 7 matmuls):\n", M);
    std::printf("  ADD-only batched: %.3f ms (TVMAC=0)\n", sum_addb);
    std::printf("  cuBLAS INT8 TC  : %.3f ms\n", sum_int8);
    std::printf("  Speedup         : %.2fx\n", sum_int8 / sum_addb);
    std::printf("  Per-token-effective: %.1f tokens/s (matmul fabric / 30 layers)\n",
        (double)M / (sum_addb * 30 / 1000.0));

    cublasDestroy(cublas);
    return 0;
}
