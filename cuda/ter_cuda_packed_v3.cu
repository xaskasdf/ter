// ter_cuda_packed_v3.cu -- ternary-optimized packed kernels.
//
// v4_wide: best previous (1 out/thread, 256 threads/block, shared X, scalar K-loop)
// v6_dp4a: same but K-loop processes 4 trits per iteration via __dp4a
//          (CUDA SIMD int8x4 dot product)
// v7_colmaj: W stored column-major (N/4 byte-cols, K rows packed in uint32);
//            each thread reads contiguous K-chunks via uint32 loads + __dp4a
//
// __dp4a does 4 int8 MACs in one instruction; should give ~4x over scalar
// when not memory-bound. Layout v7 also collapses W reads to 1 uint32 per
// 16 K-elements (vs 16 separate byte reads in v6 row-major).

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

// v4_wide: scalar K-loop reference
__global__ void mm_v4_wide(
    const int8_t* X, const uint8_t* W, int K, int N, float scale, __half* out)
{
    constexpr int K_TILE = 512;
    __shared__ int8_t X_smem[K_TILE];
    int tid = threadIdx.x;
    int j = blockIdx.x * blockDim.x + tid;
    if (j >= N) return;
    int N_bytes = N / 4, j_byte = j / 4, sub_bit = (j & 3) * 2;
    int acc = 0;
    for (int k0 = 0; k0 < K; k0 += K_TILE) {
        for (int i = tid; i < K_TILE && k0 + i < K; i += blockDim.x) X_smem[i] = X[k0 + i];
        __syncthreads();
        int kmax = min(K_TILE, K - k0);
        for (int kk = 0; kk < kmax; ++kk) {
            int xv = X_smem[kk]; if (xv == 0) continue;
            uint8_t b = W[(k0 + kk) * N_bytes + j_byte];
            int t = (b >> sub_bit) & 3;
            acc += xv * ((t == 1) - (t == 2));
        }
        __syncthreads();
    }
    out[j] = __float2half((float)acc * scale);
}

// v6_dp4a: __dp4a inner loop, processes 4 K-trits per call.
// W layout: row-major (K, N/4) bytes (same as v4); 4 K-rows are strided.
__global__ void mm_v6_dp4a(
    const int8_t* X, const uint8_t* W, int K, int N, float scale, __half* out)
{
    constexpr int K_TILE = 512;
    __shared__ int8_t X_smem[K_TILE];
    int tid = threadIdx.x;
    int j = blockIdx.x * blockDim.x + tid;
    if (j >= N) return;
    int N_bytes = N / 4, j_byte = j / 4, sub_bit = (j & 3) * 2;
    int acc = 0;

    for (int k0 = 0; k0 < K; k0 += K_TILE) {
        for (int i = tid; i < K_TILE && k0 + i < K; i += blockDim.x) X_smem[i] = X[k0 + i];
        __syncthreads();
        int kmax = min(K_TILE, K - k0);

        // 4-at-a-time __dp4a path
        int kk = 0;
        for (; kk + 4 <= kmax; kk += 4) {
            uint8_t b0 = W[(k0+kk+0) * N_bytes + j_byte];
            uint8_t b1 = W[(k0+kk+1) * N_bytes + j_byte];
            uint8_t b2 = W[(k0+kk+2) * N_bytes + j_byte];
            uint8_t b3 = W[(k0+kk+3) * N_bytes + j_byte];
            int t0=(b0>>sub_bit)&3, t1=(b1>>sub_bit)&3, t2=(b2>>sub_bit)&3, t3=(b3>>sub_bit)&3;
            int w0=(t0==1)-(t0==2), w1=(t1==1)-(t1==2), w2=(t2==1)-(t2==2), w3=(t3==1)-(t3==2);
            int32_t W4 = (uint8_t)w0 | ((uint8_t)w1 << 8) | ((uint8_t)w2 << 16) | ((uint8_t)w3 << 24);
            int32_t X4 = (uint8_t)X_smem[kk] | ((uint8_t)X_smem[kk+1] << 8)
                       | ((uint8_t)X_smem[kk+2] << 16) | ((uint8_t)X_smem[kk+3] << 24);
            acc = __dp4a(X4, W4, acc);
        }
        // Tail
        for (; kk < kmax; ++kk) {
            int xv = X_smem[kk]; if (xv == 0) continue;
            uint8_t b = W[(k0 + kk) * N_bytes + j_byte];
            int t = (b >> sub_bit) & 3;
            acc += xv * ((t == 1) - (t == 2));
        }
        __syncthreads();
    }
    out[j] = __float2half((float)acc * scale);
}

// v7_colmaj: W stored column-major-by-byte. Layout: (N/4 byte-cols, K rows).
// Each byte still holds 4 trits for output cols [4*j_byte, 4*j_byte+3].
// For our thread (output col j), 16 contiguous K-rows = 16 contiguous bytes
// at col j_byte = 1 uint128 load (or 4 uint32 loads).
//
// We use uint32 loads (4 K-bytes at once); each byte still holds 4 trits.
// So one uint32 = 16 trits = 4 K-rows for our column. Apply __dp4a 4 times.
__global__ void mm_v7_colmaj(
    const int8_t* X, const uint8_t* W_col, int K, int N, float scale, __half* out)
{
    constexpr int K_TILE = 512;
    __shared__ int8_t X_smem[K_TILE];
    int tid = threadIdx.x;
    int j = blockIdx.x * blockDim.x + tid;
    if (j >= N) return;
    int j_byte = j / 4, sub_bit = (j & 3) * 2;
    // Layout: W_col[j_byte * K + k] gives byte at byte-col j_byte, K-row k.
    const uint8_t* W_my_col = W_col + (size_t)j_byte * K;
    int acc = 0;

    for (int k0 = 0; k0 < K; k0 += K_TILE) {
        for (int i = tid; i < K_TILE && k0 + i < K; i += blockDim.x) X_smem[i] = X[k0 + i];
        __syncthreads();
        int kmax = min(K_TILE, K - k0);

        // 4-at-a-time using contiguous reads (4 bytes = 1 uint32 from W_my_col)
        const uint32_t* W32 = reinterpret_cast<const uint32_t*>(W_my_col + k0);
        int kk = 0;
        for (; kk + 4 <= kmax; kk += 4) {
            uint32_t w_u32 = W32[kk / 4];
            uint8_t b0 = (w_u32 >> 0) & 0xff;
            uint8_t b1 = (w_u32 >> 8) & 0xff;
            uint8_t b2 = (w_u32 >> 16) & 0xff;
            uint8_t b3 = (w_u32 >> 24) & 0xff;
            int t0=(b0>>sub_bit)&3, t1=(b1>>sub_bit)&3, t2=(b2>>sub_bit)&3, t3=(b3>>sub_bit)&3;
            int w0=(t0==1)-(t0==2), w1=(t1==1)-(t1==2), w2=(t2==1)-(t2==2), w3=(t3==1)-(t3==2);
            int32_t W4 = (uint8_t)w0 | ((uint8_t)w1 << 8) | ((uint8_t)w2 << 16) | ((uint8_t)w3 << 24);
            int32_t X4 = (uint8_t)X_smem[kk] | ((uint8_t)X_smem[kk+1] << 8)
                       | ((uint8_t)X_smem[kk+2] << 16) | ((uint8_t)X_smem[kk+3] << 24);
            acc = __dp4a(X4, W4, acc);
        }
        for (; kk < kmax; ++kk) {
            int xv = X_smem[kk]; if (xv == 0) continue;
            uint8_t b = W_my_col[k0 + kk];
            int t = (b >> sub_bit) & 3;
            acc += xv * ((t == 1) - (t == 2));
        }
        __syncthreads();
    }
    out[j] = __float2half((float)acc * scale);
}

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
    int n_iters = (argc > 1) ? std::atoi(argv[1]) : 50;

    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s, %d SMs\n\n", prop.name, prop.multiProcessorCount);

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

    std::printf("%-10s %6s %8s %10s %10s %10s %10s %12s %12s %12s %12s\n",
        "shape", "K", "N",
        "v4 ms", "v6dp4a ms", "v7col ms", "INT8 ms",
        "v4 GMAC/s", "v6 GMAC/s", "v7 GMAC/s", "INT8 GMAC/s");

    double tot_v4 = 0, tot_v6 = 0, tot_v7 = 0, tot_int8 = 0;
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

        // Row-major packed (for v4/v6)
        std::vector<uint8_t> hW_row(Wp);
        for (size_t k = 0; k < (size_t)K; ++k)
            for (size_t jb = 0; jb < (size_t)N/4; ++jb) {
                uint8_t b = 0;
                for (int t = 0; t < 4; ++t) {
                    int v = hW_int8[k * N + jb * 4 + t];
                    int code = (v == 1) ? 1 : (v == -1 ? 2 : 0);
                    b |= (code & 0x3) << (t * 2);
                }
                hW_row[k * (N/4) + jb] = b;
            }
        // Column-major byte (for v7): same bytes, transposed by byte-col
        std::vector<uint8_t> hW_col(Wp);
        for (size_t jb = 0; jb < (size_t)N/4; ++jb)
            for (size_t k = 0; k < (size_t)K; ++k)
                hW_col[jb * K + k] = hW_row[k * (N/4) + jb];

        int8_t  *dX, *dW_i8;
        uint8_t *dW_row, *dW_col;
        __half  *dOut;
        int32_t *dOut_i32;
        CK(cudaMalloc(&dX,       K));
        CK(cudaMalloc(&dW_row,   Wp));
        CK(cudaMalloc(&dW_col,   Wp));
        CK(cudaMalloc(&dW_i8,    Wn));
        CK(cudaMalloc(&dOut,     N * sizeof(__half)));
        CK(cudaMalloc(&dOut_i32, N * sizeof(int32_t)));
        CK(cudaMemcpy(dX,     hX.data(),     K,  cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dW_row, hW_row.data(), Wp, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dW_col, hW_col.data(), Wp, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dW_i8,  hW_int8.data(), Wn, cudaMemcpyHostToDevice));

        float scale = 1.0f / 100.0f;
        int t256 = 256;
        int blocks = (N + t256 - 1) / t256;
        auto l_v4 = [&]() { mm_v4_wide   <<<blocks, t256>>>(dX, dW_row, K, N, scale, dOut); };
        auto l_v6 = [&]() { mm_v6_dp4a   <<<blocks, t256>>>(dX, dW_row, K, N, scale, dOut); };
        auto l_v7 = [&]() { mm_v7_colmaj <<<blocks, t256>>>(dX, dW_col, K, N, scale, dOut); };
        const int32_t aI = 1, bI = 0;
        auto l_int8 = [&]() {
            CB(cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, 1, K,
                &aI, dW_i8, CUDA_R_8I, N, dX, CUDA_R_8I, K,
                &bI, dOut_i32, CUDA_R_32I, N,
                CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        };

        double ms_v4 = time_it(l_v4, n_iters);
        double ms_v6 = time_it(l_v6, n_iters);
        double ms_v7 = time_it(l_v7, n_iters);
        double ms_int8 = time_it(l_int8, n_iters);

        uint64_t lane_macs = (uint64_t)((K + 26) / 27) * (uint64_t)N * 27ull;
        auto g = [&](double ms) { return ms <= 0 ? 0.0 : (double)lane_macs / (ms * 1e6); };

        std::printf("%-10s %6d %8d %10.4f %10.4f %10.4f %10.4f %12.1f %12.1f %12.1f %12.1f\n",
            m.name, K, N, ms_v4, ms_v6, ms_v7, ms_int8,
            g(ms_v4), g(ms_v6), g(ms_v7), g(ms_int8));

        tot_v4 += ms_v4 * m.reps;
        tot_v6 += ms_v6 * m.reps;
        tot_v7 += ms_v7 * m.reps;
        tot_int8 += ms_int8 * m.reps;
        tot_lane_macs += lane_macs * m.reps;

        cudaFree(dX); cudaFree(dW_row); cudaFree(dW_col); cudaFree(dW_i8);
        cudaFree(dOut); cudaFree(dOut_i32);
    }

    auto g = [&](double ms) { return (double)tot_lane_macs / (ms * 1e6); };
    std::printf("\n=== Llama 1B forward equivalent (matmul fabric only, M=1) ===\n");
    std::printf("v4 wide (scalar K-loop)   : %8.3f ms / forward (%.1f GMAC/s)\n", tot_v4, g(tot_v4));
    std::printf("v6 dp4a (4 trits/instr)   : %8.3f ms / forward (%.1f GMAC/s)\n", tot_v6, g(tot_v6));
    std::printf("v7 colmaj (uint32 + dp4a) : %8.3f ms / forward (%.1f GMAC/s)\n", tot_v7, g(tot_v7));
    std::printf("cuBLAS INT8 TC (reference): %8.3f ms / forward (%.1f GMAC/s)\n", tot_int8, g(tot_int8));
    std::printf("\nBest packed / INT8 TC ratio: %.2fx\n",
        std::min({tot_v4, tot_v6, tot_v7}) / tot_int8);

    cublasDestroy(cublas);
    return 0;
}
