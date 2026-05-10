// ter_cuda_packed_v4.cu -- two more ternary-optimized kernels.
//
// v8_warp: warp-cooperative K-reduction. Each WARP computes one output column.
//   32 threads per warp each handle K/32 K-elements; warp-shuffles reduce
//   partial sums. Best for small N where v7's 1-thread-per-col runs out of
//   parallelism (Wk/Wv N=512 only spawn ~16 warps).
//
// v9_multi: 4 output cols per thread + __dp4a, building on v7. Best for
//   matmuls that aren't memory-bound and benefit from tighter compute fusion.
//
// Both use the col-major byte W layout from v7 (W_col[j_byte * K + k]).

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

// v7 from packed_v3 (kept as reference)
__global__ void mm_v7_colmaj(
    const int8_t* X, const uint8_t* W_col, int K, int N, float scale, __half* out)
{
    constexpr int K_TILE = 512;
    __shared__ int8_t X_smem[K_TILE];
    int tid = threadIdx.x;
    int j = blockIdx.x * blockDim.x + tid;
    if (j >= N) return;
    int j_byte = j / 4, sub_bit = (j & 3) * 2;
    const uint8_t* W_my_col = W_col + (size_t)j_byte * K;
    int acc = 0;
    for (int k0 = 0; k0 < K; k0 += K_TILE) {
        for (int i = tid; i < K_TILE && k0 + i < K; i += blockDim.x) X_smem[i] = X[k0 + i];
        __syncthreads();
        int kmax = min(K_TILE, K - k0);
        const uint32_t* W32 = reinterpret_cast<const uint32_t*>(W_my_col + k0);
        int kk = 0;
        for (; kk + 4 <= kmax; kk += 4) {
            uint32_t w_u32 = W32[kk / 4];
            uint8_t b0 = w_u32 & 0xff, b1 = (w_u32>>8)&0xff, b2 = (w_u32>>16)&0xff, b3 = (w_u32>>24)&0xff;
            int t0=(b0>>sub_bit)&3, t1=(b1>>sub_bit)&3, t2=(b2>>sub_bit)&3, t3=(b3>>sub_bit)&3;
            int w0=(t0==1)-(t0==2), w1=(t1==1)-(t1==2), w2=(t2==1)-(t2==2), w3=(t3==1)-(t3==2);
            int32_t W4 = (uint8_t)w0 | ((uint8_t)w1<<8) | ((uint8_t)w2<<16) | ((uint8_t)w3<<24);
            int32_t X4 = (uint8_t)X_smem[kk] | ((uint8_t)X_smem[kk+1]<<8)
                       | ((uint8_t)X_smem[kk+2]<<16) | ((uint8_t)X_smem[kk+3]<<24);
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

// v8 warp-coop: 1 warp = 1 output col; 32 threads each handle K/32 K-elements.
// Threads in a warp read 32 contiguous K-bytes (perfect coalesced load with
// the v7 col-major layout). Final warp-shuffle reduces.
__global__ void mm_v8_warp(
    const int8_t* X, const uint8_t* W_col, int K, int N, float scale, __half* out)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (warp_id >= N) return;
    int j_byte = warp_id / 4;
    int sub_bit = (warp_id & 3) * 2;
    const uint8_t* W_my_col = W_col + (size_t)j_byte * K;

    int acc = 0;
    // Each thread strides over K with step 32 * 4 (since dp4a covers 4 K)
    // For now keep it simple: each thread handles K/32 K-elements with __dp4a.
    int k_per_thread = (K + 31) / 32;
    int k_start = lane * k_per_thread;
    int k_end = min(K, k_start + k_per_thread);

    int kk = k_start;
    // 4-at-a-time
    for (; kk + 4 <= k_end; kk += 4) {
        uint32_t w_u32 = *reinterpret_cast<const uint32_t*>(W_my_col + kk);
        uint8_t b0 = w_u32 & 0xff, b1 = (w_u32>>8)&0xff, b2 = (w_u32>>16)&0xff, b3 = (w_u32>>24)&0xff;
        int t0=(b0>>sub_bit)&3, t1=(b1>>sub_bit)&3, t2=(b2>>sub_bit)&3, t3=(b3>>sub_bit)&3;
        int w0=(t0==1)-(t0==2), w1=(t1==1)-(t1==2), w2=(t2==1)-(t2==2), w3=(t3==1)-(t3==2);
        int32_t W4 = (uint8_t)w0 | ((uint8_t)w1<<8) | ((uint8_t)w2<<16) | ((uint8_t)w3<<24);
        int32_t X4 = (uint8_t)X[kk] | ((uint8_t)X[kk+1]<<8)
                   | ((uint8_t)X[kk+2]<<16) | ((uint8_t)X[kk+3]<<24);
        acc = __dp4a(X4, W4, acc);
    }
    for (; kk < k_end; ++kk) {
        int xv = X[kk]; if (xv == 0) continue;
        uint8_t b = W_my_col[kk];
        int t = (b >> sub_bit) & 3;
        acc += xv * ((t == 1) - (t == 2));
    }

    // Warp-shuffle reduce
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }

    if (lane == 0) {
        out[warp_id] = __float2half((float)acc * scale);
    }
}

// v9 multi-out: 4 output cols per thread (same byte!), __dp4a, col-major layout.
__global__ void mm_v9_multi(
    const int8_t* X, const uint8_t* W_col, int K, int N, float scale, __half* out)
{
    constexpr int K_TILE = 512;
    __shared__ int8_t X_smem[K_TILE];
    int tid = threadIdx.x;
    int j_base = (blockIdx.x * blockDim.x + tid) * 4;
    if (j_base >= N) return;
    int j_byte = j_base / 4;  // all 4 output cols share the same byte_col
    const uint8_t* W_my_col = W_col + (size_t)j_byte * K;
    int acc[4] = {0, 0, 0, 0};

    for (int k0 = 0; k0 < K; k0 += K_TILE) {
        for (int i = tid; i < K_TILE && k0 + i < K; i += blockDim.x) X_smem[i] = X[k0 + i];
        __syncthreads();
        int kmax = min(K_TILE, K - k0);
        const uint32_t* W32 = reinterpret_cast<const uint32_t*>(W_my_col + k0);
        int kk = 0;
        for (; kk + 4 <= kmax; kk += 4) {
            uint32_t w_u32 = W32[kk / 4];
            uint8_t b0 = w_u32 & 0xff, b1 = (w_u32>>8)&0xff, b2 = (w_u32>>16)&0xff, b3 = (w_u32>>24)&0xff;
            int32_t X4 = (uint8_t)X_smem[kk] | ((uint8_t)X_smem[kk+1]<<8)
                       | ((uint8_t)X_smem[kk+2]<<16) | ((uint8_t)X_smem[kk+3]<<24);
            #pragma unroll
            for (int sub = 0; sub < 4; ++sub) {
                int sb = sub * 2;
                int t0=(b0>>sb)&3, t1=(b1>>sb)&3, t2=(b2>>sb)&3, t3=(b3>>sb)&3;
                int w0=(t0==1)-(t0==2), w1=(t1==1)-(t1==2), w2=(t2==1)-(t2==2), w3=(t3==1)-(t3==2);
                int32_t W4 = (uint8_t)w0 | ((uint8_t)w1<<8) | ((uint8_t)w2<<16) | ((uint8_t)w3<<24);
                acc[sub] = __dp4a(X4, W4, acc[sub]);
            }
        }
        for (; kk < kmax; ++kk) {
            int xv = X_smem[kk]; if (xv == 0) continue;
            uint8_t b = W_my_col[k0 + kk];
            #pragma unroll
            for (int sub = 0; sub < 4; ++sub) {
                int t = (b >> (sub * 2)) & 3;
                acc[sub] += xv * ((t == 1) - (t == 2));
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int sub = 0; sub < 4; ++sub)
        out[j_base + sub] = __float2half((float)acc[sub] * scale);
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
        "v7 ms", "v8warp ms", "v9multi ms", "INT8 ms",
        "v7 GMAC/s", "v8 GMAC/s", "v9 GMAC/s", "INT8 GMAC/s");

    double tot_v7 = 0, tot_v8 = 0, tot_v9 = 0, tot_int8 = 0;
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
        std::vector<uint8_t> hW_row(Wp), hW_col(Wp);
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
        for (size_t jb = 0; jb < (size_t)N/4; ++jb)
            for (size_t k = 0; k < (size_t)K; ++k)
                hW_col[jb * K + k] = hW_row[k * (N/4) + jb];

        int8_t  *dX, *dW_i8;  uint8_t *dW_col;
        __half  *dOut; int32_t *dOut_i32;
        CK(cudaMalloc(&dX, K)); CK(cudaMalloc(&dW_col, Wp)); CK(cudaMalloc(&dW_i8, Wn));
        CK(cudaMalloc(&dOut, N*sizeof(__half))); CK(cudaMalloc(&dOut_i32, N*sizeof(int32_t)));
        CK(cudaMemcpy(dX, hX.data(), K, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dW_col, hW_col.data(), Wp, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dW_i8, hW_int8.data(), Wn, cudaMemcpyHostToDevice));

        float scale = 1.0f / 100.0f;

        int t256 = 256;
        int blocks_v7 = (N + t256 - 1) / t256;
        auto l_v7 = [&]() { mm_v7_colmaj<<<blocks_v7, t256>>>(dX, dW_col, K, N, scale, dOut); };
        // v8: warp-coop, 1 warp per output col. block = 256 threads = 8 warps = 8 output cols
        int blocks_v8 = (N + 8 - 1) / 8;
        auto l_v8 = [&]() { mm_v8_warp<<<blocks_v8, t256>>>(dX, dW_col, K, N, scale, dOut); };
        // v9: 4 outs/thread
        int blocks_v9 = (N/4 + t256 - 1) / t256;
        auto l_v9 = [&]() { mm_v9_multi<<<blocks_v9, t256>>>(dX, dW_col, K, N, scale, dOut); };

        const int32_t aI = 1, bI = 0;
        auto l_int8 = [&]() {
            CB(cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, 1, K,
                &aI, dW_i8, CUDA_R_8I, N, dX, CUDA_R_8I, K,
                &bI, dOut_i32, CUDA_R_32I, N,
                CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        };

        double ms_v7 = time_it(l_v7, n_iters);
        double ms_v8 = time_it(l_v8, n_iters);
        double ms_v9 = time_it(l_v9, n_iters);
        double ms_int8 = time_it(l_int8, n_iters);

        uint64_t lane_macs = (uint64_t)((K + 26) / 27) * (uint64_t)N * 27ull;
        auto g = [&](double ms) { return ms <= 0 ? 0.0 : (double)lane_macs / (ms * 1e6); };

        std::printf("%-10s %6d %8d %10.4f %10.4f %10.4f %10.4f %12.1f %12.1f %12.1f %12.1f\n",
            m.name, K, N, ms_v7, ms_v8, ms_v9, ms_int8,
            g(ms_v7), g(ms_v8), g(ms_v9), g(ms_int8));

        tot_v7 += ms_v7 * m.reps;
        tot_v8 += ms_v8 * m.reps;
        tot_v9 += ms_v9 * m.reps;
        tot_int8 += ms_int8 * m.reps;
        tot_lane_macs += lane_macs * m.reps;

        cudaFree(dX); cudaFree(dW_col); cudaFree(dW_i8); cudaFree(dOut); cudaFree(dOut_i32);
    }

    auto g = [&](double ms) { return (double)tot_lane_macs / (ms * 1e6); };
    std::printf("\n=== Llama 1B forward equivalent (matmul fabric only, M=1) ===\n");
    std::printf("v7 colmaj  (1 out/thread + dp4a) : %8.3f ms (%.1f GMAC/s)\n", tot_v7, g(tot_v7));
    std::printf("v8 warp    (1 warp/col + reduce) : %8.3f ms (%.1f GMAC/s)\n", tot_v8, g(tot_v8));
    std::printf("v9 multi   (4 out/thread + dp4a) : %8.3f ms (%.1f GMAC/s)\n", tot_v9, g(tot_v9));
    std::printf("INT8 TC    (cuBLAS reference)    : %8.3f ms (%.1f GMAC/s)\n", tot_int8, g(tot_int8));
    std::printf("\nBest packed / INT8 TC ratio: %.2fx\n",
        std::min({tot_v7, tot_v8, tot_v9}) / tot_int8);

    cublasDestroy(cublas);
    return 0;
}
