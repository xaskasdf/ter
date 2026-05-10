// ter_cuda_packed_v6.cu -- push past v11 with branchless decode + deeper unroll.
//
// v11 baseline: 4 cols/warp + dp4a + warp-coop reduce, with (t==1)-(t==2) decode.
// v12 branchless: decode = (t & 1) - ((t >> 1) & 1) -- same result, no compares.
// v13 deep_unroll: v12 + 16-K-element body per loop iter (4 dp4a per acc per body).
// v14 extreme: v13 + processed-pair packing trick (2 trits per byte simultaneously
//              via SIMD bit splat -- experimental).

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

// v11 reference: 4 cols per warp, comparison decode
__global__ void mm_v11_warp4(
    const int8_t* X, const uint8_t* W_col, int K, int N, float scale, __half* out)
{
    int warp_block_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    int j_base = warp_block_id * 4;
    if (j_base >= N) return;
    int j_byte = j_base / 4;
    const uint8_t* Wc = W_col + (size_t)j_byte * K;
    int kpt = (K + 31) / 32;
    int kstart = lane * kpt, kend = min(K, kstart + kpt);
    int acc[4] = {0, 0, 0, 0};

    int kk = kstart;
    for (; kk + 4 <= kend; kk += 4) {
        uint32_t w = *reinterpret_cast<const uint32_t*>(Wc + kk);
        uint8_t b0 = w & 0xff, b1 = (w>>8) & 0xff, b2 = (w>>16) & 0xff, b3 = (w>>24) & 0xff;
        int32_t X4 = *reinterpret_cast<const uint32_t*>(X + kk);
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            int sb = sub * 2;
            int t0=(b0>>sb)&3, t1=(b1>>sb)&3, t2=(b2>>sb)&3, t3=(b3>>sb)&3;
            int wv0=(t0==1)-(t0==2), wv1=(t1==1)-(t1==2), wv2=(t2==1)-(t2==2), wv3=(t3==1)-(t3==2);
            int32_t W4 = (uint8_t)wv0 | ((uint8_t)wv1<<8) | ((uint8_t)wv2<<16) | ((uint8_t)wv3<<24);
            acc[sub] = __dp4a(X4, W4, acc[sub]);
        }
    }
    for (; kk < kend; ++kk) {
        int xv = X[kk]; if (xv == 0) continue;
        uint8_t b = Wc[kk];
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            int t = (b >> (sub * 2)) & 3;
            acc[sub] += xv * ((t == 1) - (t == 2));
        }
    }
    #pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
        for (int o = 16; o > 0; o >>= 1) acc[sub] += __shfl_xor_sync(0xffffffff, acc[sub], o);
    }
    if (lane == 0) {
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub)
            out[j_base + sub] = __float2half((float)acc[sub] * scale);
    }
}

// v12: branchless decode  w = (t & 1) - ((t >> 1) & 1)
__device__ __forceinline__ int dec_trit(int t) {
    return (t & 1) - ((t >> 1) & 1);
}
__device__ __forceinline__ int32_t pack_w4_branchless(uint8_t b, int sub) {
    int sb = sub * 2;
    int t0=(b>>sb)&3, t1=t0; // dummy; this fn is not used for v12 (inlined below)
    (void)t1;
    int wv0 = dec_trit(t0);
    return wv0;
}

__global__ void mm_v12_branchless(
    const int8_t* X, const uint8_t* W_col, int K, int N, float scale, __half* out)
{
    int warp_block_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    int j_base = warp_block_id * 4;
    if (j_base >= N) return;
    int j_byte = j_base / 4;
    const uint8_t* Wc = W_col + (size_t)j_byte * K;
    int kpt = (K + 31) / 32;
    int kstart = lane * kpt, kend = min(K, kstart + kpt);
    int acc[4] = {0, 0, 0, 0};

    int kk = kstart;
    for (; kk + 4 <= kend; kk += 4) {
        uint32_t w = *reinterpret_cast<const uint32_t*>(Wc + kk);
        uint8_t b0 = w & 0xff, b1 = (w>>8) & 0xff, b2 = (w>>16) & 0xff, b3 = (w>>24) & 0xff;
        int32_t X4 = *reinterpret_cast<const uint32_t*>(X + kk);
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            int sb = sub * 2;
            int t0=(b0>>sb)&3, t1=(b1>>sb)&3, t2=(b2>>sb)&3, t3=(b3>>sb)&3;
            int wv0=dec_trit(t0), wv1=dec_trit(t1), wv2=dec_trit(t2), wv3=dec_trit(t3);
            int32_t W4 = (uint8_t)wv0 | ((uint8_t)wv1<<8) | ((uint8_t)wv2<<16) | ((uint8_t)wv3<<24);
            acc[sub] = __dp4a(X4, W4, acc[sub]);
        }
    }
    for (; kk < kend; ++kk) {
        int xv = X[kk]; if (xv == 0) continue;
        uint8_t b = Wc[kk];
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            int t = (b >> (sub * 2)) & 3;
            acc[sub] += xv * dec_trit(t);
        }
    }
    #pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
        for (int o = 16; o > 0; o >>= 1) acc[sub] += __shfl_xor_sync(0xffffffff, acc[sub], o);
    }
    if (lane == 0) {
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub)
            out[j_base + sub] = __float2half((float)acc[sub] * scale);
    }
}

// v13: branchless + 16-K unroll body (4 dp4a per accumulator per body).
// 16 K-elements per outer iter = 4 W-uint32 reads + 4 X-uint32 reads,
// each pair drives 4 dp4a (one per col). Total: 16 dp4a per body.
__global__ void mm_v13_unrolled(
    const int8_t* X, const uint8_t* W_col, int K, int N, float scale, __half* out)
{
    int warp_block_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    int j_base = warp_block_id * 4;
    if (j_base >= N) return;
    int j_byte = j_base / 4;
    const uint8_t* Wc = W_col + (size_t)j_byte * K;
    int kpt = ((K + 31) / 32 + 15) & ~15;     // round up to 16
    int kstart = lane * kpt, kend = min(K, kstart + kpt);
    int acc[4] = {0, 0, 0, 0};

    int kk = kstart;
    // 16-at-a-time: 4 inner sub-iters of 4 K-elements each
    for (; kk + 16 <= kend; kk += 16) {
        const uint32_t* W32 = reinterpret_cast<const uint32_t*>(Wc + kk);
        const uint32_t* X32 = reinterpret_cast<const uint32_t*>(X + kk);
        uint32_t w0 = W32[0], w1 = W32[1], w2 = W32[2], w3 = W32[3];
        uint32_t x0 = X32[0], x1 = X32[1], x2 = X32[2], x3 = X32[3];
        #pragma unroll
        for (int qi = 0; qi < 4; ++qi) {
            uint32_t w = (qi==0)?w0:(qi==1?w1:(qi==2?w2:w3));
            uint32_t x = (qi==0)?x0:(qi==1?x1:(qi==2?x2:x3));
            uint8_t b0 = w & 0xff, b1 = (w>>8) & 0xff, b2 = (w>>16) & 0xff, b3 = (w>>24) & 0xff;
            #pragma unroll
            for (int sub = 0; sub < 4; ++sub) {
                int sb = sub * 2;
                int t0=(b0>>sb)&3, t1=(b1>>sb)&3, t2=(b2>>sb)&3, t3=(b3>>sb)&3;
                int wv0=dec_trit(t0), wv1=dec_trit(t1), wv2=dec_trit(t2), wv3=dec_trit(t3);
                int32_t W4 = (uint8_t)wv0 | ((uint8_t)wv1<<8) | ((uint8_t)wv2<<16) | ((uint8_t)wv3<<24);
                acc[sub] = __dp4a((int32_t)x, W4, acc[sub]);
            }
        }
    }
    // 4-at-a-time tail
    for (; kk + 4 <= kend; kk += 4) {
        uint32_t w = *reinterpret_cast<const uint32_t*>(Wc + kk);
        uint8_t b0 = w & 0xff, b1 = (w>>8) & 0xff, b2 = (w>>16) & 0xff, b3 = (w>>24) & 0xff;
        int32_t X4 = *reinterpret_cast<const uint32_t*>(X + kk);
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            int sb = sub * 2;
            int t0=(b0>>sb)&3, t1=(b1>>sb)&3, t2=(b2>>sb)&3, t3=(b3>>sb)&3;
            int wv0=dec_trit(t0), wv1=dec_trit(t1), wv2=dec_trit(t2), wv3=dec_trit(t3);
            int32_t W4 = (uint8_t)wv0 | ((uint8_t)wv1<<8) | ((uint8_t)wv2<<16) | ((uint8_t)wv3<<24);
            acc[sub] = __dp4a(X4, W4, acc[sub]);
        }
    }
    for (; kk < kend; ++kk) {
        int xv = X[kk]; if (xv == 0) continue;
        uint8_t b = Wc[kk];
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            int t = (b >> (sub * 2)) & 3;
            acc[sub] += xv * dec_trit(t);
        }
    }
    #pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
        for (int o = 16; o > 0; o >>= 1) acc[sub] += __shfl_xor_sync(0xffffffff, acc[sub], o);
    }
    if (lane == 0) {
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub)
            out[j_base + sub] = __float2half((float)acc[sub] * scale);
    }
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
    int n_iters = (argc > 1) ? std::atoi(argv[1]) : 100;
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

    std::printf("%-10s %6s %8s %10s %10s %10s %10s %12s %12s %12s %12s\n",
        "shape", "K", "N",
        "v11 ms", "v12 ms", "v13 ms", "INT8 ms",
        "v11 GMAC/s", "v12 GMAC/s", "v13 GMAC/s", "INT8 GMAC/s");

    double tot_v11 = 0, tot_v12 = 0, tot_v13 = 0, tot_int8 = 0, tot_best = 0;
    uint64_t tot_lane_macs = 0;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> d3(-1, 1);

    for (auto& m : mats) {
        int K = m.K, N = m.N;
        size_t Wn = (size_t)K * N, Wp = (size_t)K * (N / 4);
        std::vector<int8_t> hX(K), hW_int8(Wn);
        for (auto& v : hX) v = (int8_t)d3(rng);
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

        int8_t *dX, *dW_i8;  uint8_t *dW_col;
        __half *dOut; int32_t *dOut_i32;
        CK(cudaMalloc(&dX, K)); CK(cudaMalloc(&dW_col, Wp)); CK(cudaMalloc(&dW_i8, Wn));
        CK(cudaMalloc(&dOut, N*sizeof(__half))); CK(cudaMalloc(&dOut_i32, N*sizeof(int32_t)));
        CK(cudaMemcpy(dX, hX.data(), K, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dW_col, hW_col.data(), Wp, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dW_i8, hW_int8.data(), Wn, cudaMemcpyHostToDevice));

        float scale = 1.0f / 100.0f;
        int t256 = 256;
        int blocks = (N/4 + 8 - 1) / 8;  // 8 warps per block, 4 cols per warp
        auto l_v11 = [&]() { mm_v11_warp4    <<<blocks, t256>>>(dX, dW_col, K, N, scale, dOut); };
        auto l_v12 = [&]() { mm_v12_branchless<<<blocks, t256>>>(dX, dW_col, K, N, scale, dOut); };
        auto l_v13 = [&]() { mm_v13_unrolled <<<blocks, t256>>>(dX, dW_col, K, N, scale, dOut); };
        const int32_t aI = 1, bI = 0;
        auto l_int8 = [&]() {
            CB(cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, 1, K,
                &aI, dW_i8, CUDA_R_8I, N, dX, CUDA_R_8I, K,
                &bI, dOut_i32, CUDA_R_32I, N,
                CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        };

        double ms_v11 = time_it(l_v11, n_iters);
        double ms_v12 = time_it(l_v12, n_iters);
        double ms_v13 = time_it(l_v13, n_iters);
        double ms_int8 = time_it(l_int8, n_iters);

        uint64_t lane_macs = (uint64_t)((K + 26) / 27) * (uint64_t)N * 27ull;
        auto g = [&](double ms) { return ms <= 0 ? 0.0 : (double)lane_macs / (ms * 1e6); };

        std::printf("%-10s %6d %8d %10.4f %10.4f %10.4f %10.4f %12.1f %12.1f %12.1f %12.1f\n",
            m.name, K, N, ms_v11, ms_v12, ms_v13, ms_int8,
            g(ms_v11), g(ms_v12), g(ms_v13), g(ms_int8));

        tot_v11 += ms_v11 * m.reps;
        tot_v12 += ms_v12 * m.reps;
        tot_v13 += ms_v13 * m.reps;
        tot_int8 += ms_int8 * m.reps;
        tot_best += std::min({ms_v11, ms_v12, ms_v13}) * m.reps;
        tot_lane_macs += lane_macs * m.reps;

        cudaFree(dX); cudaFree(dW_col); cudaFree(dW_i8); cudaFree(dOut); cudaFree(dOut_i32);
    }

    auto g = [&](double ms) { return (double)tot_lane_macs / (ms * 1e6); };
    std::printf("\n=== Llama 1B forward equivalent (matmul fabric only, M=1) ===\n");
    std::printf("v11 warp4 (compare decode)       : %8.3f ms (%.1f GMAC/s)\n", tot_v11, g(tot_v11));
    std::printf("v12 branchless decode            : %8.3f ms (%.1f GMAC/s)\n", tot_v12, g(tot_v12));
    std::printf("v13 branchless + 16-K unroll     : %8.3f ms (%.1f GMAC/s)\n", tot_v13, g(tot_v13));
    std::printf("Best-per-shape dispatch          : %8.3f ms (%.1f GMAC/s)\n", tot_best, g(tot_best));
    std::printf("INT8 TC reference                : %8.3f ms (%.1f GMAC/s)\n", tot_int8, g(tot_int8));
    std::printf("\nBest dispatch / INT8 TC ratio: %.2fx (>1 means slower; <1 means FASTER)\n",
        tot_best / tot_int8);

    cublasDestroy(cublas);
    return 0;
}
