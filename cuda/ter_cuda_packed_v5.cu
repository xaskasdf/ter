// ter_cuda_packed_v5.cu -- v10 attacks the FFN expand bottleneck.
//
// v8 warp-coop is good for small N. v10 unrolls the inner K-loop by 4 (16
// K-elements per loop body, 4 dp4a calls per body, 4 uint32 W reads). The
// goal: amortize per-iteration loop+branch overhead (currently ~30 ops per
// dp4a). With 4-way unroll we get ~30/4 = 7.5 ops per dp4a -> 3-4x throughput.
//
// Bonus: v11 multi-col warp-coop. Each warp computes 4 output columns
// (sharing the same byte_col). 4 dp4a per K-quad, 4x output efficiency
// per warp (best for FFN expand-like K=2048 N=8192).

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

// v8 reference: 1 warp = 1 col, no unroll
__global__ void mm_v8_warp(
    const int8_t* X, const uint8_t* W_col, int K, int N, float scale, __half* out)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (warp_id >= N) return;
    int j_byte = warp_id / 4, sub_bit = (warp_id & 3) * 2;
    const uint8_t* Wc = W_col + (size_t)j_byte * K;
    int kpt = (K + 31) / 32;
    int kstart = lane * kpt, kend = min(K, kstart + kpt);
    int acc = 0;
    int kk = kstart;
    for (; kk + 4 <= kend; kk += 4) {
        uint32_t w = *reinterpret_cast<const uint32_t*>(Wc + kk);
        uint8_t b0=w&0xff, b1=(w>>8)&0xff, b2=(w>>16)&0xff, b3=(w>>24)&0xff;
        int t0=(b0>>sub_bit)&3, t1=(b1>>sub_bit)&3, t2=(b2>>sub_bit)&3, t3=(b3>>sub_bit)&3;
        int w0=(t0==1)-(t0==2), w1=(t1==1)-(t1==2), w2=(t2==1)-(t2==2), w3=(t3==1)-(t3==2);
        int32_t W4 = (uint8_t)w0 | ((uint8_t)w1<<8) | ((uint8_t)w2<<16) | ((uint8_t)w3<<24);
        int32_t X4 = (uint8_t)X[kk] | ((uint8_t)X[kk+1]<<8)
                   | ((uint8_t)X[kk+2]<<16) | ((uint8_t)X[kk+3]<<24);
        acc = __dp4a(X4, W4, acc);
    }
    for (; kk < kend; ++kk) {
        int xv = X[kk]; if (xv == 0) continue;
        uint8_t b = Wc[kk]; int t = (b >> sub_bit) & 3;
        acc += xv * ((t == 1) - (t == 2));
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
    if (lane == 0) out[warp_id] = __float2half((float)acc * scale);
}

// v10: warp-coop + inner K-loop unrolled by 4 (16 K-elements per body, 4 dp4a)
__global__ void mm_v10_unroll(
    const int8_t* X, const uint8_t* W_col, int K, int N, float scale, __half* out)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (warp_id >= N) return;
    int j_byte = warp_id / 4, sub_bit = (warp_id & 3) * 2;
    const uint8_t* Wc = W_col + (size_t)j_byte * K;
    int kpt = ((K + 31) / 32 + 15) & ~15;     // round up to multiple of 16
    int kstart = lane * kpt, kend = min(K, kstart + kpt);
    int acc = 0;
    int kk = kstart;

    // 16-at-a-time (4 dp4a per body)
    for (; kk + 16 <= kend; kk += 16) {
        const uint32_t* W32 = reinterpret_cast<const uint32_t*>(Wc + kk);
        const uint32_t* X32 = reinterpret_cast<const uint32_t*>(X + kk);
        uint32_t w0 = W32[0], w1 = W32[1], w2 = W32[2], w3 = W32[3];
        uint32_t x0 = X32[0], x1 = X32[1], x2 = X32[2], x3 = X32[3];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint32_t w = (i==0)?w0:(i==1?w1:(i==2?w2:w3));
            uint32_t x = (i==0)?x0:(i==1?x1:(i==2?x2:x3));
            uint8_t b0=w&0xff, b1=(w>>8)&0xff, b2=(w>>16)&0xff, b3=(w>>24)&0xff;
            int t0=(b0>>sub_bit)&3, t1=(b1>>sub_bit)&3, t2=(b2>>sub_bit)&3, t3=(b3>>sub_bit)&3;
            int wv0=(t0==1)-(t0==2), wv1=(t1==1)-(t1==2), wv2=(t2==1)-(t2==2), wv3=(t3==1)-(t3==2);
            int32_t W4 = (uint8_t)wv0 | ((uint8_t)wv1<<8) | ((uint8_t)wv2<<16) | ((uint8_t)wv3<<24);
            acc = __dp4a((int32_t)x, W4, acc);
        }
    }
    // 4-at-a-time tail
    for (; kk + 4 <= kend; kk += 4) {
        uint32_t w = *reinterpret_cast<const uint32_t*>(Wc + kk);
        uint8_t b0=w&0xff, b1=(w>>8)&0xff, b2=(w>>16)&0xff, b3=(w>>24)&0xff;
        int t0=(b0>>sub_bit)&3, t1=(b1>>sub_bit)&3, t2=(b2>>sub_bit)&3, t3=(b3>>sub_bit)&3;
        int w0=(t0==1)-(t0==2), w1=(t1==1)-(t1==2), w2=(t2==1)-(t2==2), w3=(t3==1)-(t3==2);
        int32_t W4 = (uint8_t)w0 | ((uint8_t)w1<<8) | ((uint8_t)w2<<16) | ((uint8_t)w3<<24);
        int32_t X4 = *reinterpret_cast<const uint32_t*>(X + kk);
        acc = __dp4a(X4, W4, acc);
    }
    for (; kk < kend; ++kk) {
        int xv = X[kk]; if (xv == 0) continue;
        uint8_t b = Wc[kk]; int t = (b >> sub_bit) & 3;
        acc += xv * ((t == 1) - (t == 2));
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
    if (lane == 0) out[warp_id] = __float2half((float)acc * scale);
}

// v11: 4 cols per warp (sharing byte_col); each thread accumulates 4 partial sums
__global__ void mm_v11_warp4(
    const int8_t* X, const uint8_t* W_col, int K, int N, float scale, __half* out)
{
    int warp_block_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    int j_base = warp_block_id * 4;
    if (j_base >= N) return;
    int j_byte = j_base / 4;  // all 4 share the same byte_col
    const uint8_t* Wc = W_col + (size_t)j_byte * K;
    int kpt = (K + 31) / 32;
    int kstart = lane * kpt, kend = min(K, kstart + kpt);
    int acc[4] = {0, 0, 0, 0};

    int kk = kstart;
    for (; kk + 4 <= kend; kk += 4) {
        uint32_t w = *reinterpret_cast<const uint32_t*>(Wc + kk);
        uint8_t b0=w&0xff, b1=(w>>8)&0xff, b2=(w>>16)&0xff, b3=(w>>24)&0xff;
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
        "v8 ms", "v10unr ms", "v11w4 ms", "INT8 ms",
        "v8 GMAC/s", "v10 GMAC/s", "v11 GMAC/s", "INT8 GMAC/s");

    double tot_v8 = 0, tot_v10 = 0, tot_v11 = 0, tot_int8 = 0;
    double tot_best = 0;
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
        // v8/v10: 1 warp per col -> blocks = N / 8 (8 warps per block)
        int blocks_v8 = (N + 8 - 1) / 8;
        auto l_v8  = [&]() { mm_v8_warp   <<<blocks_v8, t256>>>(dX, dW_col, K, N, scale, dOut); };
        auto l_v10 = [&]() { mm_v10_unroll<<<blocks_v8, t256>>>(dX, dW_col, K, N, scale, dOut); };
        // v11: 4 cols per warp -> blocks = N / 32
        int blocks_v11 = (N/4 + 8 - 1) / 8;
        auto l_v11 = [&]() { mm_v11_warp4 <<<blocks_v11, t256>>>(dX, dW_col, K, N, scale, dOut); };

        const int32_t aI = 1, bI = 0;
        auto l_int8 = [&]() {
            CB(cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, 1, K,
                &aI, dW_i8, CUDA_R_8I, N, dX, CUDA_R_8I, K,
                &bI, dOut_i32, CUDA_R_32I, N,
                CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        };

        double ms_v8  = time_it(l_v8, n_iters);
        double ms_v10 = time_it(l_v10, n_iters);
        double ms_v11 = time_it(l_v11, n_iters);
        double ms_int8 = time_it(l_int8, n_iters);

        uint64_t lane_macs = (uint64_t)((K + 26) / 27) * (uint64_t)N * 27ull;
        auto g = [&](double ms) { return ms <= 0 ? 0.0 : (double)lane_macs / (ms * 1e6); };

        std::printf("%-10s %6d %8d %10.4f %10.4f %10.4f %10.4f %12.1f %12.1f %12.1f %12.1f\n",
            m.name, K, N, ms_v8, ms_v10, ms_v11, ms_int8,
            g(ms_v8), g(ms_v10), g(ms_v11), g(ms_int8));

        tot_v8 += ms_v8 * m.reps;
        tot_v10 += ms_v10 * m.reps;
        tot_v11 += ms_v11 * m.reps;
        tot_int8 += ms_int8 * m.reps;
        tot_best += std::min({ms_v8, ms_v10, ms_v11}) * m.reps;
        tot_lane_macs += lane_macs * m.reps;

        cudaFree(dX); cudaFree(dW_col); cudaFree(dW_i8); cudaFree(dOut); cudaFree(dOut_i32);
    }

    auto g = [&](double ms) { return (double)tot_lane_macs / (ms * 1e6); };
    std::printf("\n=== Llama 1B forward equivalent (matmul fabric only, M=1) ===\n");
    std::printf("v8  warp                : %8.3f ms (%.1f GMAC/s)\n", tot_v8,  g(tot_v8));
    std::printf("v10 warp + 16-K unroll  : %8.3f ms (%.1f GMAC/s)\n", tot_v10, g(tot_v10));
    std::printf("v11 warp x 4 cols       : %8.3f ms (%.1f GMAC/s)\n", tot_v11, g(tot_v11));
    std::printf("Best-per-shape dispatch : %8.3f ms (%.1f GMAC/s)\n", tot_best, g(tot_best));
    std::printf("INT8 TC reference       : %8.3f ms (%.1f GMAC/s)\n", tot_int8, g(tot_int8));
    std::printf("\nBest dispatch / INT8 TC ratio: %.2fx\n", tot_best / tot_int8);

    cublasDestroy(cublas);
    return 0;
}
