// ter_cuda_packed_scale.cu -- bench v11_warp4 + ADD-only at Llama 3.1 8B and
// 70B shapes (synthetic data, no GGUF needed). Validates the architectural
// memory-bandwidth advantage at scale.
//
// 8B: H=4096, F=14336, n_layers=32, V=128256, n_heads=32, n_kv_heads=8
// 70B: H=8192, F=28672, n_layers=80, V=128256, n_heads=64, n_kv_heads=8

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <random>
#include <string>

#define CK(call) do { cudaError_t e=(call); if(e){std::fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));std::exit(1);} } while(0)
#define CB(call) do { cublasStatus_t s=(call); if(s){std::fprintf(stderr,"cuBLAS %s:%d %d\n",__FILE__,__LINE__,(int)s);std::exit(1);} } while(0)

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
    for (int sub = 0; sub < 4; ++sub)
        for (int o = 16; o > 0; o >>= 1) acc[sub] += __shfl_xor_sync(0xffffffff, acc[sub], o);
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

void run_model(const char* name, MatShape* mats, int nmats, int n_iters,
               cublasHandle_t cublas)
{
    std::printf("\n==================== %s ====================\n", name);
    std::printf("%-10s %6s %8s %12s %12s %14s %14s %10s\n",
        "shape", "K", "N", "packed ms", "INT8 ms",
        "packed GMAC/s", "INT8 GMAC/s", "ratio");

    double tot_pkd = 0, tot_int8 = 0;
    uint64_t tot_lane_macs = 0;
    size_t tot_packed_bytes = 0, tot_int8_bytes = 0;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dt(0, 2);
    std::uniform_int_distribution<int> dx(-1, 1);

    for (int i = 0; i < nmats; ++i) {
        auto& m = mats[i];
        int K = m.K, N = m.N;
        size_t Wn = (size_t)K * N, Wp = (size_t)K * (N / 4);

        std::vector<int8_t> hX(K), hW_int8(Wn);
        for (auto& v : hX) v = (int8_t)dx(rng);
        std::vector<int> hW_codes(Wn);
        for (auto& c : hW_codes) c = dt(rng);
        for (size_t z = 0; z < Wn; ++z) hW_int8[z] = (hW_codes[z]==1)?1:(hW_codes[z]==2?-1:0);

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
        CK(cudaMalloc(&dX, K)); CK(cudaMalloc(&dW_col, Wp)); CK(cudaMalloc(&dW_i8, Wn));
        CK(cudaMalloc(&dOut, N*sizeof(__half))); CK(cudaMalloc(&dOut_i32, N*sizeof(int32_t)));
        CK(cudaMemcpy(dX, hX.data(), K, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dW_col, hW_col.data(), Wp, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dW_i8, hW_int8.data(), Wn, cudaMemcpyHostToDevice));

        float scale = 1.0f / 100.0f;
        int t256 = 256;
        int blocks = (N/4 + 8 - 1) / 8;
        auto l_pkd = [&]() { mm_v11_warp4<<<blocks, t256>>>(dX, dW_col, K, N, scale, dOut); };
        const int32_t aI = 1, bI = 0;
        auto l_int8 = [&]() {
            CB(cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, 1, K,
                &aI, dW_i8, CUDA_R_8I, N, dX, CUDA_R_8I, K,
                &bI, dOut_i32, CUDA_R_32I, N,
                CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        };

        double ms_pkd = time_it(l_pkd, n_iters);
        double ms_int8 = time_it(l_int8, n_iters);

        uint64_t lane_macs = (uint64_t)((K + 26) / 27) * (uint64_t)N * 27ull;
        auto g = [&](double ms) { return ms <= 0 ? 0.0 : (double)lane_macs / (ms * 1e6); };
        double ratio = ms_int8 / ms_pkd;

        std::printf("%-10s %6d %8d %12.4f %12.4f %14.1f %14.1f %10.2f\n",
            m.name, K, N, ms_pkd, ms_int8, g(ms_pkd), g(ms_int8), ratio);

        tot_pkd += ms_pkd * m.reps;
        tot_int8 += ms_int8 * m.reps;
        tot_lane_macs += lane_macs * m.reps;
        tot_packed_bytes += Wp * m.reps;
        tot_int8_bytes += Wn * m.reps;

        cudaFree(dX); cudaFree(dW_col); cudaFree(dW_i8); cudaFree(dOut); cudaFree(dOut_i32);
    }

    auto g = [&](double ms) { return (double)tot_lane_macs / (ms * 1e6); };
    std::printf("\n%s forward equivalent (matmul fabric only, M=1):\n", name);
    std::printf("  packed v11 : %10.3f ms (%.1f GMAC/s)\n", tot_pkd, g(tot_pkd));
    std::printf("  cuBLAS INT8 TC: %7.3f ms (%.1f GMAC/s)\n", tot_int8, g(tot_int8));
    std::printf("  ratio (>1 = packed FASTER): %.2fx\n", tot_int8 / tot_pkd);
    std::printf("  Memory per fwd matmul fabric: packed=%.1f GB  int8=%.1f GB  ratio=%.1fx less\n",
        tot_packed_bytes / 1e9, tot_int8_bytes / 1e9,
        (double)tot_int8_bytes / tot_packed_bytes);
}

int main(int argc, char** argv) {
    int n_iters = (argc > 1) ? std::atoi(argv[1]) : 30;
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s, %d SMs (n_iters=%d, M=1 single-token gen)\n",
        prop.name, prop.multiProcessorCount, n_iters);
    cublasHandle_t cublas; CB(cublasCreate(&cublas));

    // Llama 3.1 8B: H=4096, F=14336, layers=32, Hkv=8*128=1024, V=128256
    MatShape mats_8b[] = {
        {"Wq",       4096,    4096, 32},
        {"Wk",       4096,    1024, 32},
        {"Wv",       4096,    1024, 32},
        {"Wo",       4096,    4096, 32},
        {"Wgate",    4096,   14336, 32},
        {"Wup",      4096,   14336, 32},
        {"Wdown",   14336,    4096, 32},
        {"lm_head",  4096,  128256,  1},
    };
    run_model("Llama 3.1 8B", mats_8b, 8, n_iters, cublas);

    // Llama 3.1 70B: H=8192, F=28672, layers=80, Hkv=8*128=1024, V=128256
    MatShape mats_70b[] = {
        {"Wq",       8192,    8192, 80},
        {"Wk",       8192,    1024, 80},
        {"Wv",       8192,    1024, 80},
        {"Wo",       8192,    8192, 80},
        {"Wgate",    8192,   28672, 80},
        {"Wup",      8192,   28672, 80},
        {"Wdown",   28672,    8192, 80},
        {"lm_head",  8192,  128256,  1},
    };
    run_model("Llama 3.1 70B (synthetic shapes)", mats_70b, 8, n_iters, cublas);

    cublasDestroy(cublas);
    return 0;
}
