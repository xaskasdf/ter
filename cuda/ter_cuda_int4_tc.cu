// ter_cuda_int4_tc.cu -- Phase E: INT4 tensor cores via wmma::s4 for ternary
// matmul. Targets the prefill regime (M >= 16) where Phase A showed cuBLAS
// INT8 TC wins by 8x per-token throughput.
//
// Theory: ternary {-1, 0, +1} packs as int4 (4 bits/elem, 1 bit wasted).
// Memory density 4 bits/elem (vs our packed 2 bits/elem = -2x).
// Compute: INT4 TC peak ~568 TOPS vs INT8 TC ~284 TOPS = +2x peak.
// Net: should win at large M where compute dominates over memory.
//
// 8x8x32 wmma::s4 fragments. K must be multiple of 32.

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <random>

using namespace nvcuda;
using namespace nvcuda::wmma::experimental;

#define CK(call) do { cudaError_t e=(call); if(e){std::fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));std::exit(1);} } while(0)
#define CB(call) do { cublasStatus_t s=(call); if(s){std::fprintf(stderr,"cuBLAS %s:%d %d\n",__FILE__,__LINE__,(int)s);std::exit(1);} } while(0)

// Kernel: INT4 tensor cores. Tiles: 8x8x32. One warp computes 8x8 output tile.
// A is M x K (int4 row-major), B is K x N (int4 col-major), C is M x N (int row-major).
// M, N must be multiples of 8. K must be multiple of 32.
//
// Each warp processes one (m_tile, n_tile) of size 8x8 by accumulating
// over the K dimension in chunks of 32.

template<int M_TILE, int N_TILE, int K_TILE>
__global__ void mm_int4_tc(
    const int8_t* __restrict__ A_packed,  // M x (K/2)  -- 2 int4s per byte
    const int8_t* __restrict__ B_packed,  // (K/2) x N  -- col-major in our layout: B_packed[n*(K/2) + k_byte]
    int M, int N, int K, float scale, __half* C)
{
    static_assert(M_TILE == 8 && N_TILE == 8 && K_TILE == 32, "wmma s4 fragment");
    int tile_m = blockIdx.y;
    int tile_n = blockIdx.x;
    int m0 = tile_m * M_TILE;
    int n0 = tile_n * N_TILE;
    if (m0 >= M || n0 >= N) return;

    wmma::fragment<wmma::matrix_a, M_TILE, N_TILE, K_TILE, precision::s4, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M_TILE, N_TILE, K_TILE, precision::s4, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, M_TILE, N_TILE, K_TILE, int> c_frag;
    wmma::fill_fragment(c_frag, 0);

    int K_bytes = K / 2;  // 2 int4 per byte
    for (int k0 = 0; k0 < K; k0 += K_TILE) {
        int k_byte = k0 / 2;
        // A layout: row-major, leading dim = K elements (= K/2 bytes per row)
        const int8_t* A_ptr = A_packed + (size_t)m0 * K_bytes + k_byte;
        // B layout: col-major, leading dim = K elements
        const int8_t* B_ptr = B_packed + (size_t)n0 * K_bytes + k_byte;

        wmma::load_matrix_sync(a_frag, A_ptr, K);
        wmma::load_matrix_sync(b_frag, B_ptr, K);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store as int row-major into a temporary, then convert to fp16
    __shared__ int sC[M_TILE * N_TILE];
    wmma::store_matrix_sync(sC, c_frag, N_TILE, wmma::mem_row_major);
    __syncthreads();

    int lane = threadIdx.x & 31;
    #pragma unroll
    for (int idx = lane; idx < M_TILE * N_TILE; idx += 32) {
        int mi = idx / N_TILE;
        int ni = idx % N_TILE;
        int mm_idx = m0 + mi;
        int nn_idx = n0 + ni;
        if (mm_idx < M && nn_idx < N)
            C[(size_t)mm_idx * N + nn_idx] = __float2half((float)sC[mi * N_TILE + ni] * scale);
    }
}

static int8_t pack_2_int4(int hi, int lo) {
    // hi and lo are signed 4-bit values in [-8, 7]
    return (int8_t)(((hi & 0xf) << 4) | (lo & 0xf));
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

struct Shape { const char* name; int M; int K; int N; };

int main(int argc, char** argv) {
    int n_iters = (argc > 1) ? std::atoi(argv[1]) : 50;
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s, %d SMs (n_iters=%d)\n", prop.name, prop.multiProcessorCount, n_iters);
    cublasHandle_t cublas; CB(cublasCreate(&cublas));

    // Sweep batch sizes for FFN-expand shape (the regime where INT8 TC was beating us)
    Shape shapes[] = {
        // Llama 1B FFN-expand: H=2048, F=8192
        {"1B-Wgate-M16",   16, 2048, 8192},
        {"1B-Wgate-M64",   64, 2048, 8192},
        {"1B-Wgate-M256", 256, 2048, 8192},
        // Llama 8B FFN-expand: H=4096, F=14336
        {"8B-Wgate-M16",   16, 4096, 14336},
        {"8B-Wgate-M64",   64, 4096, 14336},
        // Llama 8B FFN-down: H=4096, F=14336 (K=14336 N=4096)
        {"8B-Wdown-M16",   16, 14336, 4096},
        {"8B-Wdown-M64",   64, 14336, 4096},
        // Llama 70B FFN-expand: H=8192, F=28672
        {"70B-Wgate-M16",  16, 8192, 28672},
        {"70B-Wdown-M16",  16, 28672, 8192},
        // Llama 1B M=1 (latency regime sanity check)
        {"1B-Wgate-M8",     8, 2048, 8192},  // M>=8 required by frag
    };

    std::printf("\n%-18s %5s %6s %6s %14s %14s %14s %12s %12s\n",
        "shape", "M", "K", "N",
        "INT4 TC ms", "INT8 TC ms", "ratio (>1=INT4 win)",
        "INT4 GMAC/s", "INT8 GMAC/s");

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dt(-1, 1);

    for (auto& s : shapes) {
        int M = s.M, K = s.K, N = s.N;
        if (K % 32 != 0 || M % 8 != 0 || N % 8 != 0) {
            std::printf("%-18s SKIP (alignment)\n", s.name);
            continue;
        }

        // Build A (X activations, M x K, ternary) and B (W weights, K x N, ternary)
        std::vector<int8_t> hA_int8((size_t)M * K), hB_int8((size_t)K * N);
        for (auto& v : hA_int8) v = (int8_t)dt(rng);
        for (auto& v : hB_int8) v = (int8_t)dt(rng);

        // Pack into int4: A_packed[m * (K/2) + k/2] holds A[m,k] (low) and A[m,k+1] (high)
        std::vector<int8_t> hA_packed((size_t)M * (K/2));
        for (int m = 0; m < M; ++m)
            for (int kb = 0; kb < K/2; ++kb)
                hA_packed[(size_t)m * (K/2) + kb] = pack_2_int4(
                    hA_int8[(size_t)m * K + kb*2 + 1],
                    hA_int8[(size_t)m * K + kb*2]);

        // B is col-major in our layout: B_packed[n * (K/2) + k/2]
        std::vector<int8_t> hB_packed((size_t)N * (K/2));
        for (int n = 0; n < N; ++n)
            for (int kb = 0; kb < K/2; ++kb)
                hB_packed[(size_t)n * (K/2) + kb] = pack_2_int4(
                    hB_int8[(size_t)(kb*2+1) * N + n],
                    hB_int8[(size_t)(kb*2) * N + n]);

        int8_t *dA_pkd, *dB_pkd, *dA_i8, *dB_i8;
        __half *dC; int32_t *dC_i32;
        size_t sA = (size_t)M * (K/2);
        size_t sB = (size_t)N * (K/2);
        size_t sAi = (size_t)M * K;
        size_t sBi = (size_t)K * N;
        CK(cudaMalloc(&dA_pkd, sA)); CK(cudaMalloc(&dB_pkd, sB));
        CK(cudaMalloc(&dA_i8, sAi)); CK(cudaMalloc(&dB_i8, sBi));
        CK(cudaMalloc(&dC, (size_t)M*N*sizeof(__half)));
        CK(cudaMalloc(&dC_i32, (size_t)M*N*sizeof(int32_t)));
        CK(cudaMemcpy(dA_pkd, hA_packed.data(), sA, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dB_pkd, hB_packed.data(), sB, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dA_i8, hA_int8.data(), sAi, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dB_i8, hB_int8.data(), sBi, cudaMemcpyHostToDevice));

        // INT4 TC kernel launch: grid = (N/8, M/8), block = 32 (one warp per tile)
        dim3 grid(N/8, M/8);
        dim3 block(32);
        float scale = 1.0f / 100.0f;
        auto l_int4 = [&]() {
            mm_int4_tc<8,8,32><<<grid, block>>>(dA_pkd, dB_pkd, M, N, K, scale, dC);
        };

        // Reference: cuBLAS INT8 TC
        const int32_t aI = 1, bI = 0;
        auto l_int8 = [&]() {
            // C[N,M] = B[N,K] * A[K,M] (col-major for cuBLAS): C = B^T * A^T but we use OP_N
            CB(cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                &aI, dB_i8, CUDA_R_8I, N, dA_i8, CUDA_R_8I, K,
                &bI, dC_i32, CUDA_R_32I, N,
                CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        };

        double ms_int4 = time_it(l_int4, n_iters);
        double ms_int8 = time_it(l_int8, n_iters);

        uint64_t macs = (uint64_t)M * (uint64_t)N * (uint64_t)K;
        auto g = [&](double ms) { return (double)macs / (ms * 1e6); };

        std::printf("%-18s %5d %6d %6d %14.4f %14.4f %14.2fx %12.1f %12.1f\n",
            s.name, M, K, N, ms_int4, ms_int8, ms_int8/ms_int4, g(ms_int4), g(ms_int8));

        cudaFree(dA_pkd); cudaFree(dB_pkd);
        cudaFree(dA_i8); cudaFree(dB_i8);
        cudaFree(dC); cudaFree(dC_i32);
    }

    cublasDestroy(cublas);
    return 0;
}
