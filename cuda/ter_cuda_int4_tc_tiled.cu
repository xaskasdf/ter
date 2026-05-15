// ter_cuda_int4_tc_tiled.cu -- Phase E v2: GEMM-tiled INT4 TC
//
// Naive kernel (ter_cuda_int4_tc.cu) loses at M>=64 FFN-expand because each
// 8x8 warp tile re-reads B columns from L2/DRAM. Multi-warp shared-memory
// blocking fixes this: a block of 16 warps cooperatively loads A and B tiles
// into shared memory, then each warp computes its 8x8 sub-tile from SMEM.
//
// Tile geometry: BM=32, BN=32, BK=32. 4x4 grid of 8x8 wmma tiles per block.
// Block size: 16 warps * 32 threads = 512 threads.

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

template<int BM, int BN, int BK>
__global__ void mm_int4_tc_tiled(
    const int8_t* __restrict__ A_packed,  // M x (K/2)  row-major
    const int8_t* __restrict__ B_packed,  // (K/2) x N  col-major  i.e. B_packed[n*(K/2) + k_byte]
    int M, int N, int K, float scale, __half* C)
{
    constexpr int WARP_M = BM / 8;
    constexpr int WARP_N = BN / 8;
    constexpr int N_WARPS = WARP_M * WARP_N;
    constexpr int BK_BYTES = BK / 2;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int warp_m = warp_id / WARP_N;
    int warp_n = warp_id % WARP_N;

    int block_m = blockIdx.y * BM;
    int block_n = blockIdx.x * BN;
    if (block_m >= M || block_n >= N) return;

    __shared__ int8_t sA[BM * BK_BYTES];
    __shared__ int8_t sB[BN * BK_BYTES];
    __shared__ int sC[BM * BN];

    wmma::fragment<wmma::matrix_a, 8, 8, 32, precision::s4, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 8, 8, 32, precision::s4, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 8, 8, 32, int> c_frag;
    wmma::fill_fragment(c_frag, 0);

    int K_bytes = K / 2;
    int A_tile_bytes = BM * BK_BYTES;
    int B_tile_bytes = BN * BK_BYTES;

    for (int k0 = 0; k0 < K; k0 += BK) {
        int k_byte = k0 / 2;

        // Cooperatively load A tile (BM rows × BK_BYTES bytes per row)
        for (int idx = tid; idx < A_tile_bytes; idx += N_WARPS * 32) {
            int r = idx / BK_BYTES;
            int kb = idx % BK_BYTES;
            int gm = block_m + r;
            sA[idx] = (gm < M) ? A_packed[(size_t)gm * K_bytes + k_byte + kb] : (int8_t)0;
        }
        // Cooperatively load B tile (BN cols × BK_BYTES bytes per col)
        for (int idx = tid; idx < B_tile_bytes; idx += N_WARPS * 32) {
            int c = idx / BK_BYTES;
            int kb = idx % BK_BYTES;
            int gn = block_n + c;
            sB[idx] = (gn < N) ? B_packed[(size_t)gn * K_bytes + k_byte + kb] : (int8_t)0;
        }
        __syncthreads();

        // Each warp loads its 8 rows of A and 8 cols of B, then mma
        const int8_t* sA_warp = sA + (warp_m * 8) * BK_BYTES;
        const int8_t* sB_warp = sB + (warp_n * 8) * BK_BYTES;
        wmma::load_matrix_sync(a_frag, sA_warp, BK);
        wmma::load_matrix_sync(b_frag, sB_warp, BK);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    // Each warp writes its 8x8 tile to shared mem (row-major within the BMxBN tile)
    int* sC_warp = sC + (warp_m * 8) * BN + (warp_n * 8);
    wmma::store_matrix_sync(sC_warp, c_frag, BN, wmma::mem_row_major);
    __syncthreads();

    // Cooperative write to C
    int n_outputs = BM * BN;
    for (int idx = tid; idx < n_outputs; idx += N_WARPS * 32) {
        int mi = idx / BN;
        int ni = idx % BN;
        int gm = block_m + mi;
        int gn = block_n + ni;
        if (gm < M && gn < N)
            C[(size_t)gm * N + gn] = __float2half((float)sC[mi * BN + ni] * scale);
    }
}

static int8_t pack_2_int4(int hi, int lo) {
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
    int n_iters = (argc > 1) ? std::atoi(argv[1]) : 100;
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s, %d SMs (n_iters=%d, GEMM-tiled BM=BN=BK=32)\n",
        prop.name, prop.multiProcessorCount, n_iters);
    cublasHandle_t cublas; CB(cublasCreate(&cublas));

    // Focus on the M=64 FFN-expand shapes where naive INT4 lost
    Shape shapes[] = {
        {"1B-Wgate-M32",   32, 2048, 8192},
        {"1B-Wgate-M64",   64, 2048, 8192},
        {"1B-Wgate-M256", 256, 2048, 8192},
        {"8B-Wgate-M32",   32, 4096, 14336},
        {"8B-Wgate-M64",   64, 4096, 14336},
        {"8B-Wgate-M256", 256, 4096, 14336},
        {"8B-Wdown-M32",   32, 14336, 4096},
        {"8B-Wdown-M64",   64, 14336, 4096},
        {"70B-Wgate-M32",  32, 8192, 28672},
        {"70B-Wgate-M64",  64, 8192, 28672},
    };

    std::printf("\n%-18s %5s %6s %6s %14s %14s %14s %12s %12s\n",
        "shape", "M", "K", "N",
        "INT4-tile ms", "INT8 TC ms", "ratio (>1=tile win)",
        "tile GMAC/s", "INT8 GMAC/s");

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dt(-1, 1);

    constexpr int BM = 32, BN = 32, BK = 32;

    for (auto& s : shapes) {
        int M = s.M, K = s.K, N = s.N;
        if (K % BK != 0 || M % BM != 0 || N % BN != 0) {
            std::printf("%-18s SKIP (alignment)\n", s.name);
            continue;
        }

        std::vector<int8_t> hA_int8((size_t)M * K), hB_int8((size_t)K * N);
        for (auto& v : hA_int8) v = (int8_t)dt(rng);
        for (auto& v : hB_int8) v = (int8_t)dt(rng);

        std::vector<int8_t> hA_packed((size_t)M * (K/2));
        for (int m = 0; m < M; ++m)
            for (int kb = 0; kb < K/2; ++kb)
                hA_packed[(size_t)m * (K/2) + kb] = pack_2_int4(
                    hA_int8[(size_t)m * K + kb*2 + 1],
                    hA_int8[(size_t)m * K + kb*2]);

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

        dim3 grid(N/BN, M/BM);
        dim3 block(16 * 32);  // 16 warps
        float scale = 1.0f / 100.0f;
        auto l_int4 = [&]() {
            mm_int4_tc_tiled<BM,BN,BK><<<grid, block>>>(dA_pkd, dB_pkd, M, N, K, scale, dC);
        };

        const int32_t aI = 1, bI = 0;
        auto l_int8 = [&]() {
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
