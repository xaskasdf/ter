// ter_cuda_hybrid_dispatch.cu -- Phase E v3: per-shape best-of-toolkit
// hybrid dispatch. Benches every Llama 1B/8B matmul shape at M=1 and M=16
// against four backends (packed v11, INT4 TC naive, INT4 TC tiled, cuBLAS
// INT8 TC), picks the best per shape, then reports total forward-pass
// matmul-fabric time for: (a) all-INT8 TC reference, (b) all-INT4-TC,
// (c) all-packed-v11, (d) hybrid (best-per-shape).
//
// Note: GPU is currently shared with game/embedding-shard workload, so
// these numbers have ~2x absolute variance. Ratios within a single run
// are stable.

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>

using namespace nvcuda;
using namespace nvcuda::wmma::experimental;

#define CK(call) do { cudaError_t e=(call); if(e){std::fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));std::exit(1);} } while(0)
#define CB(call) do { cublasStatus_t s=(call); if(s){std::fprintf(stderr,"cuBLAS %s:%d %d\n",__FILE__,__LINE__,(int)s);std::exit(1);} } while(0)

// --- Kernel 1: packed v11 warp-coop (M=1 specialist) ---
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

// --- Kernel 2: INT4 TC naive (1 warp per 8x8 tile) ---
__global__ void mm_int4_tc_naive(
    const int8_t* __restrict__ A_packed, const int8_t* __restrict__ B_packed,
    int M, int N, int K, float scale, __half* C)
{
    int tile_m = blockIdx.y;
    int tile_n = blockIdx.x;
    int m0 = tile_m * 8;
    int n0 = tile_n * 8;
    if (m0 >= M || n0 >= N) return;

    wmma::fragment<wmma::matrix_a, 8, 8, 32, precision::s4, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 8, 8, 32, precision::s4, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 8, 8, 32, int> c_frag;
    wmma::fill_fragment(c_frag, 0);

    int K_bytes = K / 2;
    for (int k0 = 0; k0 < K; k0 += 32) {
        int k_byte = k0 / 2;
        wmma::load_matrix_sync(a_frag, A_packed + (size_t)m0 * K_bytes + k_byte, K);
        wmma::load_matrix_sync(b_frag, B_packed + (size_t)n0 * K_bytes + k_byte, K);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    __shared__ int sC[64];
    wmma::store_matrix_sync(sC, c_frag, 8, wmma::mem_row_major);
    __syncthreads();
    int lane = threadIdx.x & 31;
    for (int idx = lane; idx < 64; idx += 32) {
        int mi = idx / 8, ni = idx % 8;
        if (m0 + mi < M && n0 + ni < N)
            C[(size_t)(m0 + mi) * N + (n0 + ni)] = __float2half((float)sC[idx] * scale);
    }
}

// --- Kernel 3: INT4 TC GEMM-tiled (BM=BN=BK=32, 16 warps/block) ---
template<int BM, int BN, int BK>
__global__ void mm_int4_tc_tiled(
    const int8_t* __restrict__ A_packed, const int8_t* __restrict__ B_packed,
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
        for (int idx = tid; idx < A_tile_bytes; idx += N_WARPS * 32) {
            int r = idx / BK_BYTES, kb = idx % BK_BYTES;
            int gm = block_m + r;
            sA[idx] = (gm < M) ? A_packed[(size_t)gm * K_bytes + k_byte + kb] : (int8_t)0;
        }
        for (int idx = tid; idx < B_tile_bytes; idx += N_WARPS * 32) {
            int c = idx / BK_BYTES, kb = idx % BK_BYTES;
            int gn = block_n + c;
            sB[idx] = (gn < N) ? B_packed[(size_t)gn * K_bytes + k_byte + kb] : (int8_t)0;
        }
        __syncthreads();
        wmma::load_matrix_sync(a_frag, sA + (warp_m * 8) * BK_BYTES, BK);
        wmma::load_matrix_sync(b_frag, sB + (warp_n * 8) * BK_BYTES, BK);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    int* sC_warp = sC + (warp_m * 8) * BN + (warp_n * 8);
    wmma::store_matrix_sync(sC_warp, c_frag, BN, wmma::mem_row_major);
    __syncthreads();

    for (int idx = tid; idx < BM * BN; idx += N_WARPS * 32) {
        int mi = idx / BN, ni = idx % BN;
        int gm = block_m + mi, gn = block_n + ni;
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

struct Shape { const char* name; int K; int N; int reps; };

struct ShapeBench {
    const char* name;
    int M, K, N, reps;
    double ms_packed, ms_int4_naive, ms_int4_tiled, ms_int8;
    const char* best_name;
    double best_ms;
};

static double bench_shape(int M, int K, int N, int n_iters,
                          double& ms_packed, double& ms_int4_naive,
                          double& ms_int4_tiled, double& ms_int8,
                          cublasHandle_t cublas)
{
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dt(0, 2);
    std::uniform_int_distribution<int> dx(-1, 1);

    // Packed v11 storage (col-major byte layout, 4 trits per byte)
    std::vector<int8_t> hX_int8((size_t)M * K), hX_v11(K);
    for (auto& v : hX_int8) v = (int8_t)dx(rng);
    for (int k = 0; k < K; ++k) hX_v11[k] = hX_int8[k];  // M=1 row of X (used for packed v11)

    std::vector<int> hW_codes((size_t)K * N);
    for (auto& c : hW_codes) c = dt(rng);

    std::vector<int8_t> hW_int8((size_t)K * N);
    for (size_t z = 0; z < (size_t)K*N; ++z)
        hW_int8[z] = (hW_codes[z]==1)?1:(hW_codes[z]==2?-1:0);

    // Packed v11: col-major bytes, 4 cols per byte
    std::vector<uint8_t> hW_col((size_t)K * (N/4));
    for (int jb = 0; jb < N/4; ++jb)
        for (int k = 0; k < K; ++k) {
            uint8_t b = 0;
            for (int t = 0; t < 4; ++t)
                b |= (hW_codes[k * N + jb * 4 + t] & 0x3) << (t * 2);
            hW_col[(size_t)jb * K + k] = b;
        }

    // INT4 packed: A row-major, B col-major
    std::vector<int8_t> hA_packed((size_t)M * (K/2));
    for (int m = 0; m < M; ++m)
        for (int kb = 0; kb < K/2; ++kb)
            hA_packed[(size_t)m * (K/2) + kb] = pack_2_int4(
                hX_int8[(size_t)m * K + kb*2 + 1],
                hX_int8[(size_t)m * K + kb*2]);
    std::vector<int8_t> hB_packed((size_t)N * (K/2));
    for (int n = 0; n < N; ++n)
        for (int kb = 0; kb < K/2; ++kb)
            hB_packed[(size_t)n * (K/2) + kb] = pack_2_int4(
                hW_int8[(size_t)(kb*2+1) * N + n],
                hW_int8[(size_t)(kb*2) * N + n]);

    int8_t *dX, *dW_i8, *dA_pkd, *dB_pkd;
    uint8_t *dW_col;
    __half *dOut;
    int32_t *dOut_i32;
    CK(cudaMalloc(&dX, M*K)); CK(cudaMalloc(&dW_i8, (size_t)K*N));
    CK(cudaMalloc(&dW_col, (size_t)K * (N/4)));
    CK(cudaMalloc(&dA_pkd, (size_t)M * (K/2)));
    CK(cudaMalloc(&dB_pkd, (size_t)N * (K/2)));
    CK(cudaMalloc(&dOut, (size_t)M*N*sizeof(__half)));
    CK(cudaMalloc(&dOut_i32, (size_t)M*N*sizeof(int32_t)));

    CK(cudaMemcpy(dX, hX_int8.data(), M*K, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dW_i8, hW_int8.data(), (size_t)K*N, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dW_col, hW_col.data(), (size_t)K*(N/4), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dA_pkd, hA_packed.data(), (size_t)M*(K/2), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB_pkd, hB_packed.data(), (size_t)N*(K/2), cudaMemcpyHostToDevice));

    float scale = 1.0f / 100.0f;

    // Packed v11 (only meaningful at M=1; for M>1 we still bench but it's row-by-row)
    int blocks = (N/4 + 7) / 8;
    auto l_packed = [&]() {
        // packed v11 is M=1 specialist; for M>1 we'd loop M times
        for (int m = 0; m < M; ++m) {
            mm_v11_warp4<<<blocks, 256>>>(dX + m*K, dW_col, K, N, scale, dOut + m*N);
        }
    };
    ms_packed = time_it(l_packed, n_iters);

    // INT4 TC naive
    auto l_int4_naive = [&]() {
        if (M < 8 || M % 8 != 0 || N % 8 != 0 || K % 32 != 0) return;
        dim3 grid(N/8, M/8);
        mm_int4_tc_naive<<<grid, 32>>>(dA_pkd, dB_pkd, M, N, K, scale, dOut);
    };
    ms_int4_naive = (M>=8 && M%8==0 && N%8==0 && K%32==0) ? time_it(l_int4_naive, n_iters) : 1e30;

    // INT4 TC tiled
    auto l_int4_tiled = [&]() {
        if (M < 32 || M % 32 != 0 || N % 32 != 0 || K % 32 != 0) return;
        dim3 grid(N/32, M/32);
        mm_int4_tc_tiled<32,32,32><<<grid, 16*32>>>(dA_pkd, dB_pkd, M, N, K, scale, dOut);
    };
    ms_int4_tiled = (M>=32 && M%32==0 && N%32==0 && K%32==0) ? time_it(l_int4_tiled, n_iters) : 1e30;

    // cuBLAS INT8 TC reference
    const int32_t aI = 1, bI = 0;
    auto l_int8 = [&]() {
        CB(cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
            &aI, dW_i8, CUDA_R_8I, N, dX, CUDA_R_8I, K,
            &bI, dOut_i32, CUDA_R_32I, N,
            CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };
    ms_int8 = time_it(l_int8, n_iters);

    cudaFree(dX); cudaFree(dW_i8); cudaFree(dW_col);
    cudaFree(dA_pkd); cudaFree(dB_pkd);
    cudaFree(dOut); cudaFree(dOut_i32);
    return 0;
}

void run_model(const char* name, int M, Shape* shapes, int n_shapes, int n_iters,
               cublasHandle_t cublas)
{
    std::printf("\n========================= %s  M=%d  =========================\n", name, M);
    std::printf("%-10s %5s %6s %4s %12s %12s %12s %12s %14s\n",
        "shape", "K", "N", "reps",
        "packed ms", "INT4-N ms", "INT4-T ms", "INT8 ms", "best/INT8 (>1=win)");

    double tot_packed = 0, tot_int4_naive = 0, tot_int4_tiled = 0, tot_int8 = 0;
    double tot_hybrid = 0;
    int wins_v11 = 0, wins_int4_n = 0, wins_int4_t = 0, wins_int8 = 0;

    for (int i = 0; i < n_shapes; ++i) {
        auto& s = shapes[i];
        double mp, mn, mt, m8;
        bench_shape(M, s.K, s.N, n_iters, mp, mn, mt, m8, cublas);

        // pick best (excluding INT8 reference, but include INT8 as a fallback?
        // for hybrid we always pick best of all four)
        double choices[4] = {mp, mn, mt, m8};
        const char* names[4] = {"v11", "int4-n", "int4-t", "int8"};
        int best = 0;
        for (int k = 1; k < 4; ++k) if (choices[k] < choices[best]) best = k;
        double best_ms = choices[best];
        const char* best_name = names[best];

        std::printf("%-10s %6d %6d %4d %12.4f %12.4f %12.4f %12.4f %12.2fx (%s)\n",
            s.name, s.K, s.N, s.reps,
            mp, (mn>1e10?-1:mn), (mt>1e10?-1:mt), m8,
            m8/best_ms, best_name);

        tot_packed += mp * s.reps;
        if (mn < 1e10) tot_int4_naive += mn * s.reps; else tot_int4_naive = 1e30;
        if (mt < 1e10) tot_int4_tiled += mt * s.reps; else tot_int4_tiled = 1e30;
        tot_int8 += m8 * s.reps;
        tot_hybrid += best_ms * s.reps;

        if (best == 0) wins_v11++;
        else if (best == 1) wins_int4_n++;
        else if (best == 2) wins_int4_t++;
        else wins_int8++;
    }

    std::printf("\n%s forward equivalent (matmul fabric only, M=%d):\n", name, M);
    std::printf("  all-packed-v11   : %10.3f ms (1.00x baseline for packed)\n", tot_packed);
    if (tot_int4_naive < 1e20)
        std::printf("  all-INT4-TC naive: %10.3f ms (%.2fx vs INT8)\n", tot_int4_naive, tot_int8/tot_int4_naive);
    if (tot_int4_tiled < 1e20)
        std::printf("  all-INT4-TC tiled: %10.3f ms (%.2fx vs INT8)\n", tot_int4_tiled, tot_int8/tot_int4_tiled);
    std::printf("  all-cuBLAS INT8 TC: %10.3f ms (1.00x reference)\n", tot_int8);
    std::printf("  HYBRID dispatch  : %10.3f ms (%.2fx vs INT8 reference)\n", tot_hybrid, tot_int8/tot_hybrid);
    std::printf("  Hybrid wins: v11=%d, int4-naive=%d, int4-tiled=%d, int8=%d\n",
        wins_v11, wins_int4_n, wins_int4_t, wins_int8);
}

int main(int argc, char** argv) {
    int n_iters = (argc > 1) ? std::atoi(argv[1]) : 60;
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s, %d SMs (n_iters=%d)\n", prop.name, prop.multiProcessorCount, n_iters);
    cublasHandle_t cublas; CB(cublasCreate(&cublas));

    // Llama 3.2 1B: H=2048, F=8192, layers=16, Hkv=8*64=512, V=128256
    Shape mats_1b[] = {
        {"Wq",       2048,  2048, 16},
        {"Wk",       2048,   512, 16},
        {"Wv",       2048,   512, 16},
        {"Wo",       2048,  2048, 16},
        {"Wgate",    2048,  8192, 16},
        {"Wup",      2048,  8192, 16},
        {"Wdown",    8192,  2048, 16},
        {"lm_head",  2048, 128256, 1},
    };

    // Llama 3.1 8B: H=4096, F=14336, layers=32, Hkv=8*128=1024, V=128256
    Shape mats_8b[] = {
        {"Wq",       4096,  4096, 32},
        {"Wk",       4096,  1024, 32},
        {"Wv",       4096,  1024, 32},
        {"Wo",       4096,  4096, 32},
        {"Wgate",    4096, 14336, 32},
        {"Wup",      4096, 14336, 32},
        {"Wdown",   14336,  4096, 32},
        {"lm_head",  4096, 128256, 1},
    };

    // BitNet 2B-4T: H=2560, F=6912, layers=30, Hkv=5*128=640, V=128256
    // Shapes match microsoft/bitnet-b1.58-2B-4T GGUF, weight-tied lm_head
    // (treated as packed-trit for comparison purposes; in reality lm_head
    // uses tied F16 token_embd which our hybrid kernel can't dispatch on).
    Shape mats_bitnet[] = {
        {"Wq",       2560,  2560, 30},
        {"Wk",       2560,   640, 30},
        {"Wv",       2560,   640, 30},
        {"Wo",       2560,  2560, 30},
        {"Wgate",    2560,  6912, 30},
        {"Wup",      2560,  6912, 30},
        {"Wdown",    6912,  2560, 30},
    };

    run_model("Llama 3.2 1B", 1, mats_1b, 8, n_iters, cublas);
    run_model("Llama 3.2 1B", 16, mats_1b, 8, n_iters, cublas);
    run_model("Llama 3.1 8B", 1, mats_8b, 8, n_iters, cublas);
    run_model("Llama 3.1 8B", 16, mats_8b, 8, n_iters, cublas);
    run_model("BitNet 2B-4T", 1,  mats_bitnet, 7, n_iters, cublas);
    run_model("BitNet 2B-4T", 16, mats_bitnet, 7, n_iters, cublas);
    run_model("BitNet 2B-4T", 64, mats_bitnet, 7, n_iters, cublas);

    cublasDestroy(cublas);
    return 0;
}
