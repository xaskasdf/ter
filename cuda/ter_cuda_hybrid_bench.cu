// ter_cuda_hybrid_bench.cu
//
// End-to-end bench for the v13-hybrid dispatch (v11 dp4a + v13 INT4 TC).
// For each Llama 1B GEMV shape, times three configs:
//   1. pure_v11        — always v11 dp4a
//   2. pure_v13_int4tc — always v13 INT4 Tensor Cores
//   3. hybrid_v13      — pick_kernel(K, N) decides per shape
// Then prints a weighted summary using the Llama 1B 16-layer call count.
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_86 -Xptxas -O3 \
//        ter_cuda_hybrid_bench.cu -o hbench

// Suppress the v13 file's own main() so we can reuse its kernels & helpers.
#define V13_INT4TC_NO_MAIN
#include "ter_cuda_forward_packed_v13_int4tc.cu"

// Pull in the hybrid wrapper (v11 inlined + pick + launcher).
#include "ter_cuda_hybrid_dispatch_v13.cu"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

// ---------------------------------------------------------------------------
// Mini correctness test: K=64 N=16 (hits v11 path since N<4096) AND
// K=64 N=8192 (hits v13 path). For both, compare hybrid output to a
// CPU int reference (scale=1 so half == integer accumulator).
// ---------------------------------------------------------------------------
static int hybrid_correctness_one(int K, int N) {
    // Deterministic packed-trit W (col-major 4 trits/byte) and X in [-7,7].
    std::vector<uint8_t> hW((size_t)K * (N / 4), 0);
    std::vector<int8_t>  hW_trits((size_t)K * N);
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            int t_code = (k * 7 + n * 13) % 3;        // 0,1,2
            int t_sgn  = (t_code == 1) ? +1 : (t_code == 2 ? -1 : 0);
            hW_trits[(size_t)k * N + n] = (int8_t)t_sgn;
            int j_byte = n / 4;
            int sub    = n % 4;
            hW[(size_t)j_byte * K + k] |= (uint8_t)(t_code << (sub * 2));
        }
    }
    std::vector<int8_t> hX(K);
    for (int k = 0; k < K; ++k) hX[k] = (int8_t)((k * 11 + 3) % 15 - 7);

    // CPU int reference.
    std::vector<int32_t> ref(N, 0);
    for (int n = 0; n < N; ++n) {
        int32_t acc = 0;
        for (int k = 0; k < K; ++k)
            acc += (int32_t)hX[k] * (int32_t)hW_trits[(size_t)k * N + n];
        ref[n] = acc;
    }

    HybridWeights HW = prepack_hybrid_weights(hW.data(), K, N);
    int8_t* dX = nullptr; __half* dY = nullptr; float* dS = nullptr;
    HCK(cudaMalloc(&dX, K));
    HCK(cudaMalloc(&dY, N * sizeof(__half)));
    HCK(cudaMalloc(&dS, sizeof(float)));
    HCK(cudaMemcpy(dX, hX.data(), K, cudaMemcpyHostToDevice));
    float one = 1.f;
    HCK(cudaMemcpy(dS, &one, sizeof(float), cudaMemcpyHostToDevice));

    launch_hybrid_matmul(dX, HW, dS, 1.f, dY);
    HCK(cudaDeviceSynchronize());

    std::vector<__half> hY(N);
    HCK(cudaMemcpy(hY.data(), dY, N * sizeof(__half), cudaMemcpyDeviceToHost));

    int n_mismatch = 0;
    for (int n = 0; n < N; ++n) {
        int32_t got = (int32_t)__half2float(hY[n]);
        if (got != ref[n]) {
            if (n_mismatch < 4)
                std::fprintf(stderr,
                    "[hybrid_correctness K=%d N=%d] MISMATCH n=%d got=%d ref=%d\n",
                    K, N, n, got, ref[n]);
            n_mismatch++;
        }
    }
    cudaFree(dX); cudaFree(dY); cudaFree(dS);
    free_hybrid_weights(HW);
    if (g_x_int4_buf) {
        cudaFree(g_x_int4_buf);
        g_x_int4_buf = nullptr; g_x_int4_buf_bytes = 0;
    }

    HybridKernel chosen = pick_kernel(K, N);
    const char* tag = (chosen == HybridKernel::V13_INT4TC) ? "v13_int4tc" : "v11_dp4a";
    if (n_mismatch == 0) {
        std::fprintf(stderr,
            "[hybrid_correctness] PASS K=%d N=%d -> %s (bit-exact)\n", K, N, tag);
        return 0;
    } else {
        std::fprintf(stderr,
            "[hybrid_correctness] FAIL K=%d N=%d -> %s (%d mismatches)\n",
            K, N, tag, n_mismatch);
        return 1;
    }
}

static int hybrid_correctness_suite() {
    // K=64 N=16  -> pick_kernel returns V11_DP4A (N<4096)
    // K=64 N=8192 -> pick_kernel returns V13_INT4TC (N>=4096)
    // Both must produce ref-exact output.
    int rc = 0;
    rc |= hybrid_correctness_one(64, 16);
    rc |= hybrid_correctness_one(64, 8192);
    return rc;
}

// ---------------------------------------------------------------------------
// Bench harness.
// ---------------------------------------------------------------------------
struct ShapeRow { const char* name; int K, N; };

enum class Cfg { PURE_V11, PURE_V13, HYBRID };

static const char* cfg_name(Cfg c) {
    switch (c) {
        case Cfg::PURE_V11: return "pure_v11";
        case Cfg::PURE_V13: return "pure_v13_int4tc";
        case Cfg::HYBRID:   return "hybrid_v13";
    }
    return "?";
}

static double bench_one(Cfg cfg, const ShapeRow& s, const HybridWeights& HW,
                        const int8_t* dX, const float* dS, __half* dY,
                        int warmup, int iters)
{
    auto launch = [&]() {
        if (cfg == Cfg::PURE_V11) {
            launch_v11_hyb(dX, HW.W_packed, HW.K, HW.N, dS, 0.05f, dY);
        } else if (cfg == Cfg::PURE_V13) {
            launch_mm_packed_v13_int4tc(
                dX, reinterpret_cast<const uint8_t*>(HW.W_int4),
                HW.K, HW.N, dS, 0.05f, dY);
        } else {
            launch_hybrid_matmul(dX, HW, dS, 0.05f, dY);
        }
    };

    for (int i = 0; i < warmup; ++i) launch();
    HCK(cudaDeviceSynchronize());

    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    std::vector<double> times; times.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(a);
        launch();
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        times.push_back(ms);
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

int main(int argc, char** argv) {
    cudaDeviceProp prop; HCK(cudaGetDeviceProperties(&prop, 0));
    std::fprintf(stderr, "Device: %s  (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    if (prop.major < 8) {
        std::fprintf(stderr, "ERROR: INT4 mma.m8n8k32 requires sm_80+\n");
        return 2;
    }

    // Correctness first.
    if (hybrid_correctness_suite() != 0) {
        std::fprintf(stderr, "Hybrid correctness FAILED — aborting bench.\n");
        return 3;
    }

    int iters  = (argc > 1) ? std::atoi(argv[1]) : 200;
    int warmup = (argc > 2) ? std::atoi(argv[2]) :  20;

    // Peak HBM via CUDA 13 attribute API (fallback 936 GB/s RTX 3090).
    int mhz_khz = 0, bus_bits = 0;
    cudaError_t e1 = cudaDeviceGetAttribute(&mhz_khz,  cudaDevAttrMemoryClockRate, 0);
    cudaError_t e2 = cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, 0);
    double peak_gbs =
        (e1 == cudaSuccess && e2 == cudaSuccess && mhz_khz > 0 && bus_bits > 0)
        ? 2.0 * (double)mhz_khz * 1e3 * ((double)bus_bits / 8.0) / 1e9
        : 936.0;
    std::fprintf(stderr, "[bench] peak HBM ~%.0f GB/s\n", peak_gbs);

    ShapeRow shapes[] = {
        {"K=2048_N=2048",   2048,   2048},  // Wq, Wo  (per layer, x2)
        {"K=2048_N=512",    2048,    512},  // Wk, Wv  (per layer, x2)
        {"K=2048_N=8192",   2048,   8192},  // Wgate, Wup (per layer, x2)
        {"K=8192_N=2048",   8192,   2048},  // Wdown  (per layer, x1)
        {"K=2048_N=128256", 2048, 128256},  // lm_head (once)
    };

    // Llama 1B per-token call count for each shape (16 transformer layers).
    // Per layer: 2 x Wq/Wo  + 2 x Wk/Wv  + 2 x Wgate/Wup  + 1 x Wdown.
    // lm_head: once per token.
    const int LAYERS = 16;
    int calls[5] = {
        2 * LAYERS,   // K=2048_N=2048   (Wq, Wo)
        2 * LAYERS,   // K=2048_N=512    (Wk, Wv)
        2 * LAYERS,   // K=2048_N=8192   (Wgate, Wup)
        1 * LAYERS,   // K=8192_N=2048   (Wdown)
        1             // K=2048_N=128256 (lm_head)
    };

    std::printf("config,shape,ms_median,gb_per_s,bw_eff_percent\n");

    double sum_v11    = 0.0;
    double sum_v13    = 0.0;
    double sum_hybrid = 0.0;

    for (int si = 0; si < (int)(sizeof(shapes)/sizeof(shapes[0])); ++si) {
        const ShapeRow& s = shapes[si];
        int K = s.K, N = s.N;

        // Allocate inputs.
        std::vector<int8_t>  hX(K);
        for (int k = 0; k < K; ++k) hX[k] = (int8_t)((k % 15) - 7);
        std::vector<uint8_t> hW_packed((size_t)K * (N / 4), 0xA5);

        int8_t* dX = nullptr; __half* dY = nullptr; float* dS = nullptr;
        HCK(cudaMalloc(&dX, K));
        HCK(cudaMalloc(&dY, N * sizeof(__half)));
        HCK(cudaMalloc(&dS, sizeof(float)));
        HCK(cudaMemcpy(dX, hX.data(), K, cudaMemcpyHostToDevice));
        float sx = 0.05f;
        HCK(cudaMemcpy(dS, &sx, sizeof(float), cudaMemcpyHostToDevice));

        // Pre-repack BOTH layouts once (outside timed region).
        HybridWeights HW = prepack_hybrid_weights(hW_packed.data(), K, N);

        // Weight bytes for bandwidth calc:
        //   v11   : K*(N/4) bytes (packed-trit)
        //   v13   : K*(N/2) bytes (INT4)
        //   hybrid: same as whichever kernel pick_kernel chose
        double bytes_v11 = (double)K * (N / 4);
        double bytes_v13 = (double)K * (N / 2);

        double ms_v11    = bench_one(Cfg::PURE_V11, s, HW, dX, dS, dY, warmup, iters);
        double ms_v13    = bench_one(Cfg::PURE_V13, s, HW, dX, dS, dY, warmup, iters);
        double ms_hybrid = bench_one(Cfg::HYBRID,   s, HW, dX, dS, dY, warmup, iters);

        HybridKernel chosen = pick_kernel(K, N);
        double bytes_hybrid =
            (chosen == HybridKernel::V13_INT4TC) ? bytes_v13 : bytes_v11;

        auto emit = [&](Cfg c, double ms, double bytes) {
            double gbs = bytes / ms / 1e6;
            double eff = gbs / peak_gbs * 100.0;
            std::printf("%s,%s,%.4f,%.1f,%.2f\n",
                        cfg_name(c), s.name, ms, gbs, eff);
        };
        emit(Cfg::PURE_V11, ms_v11,    bytes_v11);
        emit(Cfg::PURE_V13, ms_v13,    bytes_v13);
        emit(Cfg::HYBRID,   ms_hybrid, bytes_hybrid);

        sum_v11    += ms_v11    * calls[si];
        sum_v13    += ms_v13    * calls[si];
        sum_hybrid += ms_hybrid * calls[si];

        free_hybrid_weights(HW);
        cudaFree(dX); cudaFree(dY); cudaFree(dS);
        if (g_x_int4_buf) {
            cudaFree(g_x_int4_buf);
            g_x_int4_buf = nullptr; g_x_int4_buf_bytes = 0;
        }
    }

    double speedup_v13    = sum_v11 / sum_v13;
    double speedup_hybrid = sum_v11 / sum_hybrid;
    std::printf(
        "[summary] pure_v11=%.4fms pure_v13=%.4fms hybrid_v13=%.4fms "
        "speedup_v13=%.3fx speedup_hybrid=%.3fx\n",
        sum_v11, sum_v13, sum_hybrid, speedup_v13, speedup_hybrid);
    std::fprintf(stderr,
        "[summary] Llama 1B fabric/token (16 layers + lm_head):\n"
        "  pure_v11        = %.4f ms/token\n"
        "  pure_v13_int4tc = %.4f ms/token  (%.3fx vs v11)\n"
        "  hybrid_v13      = %.4f ms/token  (%.3fx vs v11)\n",
        sum_v11, sum_v13, speedup_v13, sum_hybrid, speedup_hybrid);
    return 0;
}
