#pragma once
// G6-GPU: roofline model for the existing CUDA backends. Replaces the planned
// FPGA layer (no hardware available); the RTX 3090 box IS available on LAN, so
// we can cross-validate predictions against measured throughput.
//
// Philosophy: same as Layer 1 CPU substrate, but for a GPU "DUT". The model
// takes a per-kernel description (MACs, mem bytes, peak op type) and returns
// time/throughput/utilization. Calibrated against ~/ter/docs/baselines/*.md.
#include <cstdint>

namespace tersim {

enum class GpuMathPath {
    Dp4a    = 0,   // packed-trit / INT8 path via __dp4a (no tensor cores)
    Int8Tc  = 1,   // INT8 tensor cores (cuBLAS GemmEx baseline)
    Int4Tc  = 2,   // INT4 tensor cores (2× INT8 TC peak)
    AddOnly = 3,   // BitNet ADD-only — no MUL, treats compute as cheap
};

struct GpuConfig {
    // RTX 3090 (Ampere SM86) defaults. Override for A100/H100/etc.
    const char* name           = "RTX 3090";
    int         sm_count       = 82;
    double      boost_clock_ghz = 1.70;

    // Peak compute, TOPS. Sources: NVIDIA Ampere whitepaper + measurements.
    double dp4a_peak_tops      = 71.0;     // ~71 TOPS INT8 dp4a aggregate
    double int8_tc_peak_tops   = 284.0;    // INT8 tensor cores
    double int4_tc_peak_tops   = 568.0;    // INT4 tensor cores (2× INT8)
    double addonly_peak_tops   = 142.0;    // half-bandwidth proxy (TVMAC=0 path
                                            // is bw-bound, compute irrelevant)

    // Memory: 936 GB/s peak HBM (GDDR6X 24GB 384-bit @ 19 Gbps).
    double hbm_peak_bw_gbps    = 936.0;

    // Achieved-fraction-of-peak at M=1, per math path. Modeled as a saturating
    // curve in the weight-bytes domain (kernel launch + per-call overhead
    // dominates at small sizes; HBM peak asymptote at large sizes):
    //
    //   eff(B) = eff_asymptote * B / (B + bytes_half)
    //
    // Calibrated against measured RTX 3090 numbers across 5 Llama 1B shapes
    // (cuda/results_{v11_v12,v13,int4tc_real,hybrid}_2026-05-17.csv).
    // Verified by test_gpu_roofline: predictions within ±25% of measurement
    // across all 4 paths × 5 shapes.
    //
    // Path             | eff_asymptote | bytes_half (MiB) | source
    // -----------------+---------------+------------------+------------------
    // Int8Tc (cuBLAS)  |  0.45         |  4.0             | 21% @ 1.3 GB Llama1B
    // Dp4a (packed v11)|  0.42         |  6.0             | 3% small → 38% lm_head
    // Int4Tc (v13)     |  0.95         |  3.0             | 5% small → 88% lm_head
    // AddOnly (BitNet) |  0.20         |  6.0             | 6% @ 335 MB Llama1B fwd
    double bw_eff_asymptote_int8tc  = 0.45;
    double bw_eff_asymptote_dp4a    = 0.42;
    double bw_eff_asymptote_int4tc  = 0.95;
    double bw_eff_asymptote_addonly = 0.20;
    double bw_eff_bytes_half_int8tc  =  4.0 * (1ull << 20);
    double bw_eff_bytes_half_dp4a    =  4.0 * (1ull << 20);   // retuned
    double bw_eff_bytes_half_int4tc  = 10.0 * (1ull << 20);   // retuned (steeper curve)
    double bw_eff_bytes_half_addonly =  6.0 * (1ull << 20);
    double compute_efficiency        = 0.50;
};

// Per-kernel description: what work and what data motion.
struct GpuKernelDesc {
    std::uint64_t macs            = 0;     // multiply-accumulates issued
    std::uint64_t mem_load_bytes  = 0;     // unique bytes read from HBM
    std::uint64_t mem_store_bytes = 0;
    GpuMathPath   path            = GpuMathPath::Dp4a;
    int           batch_m         = 1;     // M dimension of GEMM (M=1 GEMV most common)
    const char*   label           = "kernel";
};

struct GpuReport {
    double compute_time_s     = 0.0;
    double memory_time_s      = 0.0;
    double wall_time_s        = 0.0;
    bool   memory_bound       = true;
    double bw_utilization     = 0.0;        // achieved / peak
    double tops_achieved      = 0.0;
};

class GpuModel {
public:
    explicit GpuModel(GpuConfig cfg = {}) noexcept : cfg_(cfg) {}

    // Predict roofline performance for one kernel invocation.
    GpuReport project(const GpuKernelDesc& d) const noexcept;

    // Predict tokens/sec given total per-forward MACs/bytes (sum across all
    // matmuls + norms + attention + softmax). Useful for Llama 1B / BitNet
    // end-to-end calibration.
    double tokens_per_second(const GpuKernelDesc& per_forward) const noexcept;

    const GpuConfig& config() const noexcept { return cfg_; }

private:
    double peak_tops_for(GpuMathPath p) const noexcept;
    double effective_bw_gbps(int batch_m, GpuMathPath p,
                              std::uint64_t mem_bytes) const noexcept;

    GpuConfig cfg_;
};

}  // namespace tersim
