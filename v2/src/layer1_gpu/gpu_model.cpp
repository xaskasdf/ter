#include <tersim/gpu_model.hpp>
#include <algorithm>

namespace tersim {

double GpuModel::peak_tops_for(GpuMathPath p) const noexcept {
    switch (p) {
        case GpuMathPath::Dp4a:    return cfg_.dp4a_peak_tops;
        case GpuMathPath::Int8Tc:  return cfg_.int8_tc_peak_tops;
        case GpuMathPath::Int4Tc:  return cfg_.int4_tc_peak_tops;
        case GpuMathPath::AddOnly: return cfg_.addonly_peak_tops;
    }
    return cfg_.dp4a_peak_tops;
}

double GpuModel::effective_bw_gbps(int batch_m, GpuMathPath path,
                                    std::uint64_t mem_bytes) const noexcept {
    // Saturating curve: eff(B) = eff_asym * B / (B + bytes_half).
    // Captures the launch-overhead → HBM-peak regime crossover empirically
    // observed in the v11/v13_int4tc bench across Llama 1B shapes.
    double eff_asym  = cfg_.bw_eff_asymptote_dp4a;
    double bytes_half = cfg_.bw_eff_bytes_half_dp4a;
    switch (path) {
        case GpuMathPath::Int8Tc:
            eff_asym = cfg_.bw_eff_asymptote_int8tc;
            bytes_half = cfg_.bw_eff_bytes_half_int8tc; break;
        case GpuMathPath::Dp4a:
            eff_asym = cfg_.bw_eff_asymptote_dp4a;
            bytes_half = cfg_.bw_eff_bytes_half_dp4a;   break;
        case GpuMathPath::Int4Tc:
            eff_asym = cfg_.bw_eff_asymptote_int4tc;
            bytes_half = cfg_.bw_eff_bytes_half_int4tc; break;
        case GpuMathPath::AddOnly:
            eff_asym = cfg_.bw_eff_asymptote_addonly;
            bytes_half = cfg_.bw_eff_bytes_half_addonly; break;
    }
    const double B = static_cast<double>(mem_bytes);
    double eff_m1 = (B + bytes_half > 0.0) ? eff_asym * B / (B + bytes_half) : 0.0;

    // Large M (≥16) lets cuBLAS amortize → 65% HBM peak floor.
    if (batch_m >= 16) return cfg_.hbm_peak_bw_gbps * std::max(eff_m1, 0.65);
    if (batch_m <= 1)  return cfg_.hbm_peak_bw_gbps * eff_m1;
    const double t = static_cast<double>(batch_m - 1) / 15.0;
    const double frac = eff_m1 + t * (0.65 - eff_m1);
    return cfg_.hbm_peak_bw_gbps * frac;
}

GpuReport GpuModel::project(const GpuKernelDesc& d) const noexcept {
    GpuReport r;

    // Memory time = total bytes moved / effective bandwidth.
    const auto total_bytes = d.mem_load_bytes + d.mem_store_bytes;
    const auto eff_bw_Bs   = effective_bw_gbps(d.batch_m, d.path, total_bytes) * 1e9;
    r.memory_time_s = (eff_bw_Bs > 0.0) ? static_cast<double>(total_bytes) / eff_bw_Bs : 0.0;

    // Compute time = MACs / (peak_tops * efficiency). 2 ops per MAC for INT8
    // (1 mul + 1 add), but TOPS specs already count both, so MACs/peak_ops.
    if (d.path == GpuMathPath::AddOnly) {
        // ADD-only: each non-zero ternary weight = 1 add (or skip for zero).
        // Compute is effectively free vs memory; we still bound it.
        const auto peak_ops_s = peak_tops_for(d.path) * 1e12 * cfg_.compute_efficiency;
        r.compute_time_s = (peak_ops_s > 0.0)
                               ? static_cast<double>(d.macs) / peak_ops_s : 0.0;
    } else {
        const auto peak_ops_s = peak_tops_for(d.path) * 1e12 * cfg_.compute_efficiency;
        // 2 ops per MAC (multiply + accumulate) ≈ 1 TOPS-MAC; vendor specs
        // already report "ops" counting both. Use macs directly vs ops/s.
        r.compute_time_s = (peak_ops_s > 0.0)
                               ? 2.0 * static_cast<double>(d.macs) / peak_ops_s : 0.0;
    }

    r.wall_time_s    = std::max(r.compute_time_s, r.memory_time_s);
    r.memory_bound   = (r.memory_time_s >= r.compute_time_s);
    r.bw_utilization = (r.wall_time_s > 0.0)
                           ? (total_bytes / r.wall_time_s) / (cfg_.hbm_peak_bw_gbps * 1e9)
                           : 0.0;
    r.tops_achieved  = (r.wall_time_s > 0.0)
                           ? 2.0 * static_cast<double>(d.macs) / r.wall_time_s / 1e12
                           : 0.0;
    return r;
}

double GpuModel::tokens_per_second(const GpuKernelDesc& per_forward) const noexcept {
    const auto rep = project(per_forward);
    return (rep.wall_time_s > 0.0) ? 1.0 / rep.wall_time_s : 0.0;
}

}  // namespace tersim
