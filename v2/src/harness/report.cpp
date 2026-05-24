#include <tersim/report.hpp>
#include <ostream>

namespace tersim {

namespace {

// Tiny JSON escape: backslashes and double-quotes only (no Unicode handling
// needed for our content).
void write_json_string(std::ostream& os, const std::string& s) {
    os << '"';
    for (char c : s) {
        if (c == '\\' || c == '"') os << '\\';
        os << c;
    }
    os << '"';
}

const char* fu_name(FuKind k) {
    switch (k) {
        case FuKind::Scalar:  return "scalar";
        case FuKind::Vector:  return "vector";
        case FuKind::Memory:  return "memory";
        case FuKind::Branch:  return "branch";
        case FuKind::Control: return "control";
    }
    return "unknown";
}

const char* node_name(TechNode n) {
    switch (n) {
        case TechNode::N22nm: return "22nm";
        case TechNode::N7nm:  return "7nm";
        case TechNode::N3nm:  return "3nm";
    }
    return "unknown";
}

const char* path_name(GpuMathPath p) {
    switch (p) {
        case GpuMathPath::Dp4a:    return "dp4a";
        case GpuMathPath::Int8Tc:  return "int8_tc";
        case GpuMathPath::Int4Tc:  return "int4_tc";
        case GpuMathPath::AddOnly: return "add_only";
    }
    return "unknown";
}

}  // namespace

void emit_json(std::ostream& os, const Report& r) {
    os << "{\n";

    os << "  \"metadata\": {\n";
    os << "    \"timestamp\": ";    write_json_string(os, r.meta.timestamp);    os << ",\n";
    os << "    \"workload\": ";     write_json_string(os, r.meta.workload_name); os << ",\n";
    os << "    \"host_device\": ";  write_json_string(os, r.meta.host_device);  os << ",\n";
    os << "    \"op_count_total\": " << r.meta.op_count_total << ",\n";
    os << "    \"iters\": "          << r.meta.iters          << "\n";
    os << "  },\n";

    os << "  \"cpu_pipeline\": {\n";
    os << "    \"cycles_total\": " << r.cpu.cycles_total << ",\n";
    os << "    \"insns_total\": "  << r.cpu.insns_total  << ",\n";
    os << "    \"ipc\": "          << r.cpu.ipc()        << ",\n";
    os << "    \"mshr_stall_cycles\": " << r.cpu.mshr_stall_cycles << ",\n";
    os << "    \"bp_penalty_cycles\": " << r.cpu.bp_penalty_cycles << ",\n";
    os << "    \"fu_mix\": {\n";
    for (int i = 0; i < kFuKindCount; ++i) {
        os << "      \"" << fu_name(static_cast<FuKind>(i)) << "\": "
           << r.cpu.insns_by_fu[static_cast<std::size_t>(i)];
        if (i + 1 < kFuKindCount) os << ",";
        os << "\n";
    }
    os << "    }\n";
    os << "  },\n";

    os << "  \"memory\": {\n";
    os << "    \"l1d_accesses\": " << r.mem.l1d.accesses << ",\n";
    os << "    \"l1d_hits\": "     << r.mem.l1d.hits     << ",\n";
    os << "    \"l1d_misses\": "   << r.mem.l1d.misses   << ",\n";
    os << "    \"l1d_hit_rate\": " << r.mem.l1d.hit_rate() << ",\n";
    os << "    \"l2_accesses\": "  << r.mem.l2.accesses  << ",\n";
    os << "    \"l2_hits\": "      << r.mem.l2.hits      << ",\n";
    os << "    \"l2_hit_rate\": "  << r.mem.l2.hit_rate() << ",\n";
    os << "    \"dram_accesses\": " << r.mem.dram_accesses << ",\n";
    os << "    \"mshr_stall_cycles\": " << r.mem.mshr_stall_cycles << "\n";
    os << "  },\n";

    os << "  \"gpu_projection\": {\n";
    os << "    \"path\": ";   write_json_string(os, path_name(r.gpu_desc.path)); os << ",\n";
    os << "    \"macs\": "         << r.gpu_desc.macs         << ",\n";
    os << "    \"mem_bytes\": "    << r.gpu_desc.mem_load_bytes + r.gpu_desc.mem_store_bytes << ",\n";
    os << "    \"batch_m\": "      << r.gpu_desc.batch_m      << ",\n";
    os << "    \"compute_time_ms\": " << r.gpu.compute_time_s * 1e3 << ",\n";
    os << "    \"memory_time_ms\": "  << r.gpu.memory_time_s  * 1e3 << ",\n";
    os << "    \"wall_time_ms\": "    << r.gpu.wall_time_s    * 1e3 << ",\n";
    os << "    \"memory_bound\": "    << (r.gpu.memory_bound ? "true" : "false") << ",\n";
    os << "    \"bw_utilization\": "  << r.gpu.bw_utilization << ",\n";
    os << "    \"tops_achieved\": "   << r.gpu.tops_achieved  << "\n";
    os << "  },\n";

    os << "  \"asic_projection\": {\n";
    // node name is derived from the workload at projection time; caller-supplied.
    os << "    \"node\": ";  write_json_string(os, node_name(TechNode::N22nm)); os << ",\n";
    os << "    \"total_energy_pJ_per_token\": " << r.asic.total_energy_pJ_per_token << ",\n";
    os << "    \"perf_per_watt_tokps\": "       << r.asic.perf_per_watt_tokps       << ",\n";
    os << "    \"area_mm2\": "                  << r.asic.area_mm2                  << ",\n";
    os << "    \"freq_GHz\": "                  << r.asic.freq_GHz                  << ",\n";
    os << "    \"wall_time_s_per_token\": "     << r.asic.wall_time_s_per_token     << ",\n";
    os << "    \"breakdown\": {\n";
    os << "      \"compute_pJ\": " << r.asic.breakdown.compute_pJ << ",\n";
    os << "      \"memory_pJ\": "  << r.asic.breakdown.memory_pJ  << ",\n";
    os << "      \"leakage_pJ\": " << r.asic.breakdown.leakage_pJ << "\n";
    os << "    }\n";
    os << "  }";

    if (r.correctness.has_golden) {
        const auto& c = r.correctness;
        auto write_int_array = [&](const std::vector<int>& v) {
            os << "[";
            for (std::size_t i = 0; i < v.size(); ++i) {
                os << v[i];
                if (i + 1 < v.size()) os << ", ";
            }
            os << "]";
        };
        os << ",\n  \"correctness\": {\n";
        os << "    \"tokens_generated\": " << c.tokens_generated << ",\n";
        os << "    \"tokens_matched\": "   << c.tokens_matched   << ",\n";
        os << "    \"match_fraction\": "
           << (c.tokens_generated ? static_cast<double>(c.tokens_matched) / c.tokens_generated : 0.0)
           << ",\n";
        os << "    \"generated\": "; write_int_array(c.generated); os << ",\n";
        os << "    \"golden\": ";    write_int_array(c.golden);    os << "\n";
        os << "  }\n";
    } else {
        os << "\n";
    }

    os << "}\n";
}

void emit_csv_header(std::ostream& os) {
    os << "workload,cycles,insns,ipc,l1d_hit_rate,dram_accesses,"
       << "gpu_path,gpu_wall_ms,gpu_bw_util,"
       << "asic_pJ_per_tok,asic_perf_per_watt_tokps,asic_freq_GHz\n";
}

void emit_csv_row(std::ostream& os, const Report& r) {
    os << r.meta.workload_name << ","
       << r.cpu.cycles_total   << ","
       << r.cpu.insns_total    << ","
       << r.cpu.ipc()          << ","
       << r.mem.l1d.hit_rate() << ","
       << r.mem.dram_accesses  << ","
       << path_name(r.gpu_desc.path) << ","
       << r.gpu.wall_time_s * 1e3 << ","
       << r.gpu.bw_utilization << ","
       << r.asic.total_energy_pJ_per_token << ","
       << r.asic.perf_per_watt_tokps       << ","
       << r.asic.freq_GHz                  << "\n";
}

}  // namespace tersim
