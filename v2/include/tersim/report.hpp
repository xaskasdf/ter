#pragma once
// Unified report struct emitted by the tersim harness. Aggregates the four
// modeling layers: functional/op counts (from ter::Sim), CPU pipeline timing
// (Layer 1 CPU), GPU roofline projection (Layer 1 GPU), and ASIC energy/area
// (Layer 3). Emits JSON and CSV.
#include <tersim/asic_model.hpp>
#include <tersim/cache_model.hpp>
#include <tersim/cycle_model.hpp>
#include <tersim/gpu_model.hpp>

#include <ostream>
#include <string>
#include <vector>

namespace tersim {

struct WorkloadMeta {
    std::string  timestamp;       // ISO-8601
    std::string  workload_name;
    std::string  host_device;     // e.g. "MacBook i9-9880H"
    std::uint64_t op_count_total = 0;   // from ter::OpCounters::total()
    int          iters           = 1;
};

// Golden-token correctness for the `generate` workload. The simulator runs a
// real model forward and compares the greedy token stream against a reference
// (e.g. llama-cli Q8_0 golden). Present only for generate workloads.
struct CorrectnessInfo {
    bool             has_golden       = false;
    int              tokens_generated = 0;
    int              tokens_matched   = 0;   // prefix match vs golden
    std::vector<int> generated;
    std::vector<int> golden;
};

struct Report {
    WorkloadMeta            meta;
    PipelineReport          cpu;
    MemHierarchyStats       mem;
    GpuReport               gpu;       // single shape projection
    AsicReport              asic;
    // GPU/ASIC inputs preserved for traceability.
    GpuKernelDesc           gpu_desc;
    AsicWorkload            asic_workload;
    CorrectnessInfo         correctness;   // populated by generate workload
};

// Emit unified JSON to a stream. Schema: human-readable, key sections
// `metadata`, `cpu_pipeline`, `memory`, `gpu_projection`, `asic_projection`.
void emit_json(std::ostream& os, const Report& r);

// Single-row CSV (header + data) flattening the most useful scalars.
// Useful for sweeping multiple configs and aggregating later.
void emit_csv_header(std::ostream& os);
void emit_csv_row(std::ostream& os, const Report& r);

}  // namespace tersim
