// tersim_run — unified harness for the ter ternary substrate simulator.
//
// Two workloads:
//   --workload kernel  : run one bytecode kernel (default tk_matmul_b_9t)
//                        through the full Layer-1 stack.
//   --workload forward : run ONE transformer layer of a model preset through
//                        the Sim (faithful trace), then extrapolate ×n_layers
//                        + lm_head for a full-forward estimate.
//
// All workloads project onto Layer-1 GPU (roofline) and Layer-3 ASIC
// (energy/area) and emit a unified JSON or CSV report.
//
// Usage:
//   tersim_run [--workload kernel|forward]
//              [--kernel name] [--K n] [--N n] [--iters n]      (kernel mode)
//              [--model tiny|llama1b|bitnet2b]                  (forward mode)
//              [--out path] [--format json|csv]
//              [--asic-node 22nm|7nm|3nm]

#include <ter/sim.hpp>
#include <ter/kernels.hpp>
#include <ter/word.hpp>
#include <ter/tx/forward.hpp>
#include <ter/tx/layer.hpp>
#include <ter/tx/lut_setup.hpp>
#include <ter/tx/transformer.hpp>
#include <model/loader.h>
#include <inference/sampler.h>
#include <tersim/bp_model.hpp>
#include <tersim/cache_model.hpp>
#include <tersim/cycle_model.hpp>
#include <tersim/asic_model.hpp>
#include <tersim/gpu_model.hpp>
#include <tersim/report.hpp>
#include <tersim/trace.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifndef TER_KERNELS_DIR
#error "TER_KERNELS_DIR must be defined by the build system"
#endif
#ifndef TER_LUT_DIR
#error "TER_LUT_DIR must be defined by the build system"
#endif

using namespace ter;
using namespace tersim;

namespace {

std::string iso_now() {
    auto t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
    return buf;
}

TechNode parse_node(const std::string& s) {
    if (s == "22nm") return TechNode::N22nm;
    if (s == "7nm")  return TechNode::N7nm;
    if (s == "3nm")  return TechNode::N3nm;
    std::fprintf(stderr, "WARN: unknown --asic-node '%s', using 22nm\n", s.c_str());
    return TechNode::N22nm;
}

struct Args {
    std::string  workload  = "kernel";
    std::string  kernel    = "tk_matmul_b_9t";
    int          K         = 27;
    int          N         = 1;
    int          iters     = 1;
    std::string  model     = "tiny";
    std::string  out_path  = "";
    std::string  format    = "json";
    std::string  asic_node = "22nm";
    // generate workload
    std::string  model_path = "/Users/pc/osito-a-models/downloads/"
                              "llama-3.2-1b-instruct/llama-3.2-1b-instruct-q8_0.gguf";
    int          n_gen      = 16;
    int          n_trits    = 9;     // Format B precision (9 = high; 1 ≈ BitNet-style)
    int          block_size = 0;     // 0 = per-tensor scale; 32 = Q8_0-style per-block
    bool         fp_bypass  = false; // skip activation quant, float matmul (diagnostic)
    bool         bitnet     = false; // ternarize weights to {-1,0,+1} (absmean), matches reference
    bool         fp32_weights = false; // store weights as true fp32 (reference mode; implies fp_bypass)
    std::string  golden     = "";    // empty → use built-in Llama Q8_0 golden
};

// Llama 3.2 1B Q8_0 golden tokens (llama-cli, BOS=128000, greedy, 16 tokens):
//   " Tags: 2019, 2020, 2021, ..."  (source: tools/llama_pyfwd.py)
const std::vector<int> kLlamaQ8Golden = {
    28783, 25, 220, 679, 24, 11, 220, 2366,
    15, 11, 220, 2366, 16, 11, 220, 2366
};

// Transformer dims for forward workload.
struct ModelDims {
    const char* name;
    int H;        // hidden
    int HD;       // head_dim
    int Hn;       // n_heads
    int Kn;       // n_kv_heads
    int I;        // intermediate
    int layers;
    int vocab;
};

ModelDims dims_for(const std::string& m) {
    if (m == "llama1b")
        return {"llama1b", 2048, 64, 32, 8, 8192, 16, 128256};
    if (m == "bitnet2b")
        return {"bitnet2b", 2560, 128, 20, 5, 6912, 30, 128256};
    // tiny default: fast smoke (~ms on the interpreted Sim).
    return {"tiny", 64, 32, 2, 1, 128, 4, 512};
}

void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s [--workload kernel|forward]\n"
        "          [--kernel name] [--K n] [--N n] [--iters n]\n"
        "          [--model tiny|llama1b|bitnet2b]\n"
        "          [--out path] [--format json|csv] [--asic-node 22nm|7nm|3nm]\n",
        prog);
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto eat = [&](const std::string& name, std::string& dst) {
            if (k == name && i + 1 < argc) { dst = argv[++i]; return true; }
            return false;
        };
        auto eat_int = [&](const std::string& name, int& dst) {
            std::string v;
            if (eat(name, v)) { dst = std::atoi(v.c_str()); return true; }
            return false;
        };
        if (eat("--workload", a.workload))     continue;
        if (eat("--kernel", a.kernel))         continue;
        if (eat("--model",  a.model))          continue;
        if (eat("--out",    a.out_path))       continue;
        if (eat("--format", a.format))         continue;
        if (eat("--asic-node", a.asic_node))   continue;
        if (eat("--model-path", a.model_path)) continue;
        if (eat("--golden", a.golden))         continue;
        if (eat_int("--K", a.K))               continue;
        if (eat_int("--N", a.N))               continue;
        if (eat_int("--iters", a.iters))       continue;
        if (eat_int("--n-gen", a.n_gen))       continue;
        if (eat_int("--n-trits", a.n_trits))   continue;
        if (eat_int("--block-size", a.block_size)) continue;
        if (k == "--fp-bypass") { a.fp_bypass = true; continue; }
        if (k == "--bitnet")    { a.bitnet    = true; continue; }
        if (k == "--fp32-weights") { a.fp32_weights = true; a.fp_bypass = true; continue; }
        if (k == "-h" || k == "--help") { print_usage(argv[0]); std::exit(0); }
        std::fprintf(stderr, "ERROR: unknown arg '%s'\n", k.c_str());
        print_usage(argv[0]);
        std::exit(1);
    }
    return a;
}

// Build a tracer lambda that records into the given pipeline.
auto make_tracer(PipelineModel& pipe) {
    return [&pipe](std::uint64_t pc, const Instr& instr, const Sim& sim) {
        InstrTrace t;
        t.pc   = pc;
        t.op   = instr.op;
        t.dst  = static_cast<std::uint8_t>(instr.dst);
        t.src1 = static_cast<std::uint8_t>(instr.src1);
        t.src2 = static_cast<std::uint8_t>(instr.src2);
        t.imm  = instr.imm;
        switch (instr.op) {
            case Opcode::TLOAD: case Opcode::TVLOAD:
                t.mem_addr = static_cast<std::uint64_t>(
                    sim.regs().read_scalar(instr.src1).to_int()) * 8;
                t.mem_is_load = true;  break;
            case Opcode::TSTORE: case Opcode::TVSTORE:
                t.mem_addr = static_cast<std::uint64_t>(
                    sim.regs().read_scalar(instr.src2).to_int()) * 8;
                t.mem_is_store = true; break;
            default: break;
        }
        const auto next_pc = static_cast<std::uint64_t>(sim.regs().pc().to_int());
        t.branch_taken  = (next_pc != pc + 1);
        t.branch_target = t.branch_taken ? next_pc : pc + 1;
        pipe.on_retire(t);
    };
}

// ----- kernel workload -----------------------------------------------------
Report run_kernel_workload(const Args& args) {
    Sim s(8192);
    KernelTable kt;
    install_default_kernels(s, kt, TER_KERNELS_DIR);
    KernelId id = kt.find(args.kernel);
    if (!id.valid) {
        std::fprintf(stderr, "ERROR: kernel '%s' not found\n", args.kernel.c_str());
        std::exit(2);
    }

    constexpr int xa = 200, wa = 600, ya = 1000;
    for (int i = 0; i < args.K; ++i) {
        s.mem().store_word((std::size_t)(xa + i), Word27::from_int(((i * 7) % 3) - 1));
        s.mem().store_word((std::size_t)(wa + i), Word27::from_int(((i * 11) % 3) - 1));
    }

    PipelineConfig pcfg; PipelineModel pipe(pcfg);
    MemHierarchyConfig mc; mc.mshrs = 8; MemHierarchy mh(mc);
    BpConfig bcfg; bcfg.ghr_history_bits = 0; BranchPredictor bp(bcfg);
    pipe.set_mem_hierarchy(&mh);
    pipe.set_branch_predictor(&bp);
    s.set_tracer(make_tracer(pipe));

    for (int i = 0; i < args.iters; ++i) {
        std::vector<std::int64_t> kargs = {xa, wa, ya, 0, 0, 0, 0};
        s.call_kernel(kt, id, kargs);
    }

    GpuKernelDesc gd;
    gd.macs           = (std::uint64_t)args.K * args.N;
    gd.mem_load_bytes = (std::uint64_t)args.K * args.N / 4;
    gd.path = GpuMathPath::Dp4a; gd.batch_m = 1; gd.label = args.kernel.c_str();
    GpuModel gpu; auto gpu_rep = gpu.project(gd);

    AsicWorkload aw;
    aw.counters = &s.counters();
    aw.cycles_total  = pipe.report().cycles_total;
    aw.l1d_accesses  = mh.stats().l1d.accesses;
    aw.l2_accesses   = mh.stats().l2.accesses;
    aw.dram_accesses = mh.stats().dram_accesses;
    aw.tokens        = (std::uint64_t)args.iters;
    AsicModel am(AsicModel::defaults_for(parse_node(args.asic_node)));
    auto asic_rep = am.project(aw);

    Report r;
    r.meta.timestamp = iso_now();
    r.meta.workload_name = args.kernel;
    r.meta.host_device = "MacBook i9-9880H";
    r.meta.op_count_total = s.counters().total();
    r.meta.iters = args.iters;
    r.cpu = pipe.report(); r.mem = mh.stats(); r.gpu = gpu_rep; r.asic = asic_rep;
    r.gpu_desc = gd; r.asic_workload = aw;
    return r;
}

// ----- forward workload ----------------------------------------------------
Report run_forward_workload(const Args& args) {
    using namespace ter::tx;
    const ModelDims d = dims_for(args.model);

    // Synthesize random fp32 weights for one layer, quantize to ternary.
    std::mt19937 rng(0xBEEF);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    auto randvec = [&](int n) {
        std::vector<float> v(static_cast<std::size_t>(n));
        for (auto& x : v) x = dist(rng);
        return v;
    };
    const int kv = d.Kn * d.HD;
    auto Wq = randvec(d.H * d.H), Wk = randvec(d.H * kv), Wv = randvec(d.H * kv);
    auto Wo = randvec(d.H * d.H);
    auto Wg = randvec(d.H * d.I), Wu = randvec(d.H * d.I), Wd = randvec(d.I * d.H);
    std::vector<float> nw1(static_cast<std::size_t>(d.H), 1.0f);
    std::vector<float> nw2(static_cast<std::size_t>(d.H), 1.0f);

    LayerWeights L = quantize_layer(
        Wq.data(), d.H, d.H,  Wk.data(), d.H, kv,  Wv.data(), d.H, kv,
        Wo.data(), d.H, d.H,  Wg.data(), d.H, d.I, Wu.data(), d.H, d.I,
        Wd.data(), d.I, d.H,  nw1.data(), d.H, nw2.data(), d.H);

    // Large mem pool — Llama 1B layer touches a lot of scratch.
    Sim s(1 << 22);
    KernelTable kt;
    install_default_kernels(s, kt, TER_KERNELS_DIR);
    LutAddrs luts = load_default_luts(s, TER_LUT_DIR);

    KVCache cache;
    cache.resize(/*max_seq*/8, d.Kn, d.HD);

    PipelineConfig pcfg; PipelineModel pipe(pcfg);
    MemHierarchyConfig mc; mc.mshrs = 8; MemHierarchy mh(mc);
    BpConfig bcfg; bcfg.ghr_history_bits = 0; BranchPredictor bp(bcfg);
    pipe.set_mem_hierarchy(&mh);
    pipe.set_branch_predictor(&bp);
    s.set_tracer(make_tracer(pipe));

    std::vector<float> hidden_in = randvec(d.H), hidden_out;
    std::fprintf(stderr, "[tersim_run] running 1 %s layer through Sim "
                 "(H=%d I=%d heads=%d/%d)...\n", d.name, d.H, d.I, d.Hn, d.Kn);
    forward_layer(s, kt, L, cache, hidden_in, /*pos*/0,
                  d.H, d.HD, d.Hn, d.Kn, d.I, 1e-6f, 10000.0, false, luts, hidden_out);

    // Per-layer measured; extrapolate to full forward (× layers + lm_head).
    const auto per_layer = pipe.report();

    // Full-forward MACs/bytes (analytical): all linear projections × layers
    // + lm_head. Used to drive GPU + ASIC projections for the whole model.
    const std::uint64_t macs_per_layer =
        (std::uint64_t)d.H * d.H            // Wq
      + (std::uint64_t)d.H * kv * 2         // Wk + Wv
      + (std::uint64_t)d.H * d.H            // Wo
      + (std::uint64_t)d.H * d.I * 2        // Wgate + Wup
      + (std::uint64_t)d.I * d.H;           // Wdown
    const std::uint64_t macs_lm_head = (std::uint64_t)d.H * d.vocab;
    const std::uint64_t macs_full = macs_per_layer * d.layers + macs_lm_head;

    GpuKernelDesc gd;
    gd.macs           = macs_full;
    gd.mem_load_bytes = macs_full / 4;        // packed-trit 4 trits/byte
    gd.path = GpuMathPath::Dp4a; gd.batch_m = 1; gd.label = d.name;
    GpuModel gpu; auto gpu_rep = gpu.project(gd);

    // ASIC: scale measured per-layer op counts to the full model. We approximate
    // by projecting the per-layer counters and multiplying energy by layer count
    // (+ lm_head fraction). For first-order this captures the dominant compute.
    AsicWorkload aw;
    aw.counters = &s.counters();
    aw.cycles_total  = per_layer.cycles_total;
    aw.l1d_accesses  = mh.stats().l1d.accesses;
    aw.l2_accesses   = mh.stats().l2.accesses;
    aw.dram_accesses = mh.stats().dram_accesses;
    aw.tokens        = 1;
    AsicModel am(AsicModel::defaults_for(parse_node(args.asic_node)));
    auto asic_rep_layer = am.project(aw);
    // Scale per-token energy to the full model (× layers, + lm_head ≈ +1 proj).
    const double layer_scale = (double)d.layers
        + (double)macs_lm_head / (double)macs_per_layer;
    auto asic_rep = asic_rep_layer;
    asic_rep.total_energy_pJ_per_token *= layer_scale;
    asic_rep.breakdown.compute_pJ *= layer_scale;
    asic_rep.breakdown.memory_pJ  *= layer_scale;
    asic_rep.breakdown.leakage_pJ *= layer_scale;

    Report r;
    r.meta.timestamp = iso_now();
    r.meta.workload_name = std::string("forward:") + d.name +
                           " (1 layer measured ×" + std::to_string(d.layers) +
                           " + lm_head, extrapolated)";
    r.meta.host_device = "MacBook i9-9880H";
    r.meta.op_count_total = s.counters().total();
    r.meta.iters = 1;
    r.cpu = per_layer; r.mem = mh.stats(); r.gpu = gpu_rep; r.asic = asic_rep;
    r.gpu_desc = gd; r.asic_workload = aw;
    return r;
}

std::vector<int> parse_golden(const std::string& s) {
    std::vector<int> out;
    std::size_t i = 0;
    while (i < s.size()) {
        std::size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        std::string tok = s.substr(i, j - i);
        if (!tok.empty()) out.push_back(std::atoi(tok.c_str()));
        i = j + 1;
    }
    return out;
}

// ----- generate workload (golden-token correctness) ------------------------
Report run_generate_workload(const Args& args) {
    using namespace ter::tx;

    std::ifstream check(args.model_path);
    if (!check.good()) {
        std::fprintf(stderr, "ERROR: model not found: %s\n", args.model_path.c_str());
        std::exit(2);
    }

    nt::GGUFLoader loader;
    if (!loader.load(args.model_path)) {
        std::fprintf(stderr, "ERROR: failed to load GGUF %s\n", args.model_path.c_str());
        std::exit(2);
    }
    const int max_seq = args.n_gen + 2;
    // Auto-detect architecture: BitNet GGUFs report "bitnet" and need the
    // bitnet loader (sub-norms + ReLU² FFN). Native i2_s weights are already
    // ternary, so no requant — the substrate runs them exactly.
    const std::string arch = loader.config().architecture;
    const bool is_bitnet = (arch.find("bitnet") != std::string::npos)
                        || (arch.find("BitNet") != std::string::npos);
    BrandonTransformer tx = is_bitnet
        ? load_bitnet_transformer(loader, max_seq, args.n_trits)
        : load_llama_transformer(loader, max_seq, args.n_trits, /*format_a*/false,
            /*mant_trits*/9, /*bitnet*/args.bitnet, /*block_size*/args.block_size,
            /*store_fp32*/args.fp32_weights);
    std::fprintf(stderr, "[tersim_run] arch=%s loader=%s\n", arch.c_str(),
                 is_bitnet ? "bitnet" : "llama");

    ter::tx::set_forward_fp_bypass(args.fp_bypass);

    Sim s(64 * 1024);
    KernelTable kt;
    install_default_kernels(s, kt, TER_KERNELS_DIR);
    LutAddrs luts = load_default_luts(s, TER_LUT_DIR);

    int bos = loader.config().bos_token_id > 0 ? loader.config().bos_token_id : 128000;
    if (bos >= tx.vocab_size) bos = 128000;

    PipelineConfig pcfg; PipelineModel pipe(pcfg);
    MemHierarchyConfig mc; mc.mshrs = 8; MemHierarchy mh(mc);
    BpConfig bcfg; bcfg.ghr_history_bits = 0; BranchPredictor bp(bcfg);
    pipe.set_mem_hierarchy(&mh);
    pipe.set_branch_predictor(&bp);
    s.set_tracer(make_tracer(pipe));

    std::vector<int> golden = args.golden.empty() ? kLlamaQ8Golden
                                                  : parse_golden(args.golden);

    std::fprintf(stderr,
        "[tersim_run] greedy generate: model=%s n_trits=%d BOS=%d n_gen=%d\n"
        "             (real-weight forward through the Sim; ~25s/token on host)\n",
        args.model_path.c_str(), args.n_trits, bos, args.n_gen);

    std::vector<int> gen;
    int pos = 0, cur = bos;
    for (int g = 0; g < args.n_gen; ++g) {
        auto logits = forward_token(s, kt, tx, cur, pos, luts, /*state=*/nullptr);
        int next = nt::Sampler::argmax(logits.data(), tx.vocab_size);
        if (g == 0) {
            // Dump top-5 logits at pos=0 to compare against the FP reference.
            std::vector<int> idx(static_cast<std::size_t>(tx.vocab_size));
            for (int v = 0; v < tx.vocab_size; ++v) idx[static_cast<std::size_t>(v)] = v;
            std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                [&](int a, int b){ return logits[static_cast<std::size_t>(a)]
                                        > logits[static_cast<std::size_t>(b)]; });
            std::fprintf(stderr, "  pos=0 top5:");
            for (int t = 0; t < 5; ++t)
                std::fprintf(stderr, " %d(%.3f)", idx[static_cast<std::size_t>(t)],
                             logits[static_cast<std::size_t>(idx[static_cast<std::size_t>(t)])]);
            std::fprintf(stderr, "\n");
        }
        gen.push_back(next);
        std::fprintf(stderr, "  pos=%d cur=%d -> next=%d%s\n", pos, cur, next,
            (g < static_cast<int>(golden.size()) && next == golden[static_cast<std::size_t>(g)])
                ? "  [match]" : "");
        cur = next; ++pos;
    }

    int matched = 0;
    for (std::size_t i = 0; i < gen.size() && i < golden.size(); ++i)
        if (gen[i] == golden[i]) ++matched;

    // GPU/ASIC projections for the full forward (per-token totals × n_gen).
    const int H = tx.hidden_size, I = tx.intermediate_size, V = tx.vocab_size;
    const int kv = tx.n_kv_heads * tx.head_dim;
    const std::uint64_t macs_per_layer =
        (std::uint64_t)H * H + (std::uint64_t)H * kv * 2 + (std::uint64_t)H * H
      + (std::uint64_t)H * I * 2 + (std::uint64_t)I * H;
    const std::uint64_t macs_per_tok =
        macs_per_layer * (std::uint64_t)tx.n_layers + (std::uint64_t)H * V;

    GpuKernelDesc gd;
    gd.macs           = macs_per_tok;
    gd.mem_load_bytes = macs_per_tok / 4;
    gd.path = GpuMathPath::Dp4a; gd.batch_m = 1; gd.label = "llama1b-generate";
    GpuModel gpu; auto gpu_rep = gpu.project(gd);

    AsicWorkload aw;
    aw.counters = &s.counters();
    aw.cycles_total  = pipe.report().cycles_total;
    aw.l1d_accesses  = mh.stats().l1d.accesses;
    aw.l2_accesses   = mh.stats().l2.accesses;
    aw.dram_accesses = mh.stats().dram_accesses;
    aw.tokens        = (std::uint64_t)args.n_gen;
    AsicModel am(AsicModel::defaults_for(parse_node(args.asic_node)));
    auto asic_rep = am.project(aw);

    Report r;
    r.meta.timestamp = iso_now();
    r.meta.workload_name = "generate:llama1b n_trits=" + std::to_string(args.n_trits)
                         + " block_size=" + std::to_string(args.block_size);
    r.meta.host_device = "MacBook i9-9880H";
    r.meta.op_count_total = s.counters().total();
    r.meta.iters = args.n_gen;
    r.cpu = pipe.report(); r.mem = mh.stats(); r.gpu = gpu_rep; r.asic = asic_rep;
    r.gpu_desc = gd; r.asic_workload = aw;
    r.correctness.has_golden       = true;
    r.correctness.tokens_generated = static_cast<int>(gen.size());
    r.correctness.tokens_matched   = matched;
    r.correctness.generated        = gen;
    r.correctness.golden           = golden;

    std::fprintf(stderr, "[tersim_run] golden match: %d/%d tokens\n",
                 matched, static_cast<int>(gen.size()));
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    const auto args = parse_args(argc, argv);

    Report r;
    if (args.workload == "forward")       r = run_forward_workload(args);
    else if (args.workload == "generate") r = run_generate_workload(args);
    else                                  r = run_kernel_workload(args);

    auto emit = [&](std::ostream& os) {
        if (args.format == "csv") { emit_csv_header(os); emit_csv_row(os, r); }
        else                      { emit_json(os, r); }
    };
    if (args.out_path.empty()) {
        emit(std::cout);
    } else {
        std::ofstream f(args.out_path);
        if (!f) { std::fprintf(stderr, "ERROR: cannot open '%s'\n", args.out_path.c_str()); return 3; }
        emit(f);
        std::fprintf(stderr, "[tersim_run] wrote %s\n", args.out_path.c_str());
    }
    return 0;
}
