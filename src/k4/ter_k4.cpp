// F10.x: K4 ring-0 backend wrapper, freestanding-friendly at the wrapper layer.
//
// Compiled with -fno-exceptions -fno-rtti and using only raw arrays/new[]
// instead of std::vector/std::unordered_map. The wrapped Sim and KernelTable
// still use libstdc++ internally (std::vector for memory, std::unordered_map
// for counter buckets and kernel lookup); see docs/k4-freestanding.md for the
// concrete porting checklist required to boot in osito-k ring 0.
#include <ter_k4/ter_k4.h>
#include <ter/sim.hpp>
#include <ter/kernels.hpp>
#include <ter/isa.hpp>
#include <new>

struct ter_k4_handle_t {
    ter::Sim sim;
    ter::KernelTable kt;
    explicit ter_k4_handle_t(size_t words) : sim(words) {}
};

extern "C" {

ter_k4_handle_t* ter_k4_create(size_t sim_memory_words) {
    return new (std::nothrow) ter_k4_handle_t(sim_memory_words);
}

void ter_k4_destroy(ter_k4_handle_t* h) { delete h; }

int ter_k4_install_kernel(ter_k4_handle_t* h,
                          const char* name,
                          const void* blob_bytes,
                          size_t blob_n_words)
{
    if (!h || !name || !blob_bytes) return -1;
    // Stage the blob into a freshly-allocated Word27 buffer (no std::vector
    // copy ctor pull-in at this layer). KernelTable::install still copies
    // through a vector internally; that's the libstdc++ gap.
    ter::Word27* tmp = new (std::nothrow) ter::Word27[blob_n_words];
    if (!tmp) return -3;
    const ter::Word27* blob = static_cast<const ter::Word27*>(blob_bytes);
    for (size_t i = 0; i < blob_n_words; ++i) tmp[i] = blob[i];
    std::vector<ter::Word27> v(tmp, tmp + blob_n_words);
    delete[] tmp;
    h->kt.install(h->sim, name, v);
    return 0;
}

int ter_k4_call(ter_k4_handle_t* h,
                const char* name,
                const int64_t* args,
                size_t n_args)
{
    if (!h || !name) return -1;
    if (n_args > 7) return -4;  // calling convention only routes R1..R7
    auto id = h->kt.find(name);
    if (!id.valid) return -2;
    std::vector<int64_t> arg_vec(args, args + n_args);
    h->sim.call_kernel(h->kt, id, arg_vec);
    return 0;
}

void ter_k4_op_counts(const ter_k4_handle_t* h, ter_k4_op_counts_t* out) {
    if (!h || !out) return;
    const auto& c = h->sim.counters();
    out->total_ops = c.total();
    out->tvmac    = c.get(ter::Opcode::TVMAC);
    out->tvadd    = c.get(ter::Opcode::TVADD);
    out->tvsub    = c.get(ter::Opcode::TVSUB);
    out->tvmul    = c.get(ter::Opcode::TVMUL);
}

void ter_k4_reset_counters(ter_k4_handle_t* h) {
    if (h) h->sim.counters().reset();
}

}  // extern "C"
