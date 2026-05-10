#ifndef TER_K4_H
#define TER_K4_H

// Pure C interface for embedding the ter ternary CPU simulator as a ring-0
// kernel module in osito-k. The implementation lives in src/k4/ter_k4.cpp
// and links against the same Sim + KernelTable used by the host tests.
//
// Lifecycle (typical osito-k caller):
//   ter_k4_handle_t* h = ter_k4_create(64 * 1024);          // sim-memory words
//   ter_k4_install_kernel(h, "tk_matmul_b_9t", blob, n);    // load each kernel
//   ter_k4_call(h, "tk_matmul_b_9t", args, n_args);
//   ter_k4_op_counts(h, &out_counts);                       // for op-count metric
//   ter_k4_destroy(h);
//
// FREESTANDING NOTE: this header is C-clean and stable. The implementation
// currently uses libstdc++ (std::vector, std::unordered_map). For ring-0
// linkage in osito-k, src/k4/ter_k4.cpp must be re-targeted to a freestanding
// allocator + a flat-array kernel table. Tracked in the F10 follow-up.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ter_k4_handle_t ter_k4_handle_t;

ter_k4_handle_t* ter_k4_create(size_t sim_memory_words);
void             ter_k4_destroy(ter_k4_handle_t* h);

// Install a kernel blob (Word27 instructions) under a name. Returns 0 on success.
int ter_k4_install_kernel(ter_k4_handle_t* h,
                          const char* name,
                          const void* blob_bytes,
                          size_t blob_n_words);

// Invoke a kernel by name. args is an array of int64_t; n_args <= 7 fits the
// existing R1..R7 calling convention. Returns 0 on success.
int ter_k4_call(ter_k4_handle_t* h,
                const char* name,
                const int64_t* args,
                size_t n_args);

// Op-counter snapshot for the F6.4 metric.
typedef struct {
    uint64_t total_ops;
    uint64_t tvmac;
    uint64_t tvadd;
    uint64_t tvsub;
    uint64_t tvmul;
} ter_k4_op_counts_t;

void ter_k4_op_counts(const ter_k4_handle_t* h, ter_k4_op_counts_t* out);
void ter_k4_reset_counters(ter_k4_handle_t* h);

#ifdef __cplusplus
}
#endif

#endif  // TER_K4_H
