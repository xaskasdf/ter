#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace tersim {

struct CacheConfig {
    std::size_t total_bytes      = 32 * 1024;
    std::size_t line_bytes       = 64;
    int         ways             = 8;
    int         hit_latency      = 3;   // pipeline cycles when this level hits
    const char* name             = "L1D";
};

struct CacheStats {
    std::uint64_t hits     = 0;
    std::uint64_t misses   = 0;
    std::uint64_t accesses = 0;
    double hit_rate() const noexcept {
        return accesses ? static_cast<double>(hits) / accesses : 0.0;
    }
};

// Set-associative cache with strict LRU. Models tag-state only (no data);
// suitable for hit/miss + latency accounting in a single-thread pipeline.
class Cache {
public:
    explicit Cache(CacheConfig cfg);

    // Returns true on hit (regardless of latency). Updates LRU on hit and
    // installs+evicts on miss.
    bool access(std::uint64_t byte_addr) noexcept;

    int hit_latency() const noexcept { return cfg_.hit_latency; }
    const CacheStats& stats() const noexcept { return stats_; }
    const CacheConfig& config() const noexcept { return cfg_; }

private:
    struct Way { std::uint64_t tag = 0; std::uint64_t lru_age = 0; bool valid = false; };
    CacheConfig cfg_;
    std::size_t sets_;
    int         ways_;
    std::vector<Way> tags_;     // sets_ * ways_, row-major: tags_[set*ways + way]
    std::uint64_t   age_ctr_ = 0;
    CacheStats      stats_;
};

struct MemHierarchyConfig {
    CacheConfig l1d = []{
        CacheConfig c; c.total_bytes = 32 * 1024;  c.ways = 8;
        c.hit_latency = 3;  c.name = "L1D"; return c;
    }();
    CacheConfig l2 = []{
        CacheConfig c; c.total_bytes = 512 * 1024; c.ways = 8;
        c.hit_latency = 12; c.name = "L2";  return c;
    }();
    int dram_latency = 80;       // miss-to-DRAM total latency in pipeline cycles
    int mshrs        = 0;        // 0 = unlimited (G4 behavior); 8 = realistic OoO front-end
};

struct MemHierarchyStats {
    CacheStats    l1d;
    CacheStats    l2;
    std::uint64_t dram_accesses = 0;
    std::uint64_t mshr_stall_cycles = 0;
};

// Returned by MemHierarchy::access — pipeline applies both fields.
struct MemAccessResult {
    int           latency_cycles    = 0;   // total mem-stage cycles incl base hit
    std::uint64_t actual_issue_cycle = 0;  // may be > proposed if MSHR-stalled
};

// G4+G5: data-side hierarchy with MSHR cap. Instruction fetch is currently
// perfect (L1I-hit every cycle); modeled when we add front-end stalls.
class MemHierarchy {
public:
    explicit MemHierarchy(MemHierarchyConfig cfg = {});

    // Legacy API (G4): assumes unlimited MSHRs. Returns hit/miss latency only.
    int access(std::uint64_t byte_addr, bool is_load) noexcept;

    // G5: MSHR-aware. Returns latency AND the adjusted issue cycle (pushed
    // forward if all MSHRs busy). When cfg.mshrs == 0, actual_issue_cycle
    // == proposed_issue_cycle (no cap).
    MemAccessResult access_at(std::uint64_t byte_addr, bool is_load,
                              std::uint64_t proposed_issue_cycle) noexcept;

    const MemHierarchyStats& stats() const noexcept { return stats_; }
    const MemHierarchyConfig& config() const noexcept { return cfg_; }

private:
    MemHierarchyConfig cfg_;
    Cache l1d_;
    Cache l2_;
    MemHierarchyStats stats_;
    // Sorted-ascending free cycles of in-flight L1D misses. Hits don't allocate.
    // N ≤ cfg.mshrs (typically ≤ 8); linear ops are fine.
    std::vector<std::uint64_t> mshr_free_cycles_;
};

}  // namespace tersim
