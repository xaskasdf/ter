#include <tersim/cache_model.hpp>
#include <algorithm>
#include <cassert>

namespace tersim {

Cache::Cache(CacheConfig cfg)
  : cfg_(cfg),
    sets_(cfg.total_bytes / (cfg.line_bytes * static_cast<std::size_t>(cfg.ways))),
    ways_(cfg.ways),
    tags_(sets_ * static_cast<std::size_t>(cfg.ways)) {
    assert(sets_ > 0 && "cache geometry: total_bytes / (line_bytes * ways) == 0");
    assert(cfg.line_bytes > 0);
}

bool Cache::access(std::uint64_t byte_addr) noexcept {
    const auto line_no = byte_addr / cfg_.line_bytes;
    const auto set_idx = line_no % sets_;
    const auto tag     = line_no / sets_;

    ++stats_.accesses;
    const std::size_t base = set_idx * static_cast<std::size_t>(ways_);

    // Linear scan over ways — N ≤ 8 in our configs, branch-predictable.
    int hit_way = -1;
    for (int w = 0; w < ways_; ++w) {
        if (tags_[base + static_cast<std::size_t>(w)].valid &&
            tags_[base + static_cast<std::size_t>(w)].tag == tag) {
            hit_way = w; break;
        }
    }

    if (hit_way >= 0) {
        tags_[base + static_cast<std::size_t>(hit_way)].lru_age = ++age_ctr_;
        ++stats_.hits;
        return true;
    }

    // Miss → evict LRU way (smallest lru_age, or any invalid).
    int victim = 0;
    for (int w = 0; w < ways_; ++w) {
        const auto& wy = tags_[base + static_cast<std::size_t>(w)];
        if (!wy.valid) { victim = w; break; }
        if (wy.lru_age < tags_[base + static_cast<std::size_t>(victim)].lru_age) victim = w;
    }
    auto& v = tags_[base + static_cast<std::size_t>(victim)];
    v.tag = tag; v.valid = true; v.lru_age = ++age_ctr_;
    ++stats_.misses;
    return false;
}

MemHierarchy::MemHierarchy(MemHierarchyConfig cfg)
  : cfg_(cfg), l1d_(cfg.l1d), l2_(cfg.l2) {}

int MemHierarchy::access(std::uint64_t byte_addr, bool is_load) noexcept {
    return access_at(byte_addr, is_load, 0).latency_cycles;
}

MemAccessResult MemHierarchy::access_at(std::uint64_t byte_addr, bool /*is_load*/,
                                        std::uint64_t proposed_issue_cycle) noexcept {
    MemAccessResult r{};
    r.actual_issue_cycle = proposed_issue_cycle;

    // Prune expired MSHRs (already serviced before our proposed issue).
    if (!mshr_free_cycles_.empty()) {
        const auto end = std::remove_if(mshr_free_cycles_.begin(), mshr_free_cycles_.end(),
                                        [&](auto c) { return c <= proposed_issue_cycle; });
        mshr_free_cycles_.erase(end, mshr_free_cycles_.end());
    }

    if (l1d_.access(byte_addr)) {
        stats_.l1d = l1d_.stats();
        r.latency_cycles = l1d_.hit_latency();
        return r;
    }

    // L1D miss → need an MSHR.
    if (cfg_.mshrs > 0 && static_cast<int>(mshr_free_cycles_.size()) >= cfg_.mshrs) {
        // Stall to when the oldest in-flight miss completes.
        const auto oldest = *std::min_element(mshr_free_cycles_.begin(),
                                              mshr_free_cycles_.end());
        stats_.mshr_stall_cycles += (oldest - r.actual_issue_cycle);
        r.actual_issue_cycle = oldest;
        // Re-prune everything ≤ new issue.
        const auto end = std::remove_if(mshr_free_cycles_.begin(), mshr_free_cycles_.end(),
                                        [&](auto c) { return c <= r.actual_issue_cycle; });
        mshr_free_cycles_.erase(end, mshr_free_cycles_.end());
    }

    int latency;
    if (l2_.access(byte_addr)) {
        latency = l2_.hit_latency();
    } else {
        ++stats_.dram_accesses;
        latency = cfg_.dram_latency;
    }
    stats_.l1d = l1d_.stats();
    stats_.l2  = l2_.stats();

    // Allocate MSHR for the duration of this miss.
    mshr_free_cycles_.push_back(r.actual_issue_cycle + static_cast<std::uint64_t>(latency));
    r.latency_cycles = latency;
    return r;
}

}  // namespace tersim
