#pragma once
#include <ter/word.hpp>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace ter {

class Sim;

struct KernelId {
    size_t entry_addr = 0;
    bool   valid = false;
};

// Flat-array kernel table: fixed kSlots entries, linear search by name.
// Replaces std::unordered_map<std::string, KernelId> -- one fewer libstdc++
// dependency for the K4 build path. Kernel count is small (~5 today, headroom
// up to kSlots = 32) so linear search is fine.
class KernelTable {
public:
    static constexpr size_t kSlots    = 32;
    static constexpr size_t kNameMax  = 32;

    KernelId install(Sim& sim, const std::string& name, const std::vector<Word27>& blob);
    KernelId find(const std::string& name) const noexcept;
    KernelId find(const char* name) const noexcept;

private:
    struct Entry {
        char     name[kNameMax]{};   // null-terminated
        KernelId id;
        bool     used = false;
    };
    std::array<Entry, kSlots> entries_{};
    size_t next_addr_ = 0;
};

void install_default_kernels(Sim& sim, KernelTable& kt, const std::string& kernels_dir);

}
