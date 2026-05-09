#pragma once
#include <ter/word.hpp>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace ter {

class Sim;

struct KernelId {
    size_t entry_addr = 0;
    bool   valid = false;
};

class KernelTable {
public:
    KernelId install(Sim& sim, const std::string& name, const std::vector<Word27>& blob);
    KernelId find(const std::string& name) const noexcept;

private:
    std::unordered_map<std::string, KernelId> by_name_;
    size_t next_addr_ = 0;
};

void install_default_kernels(Sim& sim, KernelTable& kt, const std::string& kernels_dir);

}
