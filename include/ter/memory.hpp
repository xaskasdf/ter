#pragma once
#include <ter/word.hpp>
#include <ter/pack.hpp>
#include <vector>
#include <cstdint>
#include <stdexcept>

namespace ter {

class Memory {
public:
    explicit Memory(size_t n_words) : data_(n_words, 0) {}

    Memory(const Memory&) = delete;
    Memory& operator=(const Memory&) = delete;
    Memory(Memory&&) = default;
    Memory& operator=(Memory&&) = default;

    size_t size_words() const noexcept { return data_.size(); }

    Word27 load_word(size_t addr) const {
        if (addr >= data_.size()) throw std::out_of_range("Memory::load_word");
        return unpack_word27(data_[addr]);
    }

    void store_word(size_t addr, Word27 w) {
        if (addr >= data_.size()) throw std::out_of_range("Memory::store_word");
        data_[addr] = pack_word27(w);
    }

    uint64_t* raw() noexcept { return data_.data(); }
    const uint64_t* raw() const noexcept { return data_.data(); }

private:
    std::vector<uint64_t> data_;
};

}
