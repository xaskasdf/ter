#include <ter/sim.hpp>

namespace ter {

void Sim::load_lut(size_t addr, const std::vector<int>& values) {
    for (size_t i = 0; i < values.size(); ++i) {
        mem_.store_word(addr + i, Word27::from_int(values[i]));
    }
}

}
