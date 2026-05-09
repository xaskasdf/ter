#pragma once

#include "types.h"

namespace nt {

// ============================================================
// CPU-only device tag
// CUDA entries stripped for vendor/CPU-only build.
// ============================================================

inline const char* device_name(Device d) {
    switch (d) {
        case Device::CPU: return "CPU";
        default:          return "unknown";
    }
}

} // namespace nt
