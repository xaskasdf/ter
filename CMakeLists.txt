cmake_minimum_required(VERSION 3.20)
project(ter LANGUAGES CXX)

# Allow FetchContent deps (e.g. doctest 2.4.11) that declare cmake_minimum_required < 3.5
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

include(FetchContent)
FetchContent_Declare(doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG v2.4.11)
FetchContent_MakeAvailable(doctest)

add_library(ter)
# -O3 + AVX2/FMA on the ter library only (tests stay at default Debug). The
# inner GEMV in src/tx/forward.cpp::mm_row is now structured for compiler
# auto-vectorization; -O3 turns the int32*int64 inner loop into vpmuldq +
# vpaddq sequences. AVX-512 not required (i9-9880H lacks it; AVX2 + FMA is
# universal on x86_64 since Haswell 2013).
target_compile_options(ter PRIVATE
    -Wall -Wextra -Wpedantic -Werror
    -O3 -mavx2 -mfma)
target_include_directories(ter PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_subdirectory(src)
add_subdirectory(vendor/ntransformer)

# Only register tests when ter is the top-level project (not when consumed
# via add_subdirectory from v2/ or other downstream).
if(CMAKE_PROJECT_NAME STREQUAL "ter")
    enable_testing()
    add_subdirectory(tests)
endif()

add_executable(dump_gguf_tensors tools/dump_gguf_tensors.cpp)
target_link_libraries(dump_gguf_tensors PRIVATE nt_infra)
target_compile_options(dump_gguf_tensors PRIVATE -Wall -Wextra -Wpedantic -Werror)

# F10: K4 ring-0 backend.
# Wrapper layer (src/k4/ter_k4.cpp) is freestanding-friendly: no exceptions,
# no RTTI, no std::vector/string in the public-facing code. The wrapped Sim
# and KernelTable still use libstdc++ internally; full ring-0 boot requires
# porting those (tracked in docs/k4-freestanding.md).
add_library(ter_k4 STATIC src/k4/ter_k4.cpp)
target_compile_options(ter_k4 PRIVATE
    -Wall -Wextra -Wpedantic -Werror
    -fno-exceptions -fno-rtti)
target_compile_definitions(ter_k4 PRIVATE TER_FREESTANDING)
# Memory and RegFile bounds violations call std::abort() instead of throwing
# under TER_FREESTANDING (see headers). Hosted-mode tests still get the
# throw path so doctest can introspect failures.
target_include_directories(ter_k4 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ter_k4 PUBLIC ter)
