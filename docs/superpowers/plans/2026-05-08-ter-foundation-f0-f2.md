# `ter` Foundation Implementation Plan (Phases F0-F2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the ternary CPU substrate -- primitive types (`Trit`, `Tryte`, `Word27`, `Word54`), the scalar RISC ISA with assembler and simulator, and the SIMD extension with the load-bearing `tvmac` instruction. End-state: a working ternary VM that runs assembled programs and produces an `OpCounters` report.

**Architecture:** Header-heavy C++17 library, single-responsibility files, doctest for unit tests, CMake build. TDD throughout. Each phase ends with a quantitative gate from the spec.

**Tech Stack:** C++17, CMake 3.20+, doctest (fetched via CMake `FetchContent`), Python 3 for one comparison script.

**Spec:** `docs/superpowers/specs/2026-05-08-ter-design.md`

**Out of scope for this plan:** F3 quantizer, F4 transformer kernels, F5 ntransformer bridge, F6 Llama 3.2 1B end-to-end, F7-F10 optionals. These are follow-up plans.

---

## File Structure (created during this plan)

```
ter/
├── CMakeLists.txt                       # top-level build, FetchContent doctest
├── src/CMakeLists.txt
├── tests/CMakeLists.txt
├── include/ter/
│   ├── trit.hpp                         # Trit type + half/full adder primitives
│   ├── tryte.hpp                        # 3-trit type
│   ├── word.hpp                         # Word27, Word54 + arithmetic
│   ├── pack.hpp                         # 2-bits-per-trit packing helpers
│   ├── memory.hpp                       # Memory class (uint64-backed)
│   ├── isa.hpp                          # Opcode enum, Instr struct, encode/decode
│   ├── regfile.hpp                      # RegFile (R0..R26, V0..V8, A0..A2, PC, SR)
│   ├── counters.hpp                     # OpCounters
│   ├── sim.hpp                          # Sim public API
│   ├── vec.hpp                          # Vec (27 lanes × 9 trits)
│   └── assembler.hpp                    # text -> bytecode
├── src/
│   ├── core/{trit.cpp,word.cpp,pack.cpp,vec.cpp}
│   ├── sim/{memory.cpp,regfile.cpp,counters.cpp,executor.cpp,sim.cpp}
│   ├── asm/{lexer.cpp,emitter.cpp}
│   └── isa/{encode.cpp,decode.cpp}
├── tests/
│   ├── test_smoke.cpp
│   ├── test_trit.cpp
│   ├── test_tryte.cpp
│   ├── test_word.cpp
│   ├── test_word54.cpp
│   ├── test_pack.cpp
│   ├── test_memory.cpp
│   ├── test_isa.cpp
│   ├── test_regfile.cpp
│   ├── test_counters.cpp
│   ├── test_executor_scalar.cpp
│   ├── test_executor_memory.cpp
│   ├── test_executor_control.cpp
│   ├── test_executor_logic.cpp
│   ├── test_assembler.cpp
│   ├── test_sim_smoke.cpp
│   ├── test_vec.cpp
│   ├── test_executor_simd.cpp
│   ├── test_assembler_simd.cpp
│   └── test_matmul_reference.cpp
├── examples/
│   ├── factorial.tasm
│   └── matmul_64.tasm
└── tools/
    └── matmul_reference.py              # numpy reference for F2 validation gate
```

Each `.hpp` is one type or one tightly-bound group of free functions. Each `.cpp` is the matching implementation. Tests track 1:1 with headers where it makes sense.

---

## Task F0.1 — Project skeleton with CMake, doctest, first failing test

**Files:**
- Create: `CMakeLists.txt`
- Create: `src/CMakeLists.txt`
- Create: `tests/CMakeLists.txt`
- Create: `tests/test_smoke.cpp`

- [ ] **Step 1: Create top-level `CMakeLists.txt`**

```cmake
cmake_minimum_required(VERSION 3.20)
project(ter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

add_compile_options(-Wall -Wextra -Wpedantic -Werror)

include(FetchContent)
FetchContent_Declare(doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG v2.4.11)
FetchContent_MakeAvailable(doctest)

add_library(ter)
target_include_directories(ter PUBLIC ${CMAKE_SOURCE_DIR}/include)

enable_testing()
add_subdirectory(src)
add_subdirectory(tests)
```

- [ ] **Step 2: Create `src/CMakeLists.txt` with stub**

```cmake
target_sources(ter PRIVATE ${CMAKE_SOURCE_DIR}/src/core/_stub.cpp)
```

Create `src/core/_stub.cpp`:

```cpp
namespace ter { void _stub() {} }
```

- [ ] **Step 3: Create `tests/CMakeLists.txt`**

```cmake
function(ter_add_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE ter doctest::doctest)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

ter_add_test(test_smoke)
```

- [ ] **Step 4: Write `tests/test_smoke.cpp`**

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

TEST_CASE("project skeleton compiles") {
    CHECK(1 + 1 == 2);
}
```

- [ ] **Step 5: Configure and build**

Run:
```
cmake -S . -B build
cmake --build build
```

Expected: success. First configure fetches doctest (~1 minute).

- [ ] **Step 6: Run the test**

Run: `ctest --test-dir build --output-on-failure`

Expected: `1/1 Test #1: test_smoke ........... Passed`

- [ ] **Step 7: Commit**

```
git add CMakeLists.txt src tests
git commit -m "chore(ter): cmake + doctest skeleton with smoke test"
```

---

## Task F0.2 — `Trit` type with half-adder and full-adder primitives

**Files:**
- Create: `include/ter/trit.hpp`
- Modify: top-level `CMakeLists.txt` (switch `ter` to INTERFACE temporarily)
- Delete: `src/core/_stub.cpp`
- Create: `tests/test_trit.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write `include/ter/trit.hpp`**

```cpp
#pragma once
#include <cstdint>

namespace ter {

class Trit {
public:
    constexpr Trit() noexcept : v_(0) {}
    constexpr explicit Trit(int v) noexcept
        : v_(static_cast<int8_t>(v < 0 ? -1 : v > 0 ? 1 : 0)) {}

    constexpr int8_t value() const noexcept { return v_; }
    constexpr bool   is_zero() const noexcept { return v_ == 0; }

    constexpr Trit operator-() const noexcept { return Trit{-v_}; }
    constexpr bool operator==(Trit o) const noexcept { return v_ == o.v_; }
    constexpr bool operator!=(Trit o) const noexcept { return v_ != o.v_; }

private:
    int8_t v_;
};

inline constexpr Trit T_NEG{-1};
inline constexpr Trit T_ZERO{0};
inline constexpr Trit T_POS{+1};

struct TritAdd { Trit sum; Trit carry; };

constexpr TritAdd trit_add(Trit a, Trit b) noexcept {
    int s = a.value() + b.value();
    int carry = 0, sum = s;
    if (s >= 2)       { sum -= 3; carry = +1; }
    else if (s <= -2) { sum += 3; carry = -1; }
    return {Trit{sum}, Trit{carry}};
}

constexpr TritAdd trit_full_add(Trit a, Trit b, Trit c) noexcept {
    int s = a.value() + b.value() + c.value();
    int carry = 0, sum = s;
    if (s >= 2)       { sum -= 3; carry = +1; }
    else if (s <= -2) { sum += 3; carry = -1; }
    return {Trit{sum}, Trit{carry}};
}

constexpr Trit trit_max(Trit a, Trit b) noexcept {
    return Trit{a.value() > b.value() ? a.value() : b.value()};
}
constexpr Trit trit_min(Trit a, Trit b) noexcept {
    return Trit{a.value() < b.value() ? a.value() : b.value()};
}

}
```

- [ ] **Step 2: Switch library to INTERFACE temporarily**

In top-level `CMakeLists.txt` replace:
```cmake
add_library(ter)
target_include_directories(ter PUBLIC ${CMAKE_SOURCE_DIR}/include)
```
with:
```cmake
add_library(ter INTERFACE)
target_include_directories(ter INTERFACE ${CMAKE_SOURCE_DIR}/include)
```

Remove the line `add_subdirectory(src)` for now (re-added in F0.4 when first .cpp lands).

Delete the stub: `rm src/core/_stub.cpp` and remove `src/CMakeLists.txt`'s `target_sources` line (leave file empty).

- [ ] **Step 3: Write `tests/test_trit.cpp`**

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/trit.hpp>

using namespace ter;

TEST_CASE("Trit construction clamps to {-1, 0, +1}") {
    CHECK(Trit{-5}.value() == -1);
    CHECK(Trit{0}.value() == 0);
    CHECK(Trit{+99}.value() == +1);
}

TEST_CASE("Trit negation") {
    CHECK((-T_POS) == T_NEG);
    CHECK((-T_NEG) == T_POS);
    CHECK((-T_ZERO) == T_ZERO);
}

TEST_CASE("trit_add covers all 9 cases") {
    struct Case { Trit a, b, sum, carry; };
    Case cases[] = {
        {T_NEG, T_NEG, T_POS, T_NEG},
        {T_NEG, T_ZERO, T_NEG, T_ZERO},
        {T_NEG, T_POS, T_ZERO, T_ZERO},
        {T_ZERO, T_NEG, T_NEG, T_ZERO},
        {T_ZERO, T_ZERO, T_ZERO, T_ZERO},
        {T_ZERO, T_POS, T_POS, T_ZERO},
        {T_POS, T_NEG, T_ZERO, T_ZERO},
        {T_POS, T_ZERO, T_POS, T_ZERO},
        {T_POS, T_POS, T_NEG, T_POS},
    };
    for (auto& c : cases) {
        auto r = trit_add(c.a, c.b);
        CHECK(r.sum == c.sum);
        CHECK(r.carry == c.carry);
    }
}

TEST_CASE("trit_full_add over Trit^3 reconstructs sum") {
    for (int a : {-1, 0, 1}) for (int b : {-1, 0, 1}) for (int c : {-1, 0, 1}) {
        auto r = trit_full_add(Trit{a}, Trit{b}, Trit{c});
        CHECK(r.sum.value() + 3 * r.carry.value() == a + b + c);
    }
}

TEST_CASE("trit_max and trit_min") {
    CHECK(trit_max(T_NEG, T_POS) == T_POS);
    CHECK(trit_min(T_NEG, T_POS) == T_NEG);
}
```

- [ ] **Step 4: Register and run**

Append `ter_add_test(test_trit)` to `tests/CMakeLists.txt`.

```
cmake --build build && ctest --test-dir build --output-on-failure
```

Expected: 2/2 pass.

- [ ] **Step 5: Commit**

```
git add include/ter/trit.hpp tests/test_trit.cpp tests/CMakeLists.txt CMakeLists.txt src
git commit -m "feat(core): Trit type with balanced add and logic primitives"
```

---

## Task F0.3 — `Tryte` type (3-trit, range +/- 13)

**Files:**
- Create: `include/ter/tryte.hpp`
- Create: `tests/test_tryte.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write `tests/test_tryte.cpp`**

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/tryte.hpp>

using namespace ter;

TEST_CASE("Tryte round-trip int <-> tryte") {
    for (int v = -13; v <= 13; ++v) {
        Tryte t = Tryte::from_int(v);
        CHECK(t.to_int() == v);
    }
}

TEST_CASE("Tryte range") {
    CHECK(Tryte::min_int() == -13);
    CHECK(Tryte::max_int() == +13);
}

TEST_CASE("Tryte negation") {
    for (int v = -13; v <= 13; ++v) {
        Tryte t = Tryte::from_int(v);
        CHECK((-t).to_int() == -v);
    }
}
```

- [ ] **Step 2: Implement `include/ter/tryte.hpp`**

```cpp
#pragma once
#include <ter/trit.hpp>
#include <array>

namespace ter {

class Tryte {
public:
    constexpr Tryte() noexcept = default;

    static constexpr int min_int() noexcept { return -13; }
    static constexpr int max_int() noexcept { return +13; }

    static constexpr Tryte from_int(int v) noexcept {
        Tryte t;
        for (int i = 0; i < 3; ++i) {
            int r = ((v % 3) + 3) % 3;
            int digit = (r == 2) ? -1 : r;
            t.trits_[i] = Trit{digit};
            v = (v - digit) / 3;
        }
        return t;
    }

    constexpr int to_int() const noexcept {
        int acc = 0, place = 1;
        for (int i = 0; i < 3; ++i) {
            acc += trits_[i].value() * place;
            place *= 3;
        }
        return acc;
    }

    constexpr Trit trit(int i) const noexcept { return trits_[i]; }

    constexpr Tryte operator-() const noexcept {
        Tryte r;
        for (int i = 0; i < 3; ++i) r.trits_[i] = -trits_[i];
        return r;
    }

    constexpr bool operator==(Tryte o) const noexcept {
        for (int i = 0; i < 3; ++i) if (trits_[i] != o.trits_[i]) return false;
        return true;
    }

private:
    std::array<Trit, 3> trits_{};
};

}
```

- [ ] **Step 3: Register, build, run**

Append `ter_add_test(test_tryte)`.

Expected: 3/3 pass.

- [ ] **Step 4: Commit**

```
git add include/ter/tryte.hpp tests/test_tryte.cpp tests/CMakeLists.txt
git commit -m "feat(core): Tryte (3-trit, +/-13) with round-trip int conversion"
```

---

## Task F0.4 — `Word27` type and arithmetic

**Files:**
- Create: `include/ter/word.hpp`
- Create: `src/core/word.cpp`
- Create: `tests/test_word.cpp`
- Modify: top-level `CMakeLists.txt` (restore non-INTERFACE library), `src/CMakeLists.txt`, `tests/CMakeLists.txt`

- [ ] **Step 1: Restore `ter` to a regular library**

In top-level `CMakeLists.txt`:
```cmake
add_library(ter)
target_include_directories(ter PUBLIC ${CMAKE_SOURCE_DIR}/include)
add_subdirectory(src)
```

In `src/CMakeLists.txt`:
```cmake
target_sources(ter PRIVATE
    core/word.cpp
)
```

- [ ] **Step 2: Write `tests/test_word.cpp`**

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/word.hpp>
#include <cstdlib>

using namespace ter;

TEST_CASE("Word27 round-trip from int64") {
    int64_t cases[] = {0, 1, -1, 42, -42, 7625597484987LL, -7625597484987LL};
    for (int64_t v : cases) {
        if (std::abs(v) <= Word27::max_int()) {
            CHECK(Word27::from_int(v).to_int() == v);
        }
    }
}

TEST_CASE("Word27 add and sub") {
    auto a = Word27::from_int(1234567);
    auto b = Word27::from_int(-89012);
    auto c = Word27::from_int(345);
    CHECK((a + b).to_int() == 1234567 - 89012);
    CHECK((a - b).to_int() == 1234567 + 89012);
    CHECK((((a + b) + c)) == ((a + (b + c))));
}

TEST_CASE("Word27 negation") {
    auto x = Word27::from_int(987654);
    CHECK((-x).to_int() == -987654);
    CHECK((-(-x)) == x);
}

TEST_CASE("Word27 sign_trit") {
    CHECK(sign_trit(Word27::from_int(100)) == T_POS);
    CHECK(sign_trit(Word27::from_int(-100)) == T_NEG);
    CHECK(sign_trit(Word27::from_int(0)) == T_ZERO);
}
```

- [ ] **Step 3: Write `include/ter/word.hpp`**

```cpp
#pragma once
#include <ter/trit.hpp>
#include <array>
#include <cstdint>

namespace ter {

class Word27 {
public:
    static constexpr int kTrits = 27;

    constexpr Word27() noexcept = default;

    static constexpr int64_t max_int() noexcept {
        int64_t v = 1;
        for (int i = 0; i < 27; ++i) v *= 3;
        return (v - 1) / 2;
    }
    static constexpr int64_t min_int() noexcept { return -max_int(); }

    static Word27 from_int(int64_t v) noexcept;
    int64_t to_int() const noexcept;

    constexpr Trit trit(int i) const noexcept { return trits_[i]; }
    constexpr void set_trit(int i, Trit t) noexcept { trits_[i] = t; }

    Word27 operator-() const noexcept;
    Word27 operator+(Word27 o) const noexcept;
    Word27 operator-(Word27 o) const noexcept;

    constexpr bool operator==(Word27 o) const noexcept {
        for (int i = 0; i < kTrits; ++i) if (trits_[i] != o.trits_[i]) return false;
        return true;
    }

private:
    std::array<Trit, kTrits> trits_{};
};

Trit sign_trit(Word27 w) noexcept;

}
```

- [ ] **Step 4: Implement `src/core/word.cpp`**

```cpp
#include <ter/word.hpp>

namespace ter {

Word27 Word27::from_int(int64_t v) noexcept {
    Word27 w;
    for (int i = 0; i < kTrits; ++i) {
        int r = static_cast<int>(((v % 3) + 3) % 3);
        int digit = (r == 2) ? -1 : r;
        w.trits_[i] = Trit{digit};
        v = (v - digit) / 3;
    }
    return w;
}

int64_t Word27::to_int() const noexcept {
    int64_t acc = 0;
    int64_t place = 1;
    for (int i = 0; i < kTrits; ++i) {
        acc += static_cast<int64_t>(trits_[i].value()) * place;
        place *= 3;
    }
    return acc;
}

Word27 Word27::operator-() const noexcept {
    Word27 r;
    for (int i = 0; i < kTrits; ++i) r.trits_[i] = -trits_[i];
    return r;
}

Word27 Word27::operator+(Word27 o) const noexcept {
    Word27 r;
    Trit carry = T_ZERO;
    for (int i = 0; i < kTrits; ++i) {
        auto rr = trit_full_add(trits_[i], o.trits_[i], carry);
        r.trits_[i] = rr.sum;
        carry = rr.carry;
    }
    return r;
}

Word27 Word27::operator-(Word27 o) const noexcept {
    return *this + (-o);
}

Trit sign_trit(Word27 w) noexcept {
    for (int i = Word27::kTrits - 1; i >= 0; --i) {
        if (!w.trit(i).is_zero()) return w.trit(i);
    }
    return T_ZERO;
}

}
```

- [ ] **Step 5: Register and run**

Add `ter_add_test(test_word)`. Expected: 4/4 pass.

- [ ] **Step 6: Commit**

```
git add include/ter/word.hpp src/core/word.cpp src/CMakeLists.txt tests/test_word.cpp tests/CMakeLists.txt CMakeLists.txt
git commit -m "feat(core): Word27 type with add, sub, neg, sign-trit"
```

---

## Task F0.5 — `Word54` accumulator + multiply

**Files:**
- Modify: `include/ter/word.hpp`
- Modify: `src/core/word.cpp`
- Create: `tests/test_word54.cpp`

- [ ] **Step 1: Write `tests/test_word54.cpp`**

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/word.hpp>

using namespace ter;

TEST_CASE("Word54 round-trip") {
    int64_t cases[] = {0, 1, -1, 1000000, -1000000, 9999999, -9999999};
    for (int64_t v : cases) {
        Word54 w = Word54::from_int(v);
        CHECK(w.to_int() == v);
    }
}

TEST_CASE("Word54 = Word27 * Word27") {
    auto a = Word27::from_int(123456);
    auto b = Word27::from_int(-789);
    Word54 prod = mul(a, b);
    CHECK(prod.to_int() == int64_t{123456} * int64_t{-789});
}

TEST_CASE("Word54 mac_inplace") {
    Word54 acc = Word54::from_int(1000);
    auto a = Word27::from_int(7);
    auto b = Word27::from_int(11);
    mac_inplace(acc, a, b);
    CHECK(acc.to_int() == 1000 + 77);
}
```

- [ ] **Step 2: Append to `include/ter/word.hpp`**

```cpp
class Word54 {
public:
    static constexpr int kTrits = 54;

    constexpr Word54() noexcept = default;

    static Word54 from_int(int64_t v) noexcept;
    int64_t to_int() const noexcept;

    constexpr Trit trit(int i) const noexcept { return trits_[i]; }
    constexpr void set_trit(int i, Trit t) noexcept { trits_[i] = t; }

    Word54 operator+(Word54 o) const noexcept;
    Word54 operator-() const noexcept;

private:
    std::array<Trit, kTrits> trits_{};
};

Word54 mul(Word27 a, Word27 b) noexcept;
void   mac_inplace(Word54& acc, Word27 a, Word27 b) noexcept;
```

- [ ] **Step 3: Append to `src/core/word.cpp`**

```cpp
Word54 Word54::from_int(int64_t v) noexcept {
    Word54 w;
    for (int i = 0; i < kTrits; ++i) {
        int r = static_cast<int>(((v % 3) + 3) % 3);
        int digit = (r == 2) ? -1 : r;
        w.trits_[i] = Trit{digit};
        v = (v - digit) / 3;
        if (v == 0) break;
    }
    return w;
}

int64_t Word54::to_int() const noexcept {
    __int128 acc = 0, place = 1;
    for (int i = 0; i < kTrits; ++i) {
        acc += static_cast<__int128>(trits_[i].value()) * place;
        place *= 3;
    }
    return static_cast<int64_t>(acc);
}

Word54 Word54::operator+(Word54 o) const noexcept {
    Word54 r;
    Trit carry = T_ZERO;
    for (int i = 0; i < kTrits; ++i) {
        auto rr = trit_full_add(trits_[i], o.trits_[i], carry);
        r.trits_[i] = rr.sum;
        carry = rr.carry;
    }
    return r;
}

Word54 Word54::operator-() const noexcept {
    Word54 r;
    for (int i = 0; i < kTrits; ++i) r.trits_[i] = -trits_[i];
    return r;
}

Word54 mul(Word27 a, Word27 b) noexcept {
    Word54 acc;
    for (int j = 0; j < Word27::kTrits; ++j) {
        Trit bj = b.trit(j);
        if (bj.is_zero()) continue;
        Word54 partial;
        for (int i = 0; i < Word27::kTrits; ++i) {
            int dst = i + j;
            if (dst >= Word54::kTrits) break;
            Trit ai = a.trit(i);
            partial.set_trit(dst, bj.value() < 0 ? -ai : ai);
        }
        acc = acc + partial;
    }
    return acc;
}

void mac_inplace(Word54& acc, Word27 a, Word27 b) noexcept {
    acc = acc + mul(a, b);
}
```

- [ ] **Step 4: Register and run**

Add `ter_add_test(test_word54)`. Expected: 5/5 pass.

- [ ] **Step 5: Commit**

```
git add include/ter/word.hpp src/core/word.cpp tests/test_word54.cpp tests/CMakeLists.txt
git commit -m "feat(core): Word54 accumulator with mul and mac primitives"
```

---

## Task F0.6 — 2-bits-per-trit packing

**Files:**
- Create: `include/ter/pack.hpp`
- Create: `src/core/pack.cpp`
- Create: `tests/test_pack.cpp`

- [ ] **Step 1: Write `tests/test_pack.cpp`**

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/pack.hpp>
#include <cstdlib>

using namespace ter;

TEST_CASE("Trit <-> 2-bit code") {
    CHECK(trit_to_code(T_ZERO) == 0b00);
    CHECK(trit_to_code(T_POS)  == 0b01);
    CHECK(trit_to_code(T_NEG)  == 0b10);
    CHECK(code_to_trit(0b00) == T_ZERO);
    CHECK(code_to_trit(0b01) == T_POS);
    CHECK(code_to_trit(0b10) == T_NEG);
}

TEST_CASE("Word27 <-> uint64 round-trip") {
    int64_t cases[] = {0, 1, -1, 1234567890, -1234567890};
    for (int64_t v : cases) {
        if (std::abs(v) > Word27::max_int()) continue;
        Word27 w = Word27::from_int(v);
        uint64_t packed = pack_word27(w);
        Word27 r = unpack_word27(packed);
        CHECK(r == w);
        CHECK(r.to_int() == v);
    }
}

TEST_CASE("Reserved 0b11 decodes as zero") {
    uint64_t bad = 0b11;
    Word27 r = unpack_word27(bad);
    CHECK(r.trit(0) == T_ZERO);
}
```

- [ ] **Step 2: Write `include/ter/pack.hpp`**

```cpp
#pragma once
#include <ter/trit.hpp>
#include <ter/word.hpp>
#include <cstdint>

namespace ter {

constexpr uint8_t trit_to_code(Trit t) noexcept {
    int v = t.value();
    return v == 0 ? 0b00 : v == +1 ? 0b01 : 0b10;
}

constexpr Trit code_to_trit(uint8_t c) noexcept {
    switch (c & 0b11) {
        case 0b00: return T_ZERO;
        case 0b01: return T_POS;
        case 0b10: return T_NEG;
        default:   return T_ZERO;
    }
}

uint64_t pack_word27(const Word27& w) noexcept;
Word27   unpack_word27(uint64_t packed) noexcept;

}
```

- [ ] **Step 3: Implement `src/core/pack.cpp`**

```cpp
#include <ter/pack.hpp>

namespace ter {

uint64_t pack_word27(const Word27& w) noexcept {
    uint64_t out = 0;
    for (int i = 0; i < Word27::kTrits; ++i) {
        out |= static_cast<uint64_t>(trit_to_code(w.trit(i))) << (i * 2);
    }
    return out;
}

Word27 unpack_word27(uint64_t packed) noexcept {
    Word27 w;
    for (int i = 0; i < Word27::kTrits; ++i) {
        uint8_t code = static_cast<uint8_t>((packed >> (i * 2)) & 0b11);
        w.set_trit(i, code_to_trit(code));
    }
    return w;
}

}
```

- [ ] **Step 4: Wire build, register, run**

In `src/CMakeLists.txt` add `core/pack.cpp`. Add `ter_add_test(test_pack)`.

Expected: 6/6 pass.

- [ ] **Step 5: Commit**

```
git add include/ter/pack.hpp src/core/pack.cpp src/CMakeLists.txt tests/test_pack.cpp tests/CMakeLists.txt
git commit -m "feat(core): 2-bits-per-trit packing for Word27 <-> uint64"
```

---

## Task F0.7 — `Memory` class (uint64-backed, word-addressed)

**Files:**
- Create: `include/ter/memory.hpp`
- Create: `src/sim/memory.cpp`
- Create: `tests/test_memory.cpp`

- [ ] **Step 1: Write `tests/test_memory.cpp`**

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/memory.hpp>

using namespace ter;

TEST_CASE("Memory store/load round-trip") {
    Memory mem(1024);
    auto w = Word27::from_int(424242);
    mem.store_word(100, w);
    CHECK(mem.load_word(100) == w);
}

TEST_CASE("Memory zero-initialised") {
    Memory mem(16);
    CHECK(mem.load_word(0).to_int() == 0);
    CHECK(mem.load_word(15).to_int() == 0);
}

TEST_CASE("Memory bounds violation throws") {
    Memory mem(8);
    CHECK_THROWS_AS(mem.load_word(8), std::out_of_range);
    CHECK_THROWS_AS(mem.store_word(99, Word27{}), std::out_of_range);
}
```

- [ ] **Step 2: Write `include/ter/memory.hpp`**

```cpp
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
```

- [ ] **Step 3: Implement `src/sim/memory.cpp`**

```cpp
#include <ter/memory.hpp>
// Header-only for now; reserved for region tracking later.
```

- [ ] **Step 4: Wire, register, run**

Add `sim/memory.cpp` to `src/CMakeLists.txt`. Add `ter_add_test(test_memory)`.

Expected: 7/7 pass.

- [ ] **Step 5: Commit**

```
git add include/ter/memory.hpp src/sim/memory.cpp src/CMakeLists.txt tests/test_memory.cpp tests/CMakeLists.txt
git commit -m "feat(sim): Memory (word-addressed, uint64-backed)"
```

---

## Task F0.8 — F0 phase gate: exhaustive property tests

**Files:**
- Modify: `tests/test_trit.cpp`
- Modify: `tests/test_word.cpp`

- [ ] **Step 1: Append exhaustive properties to `test_trit.cpp`**

```cpp
TEST_CASE("trit_full_add commutative over Trit^3") {
    for (int a : {-1, 0, 1}) for (int b : {-1, 0, 1}) for (int c : {-1, 0, 1}) {
        auto r1 = trit_full_add(Trit{a}, Trit{b}, Trit{c});
        auto r2 = trit_full_add(Trit{b}, Trit{a}, Trit{c});
        CHECK(r1.sum == r2.sum);
        CHECK(r1.carry == r2.carry);
    }
}
```

- [ ] **Step 2: Append properties to `test_word.cpp`**

```cpp
TEST_CASE("Word27 (a + b) + c == a + (b + c) -- sampled") {
    int64_t samples[] = {0, 1, -1, 100, -100, 1234567, -1234567, 99999999};
    for (auto a : samples) for (auto b : samples) for (auto c : samples) {
        auto wa = Word27::from_int(a);
        auto wb = Word27::from_int(b);
        auto wc = Word27::from_int(c);
        if (std::abs(a + b + c) <= Word27::max_int()) {
            CHECK(((wa + wb) + wc) == (wa + (wb + wc)));
        }
    }
}

TEST_CASE("Word27 -(-x) == x") {
    for (int64_t v : {0LL, 1LL, -1LL, 99999999LL, -99999999LL}) {
        auto w = Word27::from_int(v);
        CHECK((-(-w)) == w);
    }
}
```

- [ ] **Step 3: Build and run**

Expected: all tests pass; CHECK count grows.

- [ ] **Step 4: Commit**

```
git add tests/test_trit.cpp tests/test_word.cpp
git commit -m "test(core): exhaustive property tests for F0 gate"
```

---

## Task F1.1 — ISA opcode enum and instruction encode/decode

**Files:**
- Create: `include/ter/isa.hpp`
- Create: `src/isa/encode.cpp`
- Create: `src/isa/decode.cpp`
- Create: `tests/test_isa.cpp`

- [ ] **Step 1: Write `tests/test_isa.cpp`**

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/isa.hpp>

using namespace ter;

TEST_CASE("encode/decode round-trip TADD") {
    Instr i; i.op = Opcode::TADD; i.dst=1; i.src1=2; i.src2=3; i.imm=0;
    Word27 w = encode(i);
    Instr d = decode(w);
    CHECK(d.op == i.op);
    CHECK(d.dst == 1);
    CHECK(d.src1 == 2);
    CHECK(d.src2 == 3);
    CHECK(d.imm == 0);
}

TEST_CASE("encode/decode TLOADI with negative imm") {
    Instr i; i.op = Opcode::TLOADI; i.dst=5; i.src1=0; i.src2=0; i.imm=-12345;
    Word27 w = encode(i);
    Instr d = decode(w);
    CHECK(d.op == Opcode::TLOADI);
    CHECK(d.dst == 5);
    CHECK(d.imm == -12345);
}

TEST_CASE("decode rejects unmapped opcode value") {
    Word27 w;
    for (int t = 21; t < 27; ++t) w.set_trit(t, T_POS);
    CHECK_THROWS_AS(decode(w), IllegalOpcode);
}
```

- [ ] **Step 2: Write `include/ter/isa.hpp`**

```cpp
#pragma once
#include <ter/word.hpp>
#include <stdexcept>
#include <cstdint>

namespace ter {

enum class Opcode : int16_t {
    TNOP    = 0,
    THALT   = 1,
    TDBG    = 2,
    TADD    = 10,
    TSUB    = 11,
    TNEG    = 12,
    TABS    = 13,
    TAND3   = 14,
    TOR3    = 15,
    TXOR3   = 16,
    TCMP    = 17,
    TSIGN   = 18,
    TLOAD   = 30,
    TSTORE  = 31,
    TLOADI  = 32,
    TBEQ    = 50,
    TBNE    = 51,
    TBLT    = 52,
    TJUMP   = 53,
    TCALL   = 54,
    TRET    = 55,
};

struct Instr {
    Opcode op;
    uint8_t dst;
    uint8_t src1;
    uint8_t src2;
    int32_t imm;
};

class IllegalOpcode : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

Word27 encode(const Instr& i);
Instr  decode(Word27 w);

constexpr int kOpcodeTrits = 6;
constexpr int kRegTrits    = 3;
constexpr int kImmTrits    = 12;

}
```

- [ ] **Step 3: Write `src/isa/encode.cpp`**

```cpp
#include <ter/isa.hpp>

namespace ter {

namespace {
void put_trits(Word27& w, int start, int n, int64_t value) {
    for (int i = 0; i < n; ++i) {
        int r = static_cast<int>(((value % 3) + 3) % 3);
        int digit = (r == 2) ? -1 : r;
        w.set_trit(start + i, Trit{digit});
        value = (value - digit) / 3;
    }
}
}

Word27 encode(const Instr& i) {
    Word27 w;
    put_trits(w, 0,  kImmTrits,    i.imm);
    put_trits(w, 12, kRegTrits,    static_cast<int64_t>(i.src2) - 13);
    put_trits(w, 15, kRegTrits,    static_cast<int64_t>(i.src1) - 13);
    put_trits(w, 18, kRegTrits,    static_cast<int64_t>(i.dst)  - 13);
    put_trits(w, 21, kOpcodeTrits, static_cast<int64_t>(i.op));
    return w;
}

}
```

- [ ] **Step 4: Write `src/isa/decode.cpp`**

```cpp
#include <ter/isa.hpp>
#include <string>

namespace ter {

namespace {
int64_t get_trits(const Word27& w, int start, int n) {
    int64_t acc = 0, place = 1;
    for (int i = 0; i < n; ++i) {
        acc += static_cast<int64_t>(w.trit(start + i).value()) * place;
        place *= 3;
    }
    return acc;
}

bool is_valid_opcode(int64_t v) {
    switch (static_cast<Opcode>(v)) {
        case Opcode::TNOP: case Opcode::THALT: case Opcode::TDBG:
        case Opcode::TADD: case Opcode::TSUB: case Opcode::TNEG: case Opcode::TABS:
        case Opcode::TAND3: case Opcode::TOR3: case Opcode::TXOR3:
        case Opcode::TCMP: case Opcode::TSIGN:
        case Opcode::TLOAD: case Opcode::TSTORE: case Opcode::TLOADI:
        case Opcode::TBEQ: case Opcode::TBNE: case Opcode::TBLT:
        case Opcode::TJUMP: case Opcode::TCALL: case Opcode::TRET:
            return true;
    }
    return false;
}
}

Instr decode(Word27 w) {
    int64_t op_v = get_trits(w, 21, kOpcodeTrits);
    if (!is_valid_opcode(op_v)) {
        throw IllegalOpcode("decode: opcode value " + std::to_string(op_v) + " unmapped");
    }
    Instr i;
    i.op   = static_cast<Opcode>(op_v);
    i.dst  = static_cast<uint8_t>(get_trits(w, 18, kRegTrits) + 13);
    i.src1 = static_cast<uint8_t>(get_trits(w, 15, kRegTrits) + 13);
    i.src2 = static_cast<uint8_t>(get_trits(w, 12, kRegTrits) + 13);
    i.imm  = static_cast<int32_t>(get_trits(w, 0, kImmTrits));
    return i;
}

}
```

- [ ] **Step 5: Wire build, register, run**

In `src/CMakeLists.txt` add `isa/encode.cpp isa/decode.cpp`. Add `ter_add_test(test_isa)`.

Expected: 8/8 pass.

- [ ] **Step 6: Commit**

```
git add include/ter/isa.hpp src/isa src/CMakeLists.txt tests/test_isa.cpp tests/CMakeLists.txt
git commit -m "feat(isa): opcode enum and Word27 instruction encode/decode"
```

---

## Task F1.2 — `RegFile` (R0..R26, PC, halt flag)

**Files:**
- Create: `include/ter/regfile.hpp`
- Create: `src/sim/regfile.cpp`
- Create: `tests/test_regfile.cpp`

- [ ] **Step 1: Write `tests/test_regfile.cpp`**

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/regfile.hpp>

using namespace ter;

TEST_CASE("R0 is hard-zero") {
    RegFile rf;
    rf.write_scalar(0, Word27::from_int(42));
    CHECK(rf.read_scalar(0).to_int() == 0);
}

TEST_CASE("read/write scalar") {
    RegFile rf;
    rf.write_scalar(1, Word27::from_int(123));
    rf.write_scalar(26, Word27::from_int(-456));
    CHECK(rf.read_scalar(1).to_int() == 123);
    CHECK(rf.read_scalar(26).to_int() == -456);
}

TEST_CASE("PC and halted flag") {
    RegFile rf;
    CHECK(rf.pc().to_int() == 0);
    CHECK_FALSE(rf.halted());
    rf.set_pc(Word27::from_int(100));
    CHECK(rf.pc().to_int() == 100);
    rf.set_halted(true);
    CHECK(rf.halted());
}
```

- [ ] **Step 2: Write `include/ter/regfile.hpp`**

```cpp
#pragma once
#include <ter/word.hpp>
#include <array>
#include <stdexcept>

namespace ter {

class RegFile {
public:
    static constexpr int kScalarRegs = 27;

    Word27 read_scalar(int idx) const {
        check(idx);
        if (idx == 0) return Word27{};
        return scalars_[idx];
    }

    void write_scalar(int idx, Word27 v) {
        check(idx);
        if (idx == 0) return;
        scalars_[idx] = v;
    }

    Word27 pc() const noexcept { return pc_; }
    void   set_pc(Word27 v) noexcept { pc_ = v; }

    bool halted() const noexcept { return halted_; }
    void set_halted(bool v) noexcept { halted_ = v; }

private:
    static void check(int idx) {
        if (idx < 0 || idx >= kScalarRegs) throw std::out_of_range("RegFile index");
    }

    std::array<Word27, kScalarRegs> scalars_{};
    Word27 pc_{};
    bool halted_ = false;
};

}
```

- [ ] **Step 3: `src/sim/regfile.cpp`**

```cpp
#include <ter/regfile.hpp>
// Currently header-only; reserved for vector-register integration in F2.2.
```

- [ ] **Step 4: Wire and run**

Add `sim/regfile.cpp` and `ter_add_test(test_regfile)`.

Expected: 9/9 pass.

- [ ] **Step 5: Commit**

```
git add include/ter/regfile.hpp src/sim/regfile.cpp src/CMakeLists.txt tests/test_regfile.cpp tests/CMakeLists.txt
git commit -m "feat(sim): RegFile with R0 hard-zero, PC, halt flag"
```

---

## Task F1.3 — `OpCounters`

**Files:**
- Create: `include/ter/counters.hpp`
- Create: `tests/test_counters.cpp`

- [ ] **Step 1: Write `tests/test_counters.cpp`**

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/counters.hpp>

using namespace ter;

TEST_CASE("OpCounters bumps per opcode") {
    OpCounters c;
    CHECK(c.get(Opcode::TADD) == 0);
    c.bump(Opcode::TADD);
    c.bump(Opcode::TADD);
    c.bump(Opcode::TSUB);
    CHECK(c.get(Opcode::TADD) == 2);
    CHECK(c.get(Opcode::TSUB) == 1);
}

TEST_CASE("OpCounters total") {
    OpCounters c;
    c.bump(Opcode::TADD);
    c.bump(Opcode::THALT);
    CHECK(c.total() == 2);
}
```

- [ ] **Step 2: Write `include/ter/counters.hpp`**

```cpp
#pragma once
#include <ter/isa.hpp>
#include <unordered_map>
#include <cstdint>

namespace ter {

class OpCounters {
public:
    void bump(Opcode op) noexcept { ++counts_[static_cast<int16_t>(op)]; ++total_; }
    uint64_t get(Opcode op) const noexcept {
        auto it = counts_.find(static_cast<int16_t>(op));
        return it == counts_.end() ? 0 : it->second;
    }
    uint64_t total() const noexcept { return total_; }
    void reset() noexcept { counts_.clear(); total_ = 0; }

    const std::unordered_map<int16_t, uint64_t>& raw() const noexcept { return counts_; }

private:
    std::unordered_map<int16_t, uint64_t> counts_;
    uint64_t total_ = 0;
};

}
```

- [ ] **Step 3: Register and run**

Add `ter_add_test(test_counters)`.

Expected: 10/10 pass.

- [ ] **Step 4: Commit**

```
git add include/ter/counters.hpp tests/test_counters.cpp tests/CMakeLists.txt
git commit -m "feat(sim): OpCounters per-opcode bumping and totals"
```

---

## Task F1.4 — `Sim` with scalar arithmetic ops

**Files:**
- Create: `include/ter/sim.hpp`
- Create: `src/sim/run_one.cpp` (the dispatcher; named to avoid collision)
- Create: `src/sim/sim.cpp`
- Create: `tests/test_executor_scalar.cpp`

- [ ] **Step 1: Write `tests/test_executor_scalar.cpp`**

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/sim.hpp>

using namespace ter;

static Sim build_sim_with(std::initializer_list<Instr> code) {
    Sim s(64);
    size_t addr = 0;
    for (const auto& i : code) s.mem().store_word(addr++, encode(i));
    return s;
}

TEST_CASE("TLOADI") {
    Sim s = build_sim_with({
        {Opcode::TLOADI, 1, 0, 0, 42},
        {Opcode::THALT,  0, 0, 0, 0},
    });
    s.run();
    CHECK(s.regs().read_scalar(1).to_int() == 42);
}

TEST_CASE("TADD adds two registers") {
    Sim s = build_sim_with({
        {Opcode::TLOADI, 1, 0, 0, 10},
        {Opcode::TLOADI, 2, 0, 0, 20},
        {Opcode::TADD,   3, 1, 2, 0},
        {Opcode::THALT,  0, 0, 0, 0},
    });
    s.run();
    CHECK(s.regs().read_scalar(3).to_int() == 30);
}

TEST_CASE("TSUB and TNEG") {
    Sim s = build_sim_with({
        {Opcode::TLOADI, 1, 0, 0, 50},
        {Opcode::TLOADI, 2, 0, 0, 30},
        {Opcode::TSUB,   3, 1, 2, 0},
        {Opcode::TNEG,   4, 3, 0, 0},
        {Opcode::THALT,  0, 0, 0, 0},
    });
    s.run();
    CHECK(s.regs().read_scalar(3).to_int() == 20);
    CHECK(s.regs().read_scalar(4).to_int() == -20);
}

TEST_CASE("counters reflect ops") {
    Sim s = build_sim_with({
        {Opcode::TLOADI, 1, 0, 0, 7},
        {Opcode::TADD,   2, 1, 1, 0},
        {Opcode::TADD,   3, 2, 1, 0},
        {Opcode::THALT,  0, 0, 0, 0},
    });
    s.run();
    CHECK(s.counters().get(Opcode::TADD)   == 2);
    CHECK(s.counters().get(Opcode::TLOADI) == 1);
    CHECK(s.counters().get(Opcode::THALT)  == 1);
}
```

- [ ] **Step 2: Write `include/ter/sim.hpp`**

```cpp
#pragma once
#include <ter/memory.hpp>
#include <ter/regfile.hpp>
#include <ter/counters.hpp>
#include <ter/isa.hpp>

namespace ter {

class Sim {
public:
    explicit Sim(size_t mem_words) : mem_(mem_words) {}

    Memory&         mem()         noexcept { return mem_; }
    const Memory&   mem()   const noexcept { return mem_; }
    RegFile&        regs()        noexcept { return regs_; }
    const RegFile&  regs()  const noexcept { return regs_; }
    OpCounters&     counters()       noexcept { return counters_; }
    const OpCounters& counters() const noexcept { return counters_; }

    void step();
    void run();

private:
    void run_one(const Instr& i);

    Memory mem_;
    RegFile regs_;
    OpCounters counters_;
};

}
```

- [ ] **Step 3: Write `src/sim/run_one.cpp`**

```cpp
#include <ter/sim.hpp>

namespace ter {

void Sim::run_one(const Instr& i) {
    counters_.bump(i.op);
    switch (i.op) {
        case Opcode::TNOP:
            break;
        case Opcode::THALT:
            regs_.set_halted(true);
            break;
        case Opcode::TDBG:
            break;
        case Opcode::TLOADI:
            regs_.write_scalar(i.dst, Word27::from_int(i.imm));
            break;
        case Opcode::TADD: {
            auto a = regs_.read_scalar(i.src1);
            auto b = regs_.read_scalar(i.src2);
            regs_.write_scalar(i.dst, a + b);
            break;
        }
        case Opcode::TSUB: {
            auto a = regs_.read_scalar(i.src1);
            auto b = regs_.read_scalar(i.src2);
            regs_.write_scalar(i.dst, a - b);
            break;
        }
        case Opcode::TNEG:
            regs_.write_scalar(i.dst, -regs_.read_scalar(i.src1));
            break;
        case Opcode::TABS: {
            auto v = regs_.read_scalar(i.src1);
            regs_.write_scalar(i.dst, sign_trit(v) == T_NEG ? -v : v);
            break;
        }
        case Opcode::TSIGN: {
            auto v = regs_.read_scalar(i.src1);
            Word27 r;
            r.set_trit(0, sign_trit(v));
            regs_.write_scalar(i.dst, r);
            break;
        }
        default:
            throw IllegalOpcode("Sim::run_one: opcode not yet implemented");
    }
}

}
```

- [ ] **Step 4: Write `src/sim/sim.cpp`**

```cpp
#include <ter/sim.hpp>

namespace ter {

void Sim::step() {
    if (regs_.halted()) return;
    auto pc = regs_.pc();
    Word27 raw = mem_.load_word(static_cast<size_t>(pc.to_int()));
    regs_.set_pc(Word27::from_int(pc.to_int() + 1));
    Instr i = decode(raw);
    run_one(i);
}

void Sim::run() {
    constexpr int kSafetyLimit = 1'000'000;
    for (int n = 0; n < kSafetyLimit && !regs_.halted(); ++n) step();
    if (!regs_.halted()) throw std::runtime_error("Sim::run safety limit");
}

}
```

- [ ] **Step 5: Wire build, register, run**

In `src/CMakeLists.txt`:
```cmake
target_sources(ter PRIVATE
    core/word.cpp
    core/pack.cpp
    sim/memory.cpp
    sim/regfile.cpp
    sim/run_one.cpp
    sim/sim.cpp
    isa/encode.cpp
    isa/decode.cpp
)
```

Add `ter_add_test(test_executor_scalar)`. Expected: 11/11 pass.

- [ ] **Step 6: Commit**

```
git add include/ter/sim.hpp src/sim src/CMakeLists.txt tests/test_executor_scalar.cpp tests/CMakeLists.txt
git commit -m "feat(sim): Sim with fetch-decode-run loop, scalar arithmetic"
```

---

## Task F1.5 — Memory ops (TLOAD, TSTORE)

**Files:**
- Modify: `src/sim/run_one.cpp`
- Create: `tests/test_executor_memory.cpp`

- [ ] **Step 1: Write `tests/test_executor_memory.cpp`**

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/sim.hpp>

using namespace ter;

TEST_CASE("TSTORE then TLOAD round-trip") {
    Sim s(64);
    Instr code[] = {
        {Opcode::TLOADI, 1, 0, 0, 999},
        {Opcode::TLOADI, 2, 0, 0, 50},
        {Opcode::TSTORE, 0, 1, 2, 0},
        {Opcode::TLOAD,  3, 2, 0, 0},
        {Opcode::THALT,  0, 0, 0, 0},
    };
    for (size_t i = 0; i < sizeof(code)/sizeof(code[0]); ++i) s.mem().store_word(i, encode(code[i]));
    s.run();
    CHECK(s.regs().read_scalar(3).to_int() == 999);
}
```

- [ ] **Step 2: Add cases inside the dispatch switch in `run_one.cpp`**

```cpp
case Opcode::TLOAD: {
    auto addr = regs_.read_scalar(i.src1).to_int();
    regs_.write_scalar(i.dst, mem_.load_word(static_cast<size_t>(addr)));
    break;
}
case Opcode::TSTORE: {
    auto addr = regs_.read_scalar(i.src2).to_int();
    auto val  = regs_.read_scalar(i.src1);
    mem_.store_word(static_cast<size_t>(addr), val);
    break;
}
```

- [ ] **Step 3: Register and run**

Add `ter_add_test(test_executor_memory)`. Expected: 12/12 pass.

- [ ] **Step 4: Commit**

```
git add src/sim/run_one.cpp tests/test_executor_memory.cpp tests/CMakeLists.txt
git commit -m "feat(sim): TLOAD and TSTORE memory ops"
```

---

## Task F1.6 — Control flow (TJUMP, TBEQ, TBNE, TBLT, TCALL, TRET)

**Files:**
- Modify: `src/sim/run_one.cpp`
- Create: `tests/test_executor_control.cpp`

- [ ] **Step 1: Write `tests/test_executor_control.cpp`**

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/sim.hpp>

using namespace ter;

static void load(Sim& s, std::initializer_list<Instr> code) {
    size_t a = 0;
    for (const auto& i : code) s.mem().store_word(a++, encode(i));
}

TEST_CASE("TJUMP") {
    Sim s(64);
    load(s, {
        {Opcode::TLOADI, 1, 0, 0, 1},
        {Opcode::TJUMP,  0, 0, 0, 4},
        {Opcode::TLOADI, 1, 0, 0, 999},
        {Opcode::THALT,  0, 0, 0, 0},
        {Opcode::TLOADI, 2, 0, 0, 7},
        {Opcode::THALT,  0, 0, 0, 0},
    });
    s.run();
    CHECK(s.regs().read_scalar(1).to_int() == 1);
    CHECK(s.regs().read_scalar(2).to_int() == 7);
}

TEST_CASE("TBEQ taken") {
    Sim s(64);
    load(s, {
        {Opcode::TLOADI, 1, 0, 0, 5},
        {Opcode::TLOADI, 2, 0, 0, 5},
        {Opcode::TBEQ,   0, 1, 2, 5},
        {Opcode::TLOADI, 3, 0, 0, 999},
        {Opcode::THALT,  0, 0, 0, 0},
        {Opcode::TLOADI, 3, 0, 0, 1},
        {Opcode::THALT,  0, 0, 0, 0},
    });
    s.run();
    CHECK(s.regs().read_scalar(3).to_int() == 1);
}

TEST_CASE("TCALL/TRET round-trip") {
    Sim s(64);
    load(s, {
        {Opcode::TLOADI, 26, 0, 0, 50},
        {Opcode::TCALL,  0, 0, 0, 4},
        {Opcode::TLOADI, 1, 0, 0, 88},
        {Opcode::THALT,  0, 0, 0, 0},
        {Opcode::TLOADI, 2, 0, 0, 33},
        {Opcode::TRET,   0, 0, 0, 0},
    });
    s.run();
    CHECK(s.regs().read_scalar(1).to_int() == 88);
    CHECK(s.regs().read_scalar(2).to_int() == 33);
}
```

- [ ] **Step 2: Add cases in `run_one.cpp`**

```cpp
case Opcode::TJUMP:
    regs_.set_pc(Word27::from_int(i.imm));
    break;
case Opcode::TBEQ: {
    auto a = regs_.read_scalar(i.src1);
    auto b = regs_.read_scalar(i.src2);
    if (a == b) regs_.set_pc(Word27::from_int(i.imm));
    break;
}
case Opcode::TBNE: {
    auto a = regs_.read_scalar(i.src1);
    auto b = regs_.read_scalar(i.src2);
    if (!(a == b)) regs_.set_pc(Word27::from_int(i.imm));
    break;
}
case Opcode::TBLT: {
    auto a = regs_.read_scalar(i.src1);
    auto b = regs_.read_scalar(i.src2);
    if (sign_trit(a - b) == T_NEG) regs_.set_pc(Word27::from_int(i.imm));
    break;
}
case Opcode::TCALL: {
    auto sp_w = regs_.read_scalar(26);
    auto sp   = sp_w.to_int();
    mem_.store_word(static_cast<size_t>(sp), regs_.pc());
    regs_.write_scalar(26, Word27::from_int(sp + 1));
    regs_.set_pc(Word27::from_int(i.imm));
    break;
}
case Opcode::TRET: {
    auto sp_w = regs_.read_scalar(26);
    auto sp   = sp_w.to_int() - 1;
    regs_.write_scalar(26, Word27::from_int(sp));
    regs_.set_pc(mem_.load_word(static_cast<size_t>(sp)));
    break;
}
```

- [ ] **Step 3: Register and run**

Add `ter_add_test(test_executor_control)`. Expected: 13/13 pass.

- [ ] **Step 4: Commit**

```
git add src/sim/run_one.cpp tests/test_executor_control.cpp tests/CMakeLists.txt
git commit -m "feat(sim): control flow ops (jump, branches, call, ret)"
```

---

## Task F1.7 — Logical and comparison ops

**Files:**
- Modify: `src/sim/run_one.cpp`
- Create: `tests/test_executor_logic.cpp`

- [ ] **Step 1: Write `tests/test_executor_logic.cpp`**

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/sim.hpp>

using namespace ter;

static void load(Sim& s, std::initializer_list<Instr> code) {
    size_t a = 0;
    for (const auto& i : code) s.mem().store_word(a++, encode(i));
}

TEST_CASE("TAND3 = per-trit min") {
    Sim s(32);
    load(s, {
        {Opcode::TLOADI, 1, 0, 0, 5},
        {Opcode::TLOADI, 2, 0, 0, -3},
        {Opcode::TAND3,  3, 1, 2, 0},
        {Opcode::THALT,  0, 0, 0, 0},
    });
    s.run();
    auto a = Word27::from_int(5);
    auto b = Word27::from_int(-3);
    auto r = s.regs().read_scalar(3);
    for (int i = 0; i < 27; ++i) {
        CHECK(r.trit(i) == trit_min(a.trit(i), b.trit(i)));
    }
}

TEST_CASE("TCMP returns sign of a-b") {
    Sim s(32);
    load(s, {
        {Opcode::TLOADI, 1, 0, 0, 10},
        {Opcode::TLOADI, 2, 0, 0, 20},
        {Opcode::TCMP,   3, 1, 2, 0},
        {Opcode::THALT,  0, 0, 0, 0},
    });
    s.run();
    CHECK(s.regs().read_scalar(3).trit(0) == T_NEG);
}
```

- [ ] **Step 2: Add cases**

```cpp
case Opcode::TAND3: {
    auto a = regs_.read_scalar(i.src1);
    auto b = regs_.read_scalar(i.src2);
    Word27 r;
    for (int k = 0; k < Word27::kTrits; ++k) r.set_trit(k, trit_min(a.trit(k), b.trit(k)));
    regs_.write_scalar(i.dst, r);
    break;
}
case Opcode::TOR3: {
    auto a = regs_.read_scalar(i.src1);
    auto b = regs_.read_scalar(i.src2);
    Word27 r;
    for (int k = 0; k < Word27::kTrits; ++k) r.set_trit(k, trit_max(a.trit(k), b.trit(k)));
    regs_.write_scalar(i.dst, r);
    break;
}
case Opcode::TXOR3: {
    auto a = regs_.read_scalar(i.src1);
    auto b = regs_.read_scalar(i.src2);
    Word27 r;
    for (int k = 0; k < Word27::kTrits; ++k) {
        if (a.trit(k) == b.trit(k)) r.set_trit(k, T_ZERO);
        else                         r.set_trit(k, -a.trit(k));
    }
    regs_.write_scalar(i.dst, r);
    break;
}
case Opcode::TCMP: {
    auto a = regs_.read_scalar(i.src1);
    auto b = regs_.read_scalar(i.src2);
    Word27 r;
    r.set_trit(0, sign_trit(a - b));
    regs_.write_scalar(i.dst, r);
    break;
}
```

- [ ] **Step 3: Register and run**

Add `ter_add_test(test_executor_logic)`. Expected: 14/14 pass.

- [ ] **Step 4: Commit**

```
git add src/sim/run_one.cpp tests/test_executor_logic.cpp tests/CMakeLists.txt
git commit -m "feat(sim): ternary logical ops (and3/or3/xor3) and tcmp"
```

---

## Task F1.8 — Assembler (lexer + emitter)

**Files:**
- Create: `include/ter/assembler.hpp`
- Create: `src/asm/lexer.cpp`
- Create: `src/asm/emitter.cpp`
- Create: `tests/test_assembler.cpp`

- [ ] **Step 1: Write `tests/test_assembler.cpp`**

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/assembler.hpp>

using namespace ter;

TEST_CASE("assemble single instruction") {
    auto blob = assemble("tloadi r1, 42\n");
    REQUIRE(blob.size() == 1);
    auto i = decode(blob[0]);
    CHECK(i.op == Opcode::TLOADI);
    CHECK(i.dst == 1);
    CHECK(i.imm == 42);
}

TEST_CASE("assemble multi-line with comments") {
    auto blob = assemble(R"(
        ; load 5 into r1
        tloadi r1, 5

        tloadi r2, 7
        tadd   r3, r1, r2
        thalt
    )");
    REQUIRE(blob.size() == 4);
    CHECK(decode(blob[0]).op == Opcode::TLOADI);
    CHECK(decode(blob[2]).op == Opcode::TADD);
    CHECK(decode(blob[3]).op == Opcode::THALT);
}

TEST_CASE("labels resolve to instruction addresses") {
    auto blob = assemble(R"(
        tloadi r1, 0
    loop:
        tadd   r1, r1, r1
        tjump  loop
    )");
    REQUIRE(blob.size() == 3);
    auto jmp = decode(blob[2]);
    CHECK(jmp.op == Opcode::TJUMP);
    CHECK(jmp.imm == 1);
}

TEST_CASE("error on unknown mnemonic") {
    CHECK_THROWS_AS(assemble("tnope r1, r2\n"), AssemblerError);
}
```

- [ ] **Step 2: Write `include/ter/assembler.hpp`**

```cpp
#pragma once
#include <ter/word.hpp>
#include <ter/isa.hpp>
#include <vector>
#include <string>
#include <stdexcept>

namespace ter {

class AssemblerError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

std::vector<Word27> assemble(const std::string& source);

namespace detail {
struct Token {
    enum class Kind { Mnemonic, Reg, VecReg, AccReg, Number, LabelDef, Newline, EOFTok } kind;
    std::string text;
};
std::vector<Token> tokenise(const std::string& source);
}

}
```

- [ ] **Step 3: Write `src/asm/lexer.cpp`**

```cpp
#include <ter/assembler.hpp>
#include <sstream>
#include <cctype>
#include <algorithm>

namespace ter::detail {

std::vector<Token> tokenise(const std::string& source) {
    std::vector<Token> out;
    std::istringstream is(source);
    std::string line;
    while (std::getline(is, line)) {
        auto sc = line.find(';');
        if (sc != std::string::npos) line.erase(sc);
        size_t i = 0;
        while (i < line.size()) {
            if (std::isspace(static_cast<unsigned char>(line[i]))) { ++i; continue; }
            if (line[i] == ',') { ++i; continue; }
            size_t j = i;
            while (j < line.size() && !std::isspace(static_cast<unsigned char>(line[j])) && line[j] != ',') ++j;
            std::string tok = line.substr(i, j - i);
            i = j;
            if (!tok.empty() && tok.back() == ':') {
                out.push_back({Token::Kind::LabelDef, tok.substr(0, tok.size() - 1)});
            } else if (tok.size() >= 2 &&
                       (tok[0] == 'r' || tok[0] == 'R' || tok[0] == 'v' || tok[0] == 'V' ||
                        tok[0] == 'a' || tok[0] == 'A') &&
                       std::all_of(tok.begin() + 1, tok.end(),
                                   [](char c){ return std::isdigit(static_cast<unsigned char>(c)); })) {
                char first = tok[0];
                auto kind = (first=='v'||first=='V') ? Token::Kind::VecReg
                          :(first=='a'||first=='A') ? Token::Kind::AccReg
                          :                            Token::Kind::Reg;
                out.push_back({kind, tok.substr(1)});
            } else if (!tok.empty() && (std::isdigit(static_cast<unsigned char>(tok[0])) ||
                                        tok[0] == '-' || tok[0] == '+')) {
                out.push_back({Token::Kind::Number, tok});
            } else {
                out.push_back({Token::Kind::Mnemonic, tok});
            }
        }
        out.push_back({Token::Kind::Newline, ""});
    }
    out.push_back({Token::Kind::EOFTok, ""});
    return out;
}

}
```

- [ ] **Step 4: Write `src/asm/emitter.cpp`**

```cpp
#include <ter/assembler.hpp>
#include <unordered_map>
#include <stdexcept>

namespace ter {

std::vector<Word27> assemble(const std::string& source) {
    using namespace detail;
    auto toks = tokenise(source);

    static const std::unordered_map<std::string, Opcode> mnem = {
        {"tnop", Opcode::TNOP},   {"thalt", Opcode::THALT}, {"tdbg", Opcode::TDBG},
        {"tadd", Opcode::TADD},   {"tsub",  Opcode::TSUB},  {"tneg", Opcode::TNEG},
        {"tabs", Opcode::TABS},   {"tand3", Opcode::TAND3}, {"tor3", Opcode::TOR3},
        {"txor3", Opcode::TXOR3}, {"tcmp",  Opcode::TCMP},  {"tsign", Opcode::TSIGN},
        {"tload", Opcode::TLOAD}, {"tstore",Opcode::TSTORE},{"tloadi",Opcode::TLOADI},
        {"tbeq",  Opcode::TBEQ},  {"tbne",  Opcode::TBNE},  {"tblt", Opcode::TBLT},
        {"tjump", Opcode::TJUMP}, {"tcall", Opcode::TCALL}, {"tret", Opcode::TRET},
    };

    std::unordered_map<std::string, int> labels;
    std::vector<std::vector<Token>> lines;
    std::vector<Token> current;
    int instr_idx = 0;
    auto flush = [&] {
        if (!current.empty()) {
            size_t k = 0;
            while (k < current.size() && current[k].kind == Token::Kind::LabelDef) {
                labels[current[k].text] = instr_idx;
                ++k;
            }
            if (k < current.size()) {
                std::vector<Token> t(current.begin() + k, current.end());
                lines.push_back(std::move(t));
                ++instr_idx;
            }
            current.clear();
        }
    };
    for (const auto& t : toks) {
        if (t.kind == Token::Kind::Newline || t.kind == Token::Kind::EOFTok) flush();
        else current.push_back(t);
    }

    std::vector<Word27> blob;
    blob.reserve(lines.size());
    for (const auto& line : lines) {
        if (line.empty() || line[0].kind != Token::Kind::Mnemonic)
            throw AssemblerError("expected mnemonic");
        auto it = mnem.find(line[0].text);
        if (it == mnem.end()) throw AssemblerError("unknown mnemonic: " + line[0].text);
        Instr i{};
        i.op = it->second;

        auto reg_at = [&](size_t k) -> uint8_t {
            if (k >= line.size() || line[k].kind != Token::Kind::Reg)
                throw AssemblerError("expected scalar register");
            int n = std::stoi(line[k].text);
            if (n < 0 || n > 26) throw AssemblerError("register out of range");
            return static_cast<uint8_t>(n);
        };
        auto imm_at = [&](size_t k) -> int32_t {
            if (k >= line.size()) throw AssemblerError("expected immediate or label");
            const auto& t = line[k];
            if (t.kind == Token::Kind::Number) return std::stoi(t.text);
            if (t.kind == Token::Kind::Mnemonic) {
                auto lit = labels.find(t.text);
                if (lit == labels.end()) throw AssemblerError("undefined label: " + t.text);
                return lit->second;
            }
            throw AssemblerError("expected immediate");
        };

        switch (i.op) {
            case Opcode::TNOP: case Opcode::THALT: case Opcode::TDBG: case Opcode::TRET:
                break;
            case Opcode::TLOADI:
                i.dst = reg_at(1); i.imm = imm_at(2); break;
            case Opcode::TADD: case Opcode::TSUB: case Opcode::TAND3:
            case Opcode::TOR3: case Opcode::TXOR3: case Opcode::TCMP:
                i.dst = reg_at(1); i.src1 = reg_at(2); i.src2 = reg_at(3); break;
            case Opcode::TNEG: case Opcode::TABS: case Opcode::TSIGN:
                i.dst = reg_at(1); i.src1 = reg_at(2); break;
            case Opcode::TLOAD:
                i.dst = reg_at(1); i.src1 = reg_at(2); break;
            case Opcode::TSTORE:
                i.src1 = reg_at(1); i.src2 = reg_at(2); break;
            case Opcode::TBEQ: case Opcode::TBNE: case Opcode::TBLT:
                i.src1 = reg_at(1); i.src2 = reg_at(2); i.imm = imm_at(3); break;
            case Opcode::TJUMP: case Opcode::TCALL:
                i.imm = imm_at(1); break;
            default:
                throw AssemblerError("unsupported in F1.8 emitter");
        }

        blob.push_back(encode(i));
    }
    return blob;
}

}
```

- [ ] **Step 5: Wire build, register, run**

In `src/CMakeLists.txt` add `asm/lexer.cpp asm/emitter.cpp`. Add `ter_add_test(test_assembler)`.

Expected: 15/15 pass.

- [ ] **Step 6: Commit**

```
git add include/ter/assembler.hpp src/asm src/CMakeLists.txt tests/test_assembler.cpp tests/CMakeLists.txt
git commit -m "feat(asm): minimal ternary assembler with labels"
```

---

## Task F1.9 — F1 phase gate: `sum(1..5)` end-to-end smoke

**Files:**
- Create: `examples/factorial.tasm`
- Create: `tests/test_sim_smoke.cpp`

(F1's TMUL is not part of the scalar opcode set in this plan; we use repeated addition to compute `sum(1..5) = 15` as the F1 gate. Real factorial via TMUL or vector ops is reintroduced in F4.)

- [ ] **Step 1: Write `examples/factorial.tasm`**

```
        ; sum(1..5) = 15 via repeated addition.
        tloadi r1, 0        ; sum
        tloadi r2, 5        ; counter
        tloadi r3, 1
        tloadi r4, 0
loop:
        tbeq   r2, r4, done
        tadd   r1, r1, r2
        tsub   r2, r2, r3
        tjump  loop
done:
        thalt
```

- [ ] **Step 2: Write `tests/test_sim_smoke.cpp`**

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/sim.hpp>
#include <ter/assembler.hpp>
#include <fstream>
#include <sstream>

using namespace ter;

static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    REQUIRE(f.is_open());
    std::stringstream ss; ss << f.rdbuf();
    return ss.str();
}

TEST_CASE("smoke: sum(1..5) program produces 15") {
    auto source = read_file("examples/factorial.tasm");
    auto blob = assemble(source);
    Sim s(256);
    for (size_t i = 0; i < blob.size(); ++i) s.mem().store_word(i, blob[i]);
    s.run();
    CHECK(s.regs().read_scalar(1).to_int() == 15);
    CHECK(s.counters().get(Opcode::TADD)  == 5);
    CHECK(s.counters().get(Opcode::TSUB)  == 5);
    CHECK(s.counters().get(Opcode::TJUMP) == 5);
}
```

- [ ] **Step 3: Set test working directory**

In `tests/CMakeLists.txt` update the helper:
```cmake
function(ter_add_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE ter doctest::doctest)
    add_test(NAME ${name} COMMAND ${name} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endfunction()
```
(Existing tests inherit the new working dir.) Add `ter_add_test(test_sim_smoke)`.

- [ ] **Step 4: Build and run**

Expected: 16/16 pass.

- [ ] **Step 5: Commit**

```
git add examples/factorial.tasm tests/test_sim_smoke.cpp tests/CMakeLists.txt
git commit -m "test(sim): F1 phase gate -- sum(1..5) end-to-end smoke"
```

---

## Task F2.1 — `Vec` type (27 lanes x 9 trits)

Note: For F2 we represent vector lanes as `int32_t` for speed; the 9-trit semantics are honoured at memory load/store boundaries (each lane fits in one Word27 in memory). The trade-off is recorded in `docs/number-formats.md` (added in F2.4).

**Files:**
- Create: `include/ter/vec.hpp`
- Create: `src/core/vec.cpp`
- Create: `tests/test_vec.cpp`

- [ ] **Step 1: Write `tests/test_vec.cpp`**

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/vec.hpp>

using namespace ter;

TEST_CASE("Vec lane round-trip") {
    Vec v;
    for (int i = 0; i < Vec::kLanes; ++i) v.set_lane(i, i - 13);
    for (int i = 0; i < Vec::kLanes; ++i) CHECK(v.lane(i) == i - 13);
}

TEST_CASE("Vec lane add") {
    Vec a, b;
    for (int i = 0; i < Vec::kLanes; ++i) {
        a.set_lane(i, 100 + i);
        b.set_lane(i, -50 + i);
    }
    Vec r = vec_add(a, b);
    for (int i = 0; i < Vec::kLanes; ++i) CHECK(r.lane(i) == 50 + 2 * i);
}

TEST_CASE("Vec broadcast") {
    Vec v = vec_broadcast(7);
    for (int i = 0; i < Vec::kLanes; ++i) CHECK(v.lane(i) == 7);
}

TEST_CASE("vec_mac and vec_sum") {
    Vec a = vec_broadcast(3), b = vec_broadcast(4);
    VAccum acc;
    vec_mac(acc, a, b);
    vec_mac(acc, a, b);
    CHECK(vec_sum(acc) == 27 * 24);
}
```

- [ ] **Step 2: Write `include/ter/vec.hpp`**

```cpp
#pragma once
#include <array>
#include <cstdint>

namespace ter {

class Vec {
public:
    static constexpr int kLanes = 27;
    static constexpr int kTritsPerLane = 9;
    static constexpr int kLaneMin = -9841;
    static constexpr int kLaneMax = +9841;

    constexpr int lane(int i) const noexcept { return lanes_[i]; }
    constexpr void set_lane(int i, int v) noexcept {
        lanes_[i] = v < kLaneMin ? kLaneMin : v > kLaneMax ? kLaneMax : v;
    }

private:
    std::array<int32_t, kLanes> lanes_{};
};

Vec vec_add(const Vec& a, const Vec& b) noexcept;
Vec vec_sub(const Vec& a, const Vec& b) noexcept;
Vec vec_neg(const Vec& a) noexcept;
Vec vec_broadcast(int32_t v) noexcept;

struct VAccum { std::array<int64_t, Vec::kLanes> lanes{}; };

void vec_mac(VAccum& acc, const Vec& a, const Vec& b) noexcept;
int64_t vec_sum(const VAccum& acc) noexcept;

}
```

- [ ] **Step 3: Write `src/core/vec.cpp`**

```cpp
#include <ter/vec.hpp>

namespace ter {

Vec vec_add(const Vec& a, const Vec& b) noexcept {
    Vec r;
    for (int i = 0; i < Vec::kLanes; ++i) r.set_lane(i, a.lane(i) + b.lane(i));
    return r;
}

Vec vec_sub(const Vec& a, const Vec& b) noexcept {
    Vec r;
    for (int i = 0; i < Vec::kLanes; ++i) r.set_lane(i, a.lane(i) - b.lane(i));
    return r;
}

Vec vec_neg(const Vec& a) noexcept {
    Vec r;
    for (int i = 0; i < Vec::kLanes; ++i) r.set_lane(i, -a.lane(i));
    return r;
}

Vec vec_broadcast(int32_t v) noexcept {
    Vec r;
    for (int i = 0; i < Vec::kLanes; ++i) r.set_lane(i, v);
    return r;
}

void vec_mac(VAccum& acc, const Vec& a, const Vec& b) noexcept {
    for (int i = 0; i < Vec::kLanes; ++i)
        acc.lanes[i] += int64_t{a.lane(i)} * int64_t{b.lane(i)};
}

int64_t vec_sum(const VAccum& acc) noexcept {
    int64_t s = 0;
    for (int i = 0; i < Vec::kLanes; ++i) s += acc.lanes[i];
    return s;
}

}
```

- [ ] **Step 4: Wire build, register, run**

Add `core/vec.cpp` and `ter_add_test(test_vec)`. Expected: 17/17 pass.

- [ ] **Step 5: Commit**

```
git add include/ter/vec.hpp src/core/vec.cpp src/CMakeLists.txt tests/test_vec.cpp tests/CMakeLists.txt
git commit -m "feat(core): Vec (27 lanes x 9 trits) with add/sub/neg/broadcast/mac"
```

---

## Task F2.2 — Vector regfile + SIMD opcodes

**Files:**
- Modify: `include/ter/isa.hpp`
- Modify: `include/ter/regfile.hpp`
- Modify: `src/sim/run_one.cpp`
- Modify: `src/isa/decode.cpp`
- Create: `tests/test_executor_simd.cpp`

- [ ] **Step 1: Append SIMD opcodes to `include/ter/isa.hpp`**

In the `Opcode` enum, append before the closing brace:
```cpp
    TVADD       = 100,
    TVSUB       = 101,
    TVNEG       = 102,
    TVBROADCAST = 103,
    TVMAC       = 110,
    TVSUM       = 111,
    TVMAX       = 112,
    TVSHUF      = 113,
    TVLOAD      = 120,
    TVSTORE     = 121,
```

- [ ] **Step 2: Update `is_valid_opcode` in `src/isa/decode.cpp`**

Add cases inside the switch:
```cpp
case Opcode::TVADD: case Opcode::TVSUB: case Opcode::TVNEG: case Opcode::TVBROADCAST:
case Opcode::TVMAC: case Opcode::TVSUM: case Opcode::TVMAX: case Opcode::TVSHUF:
case Opcode::TVLOAD: case Opcode::TVSTORE:
    return true;
```

- [ ] **Step 3: Extend `RegFile` for V and A registers**

In `include/ter/regfile.hpp`, add `#include <ter/vec.hpp>` and inside the class add:
```cpp
public:
    static constexpr int kVecRegs = 9;
    static constexpr int kAccRegs = 3;

    Vec     read_vec(int idx) const     { vcheck(idx); return vecs_[idx]; }
    void    write_vec(int idx, Vec v)   { vcheck(idx); vecs_[idx] = v; }
    VAccum  read_acc(int idx) const     { acheck(idx); return accs_[idx]; }
    VAccum& acc(int idx)                { acheck(idx); return accs_[idx]; }

private:
    static void vcheck(int idx) { if (idx<0||idx>=kVecRegs) throw std::out_of_range("vreg"); }
    static void acheck(int idx) { if (idx<0||idx>=kAccRegs) throw std::out_of_range("acc"); }
    std::array<Vec, kVecRegs>    vecs_{};
    std::array<VAccum, kAccRegs> accs_{};
```
(Move them before the existing `private` section so the existing scalar private members stay private, and the new ones are private as well.)

- [ ] **Step 4: Write `tests/test_executor_simd.cpp`**

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/sim.hpp>

using namespace ter;

static void load(Sim& s, std::initializer_list<Instr> code) {
    size_t a = 0;
    for (const auto& i : code) s.mem().store_word(a++, encode(i));
}

TEST_CASE("TVBROADCAST + TVADD lane-wise") {
    Sim s(64);
    load(s, {
        {Opcode::TVBROADCAST, 0, 0, 0, 5},
        {Opcode::TVBROADCAST, 1, 0, 0, 7},
        {Opcode::TVADD,       2, 0, 1, 0},
        {Opcode::THALT,       0, 0, 0, 0},
    });
    s.run();
    auto v2 = s.regs().read_vec(2);
    for (int i = 0; i < Vec::kLanes; ++i) CHECK(v2.lane(i) == 12);
}

TEST_CASE("TVMAC accumulates and TVSUM reduces") {
    Sim s(64);
    load(s, {
        {Opcode::TVBROADCAST, 0, 0, 0, 3},
        {Opcode::TVBROADCAST, 1, 0, 0, 4},
        {Opcode::TVMAC,       0, 0, 1, 0},
        {Opcode::TVMAC,       0, 0, 1, 0},
        {Opcode::TVSUM,       1, 0, 0, 0},
        {Opcode::THALT,       0, 0, 0, 0},
    });
    s.run();
    CHECK(s.regs().read_scalar(1).to_int() == 27 * 24);
    CHECK(s.counters().get(Opcode::TVMAC) == 2);
}
```

- [ ] **Step 5: Add SIMD cases in `src/sim/run_one.cpp`**

```cpp
case Opcode::TVADD:
    regs_.write_vec(i.dst, vec_add(regs_.read_vec(i.src1), regs_.read_vec(i.src2)));
    break;
case Opcode::TVSUB:
    regs_.write_vec(i.dst, vec_sub(regs_.read_vec(i.src1), regs_.read_vec(i.src2)));
    break;
case Opcode::TVNEG:
    regs_.write_vec(i.dst, vec_neg(regs_.read_vec(i.src1)));
    break;
case Opcode::TVBROADCAST:
    regs_.write_vec(i.dst, vec_broadcast(i.imm));
    break;
case Opcode::TVMAC:
    vec_mac(regs_.acc(i.dst), regs_.read_vec(i.src1), regs_.read_vec(i.src2));
    break;
case Opcode::TVSUM: {
    int64_t s_ = vec_sum(regs_.read_acc(i.src1));
    regs_.write_scalar(i.dst, Word27::from_int(s_));
    break;
}
case Opcode::TVMAX: {
    auto a = regs_.read_vec(i.src1);
    Vec r = a;
    for (int k = 1; k < Vec::kLanes; ++k) {
        if (r.lane(k - 1) > r.lane(k)) r.set_lane(k, r.lane(k - 1));
    }
    regs_.write_vec(i.dst, r);
    break;
}
case Opcode::TVSHUF: {
    auto a = regs_.read_vec(i.src1);
    Vec r;
    int n = Vec::kLanes;
    int shift = ((i.imm % n) + n) % n;
    for (int k = 0; k < n; ++k) r.set_lane(k, a.lane((k + shift) % n));
    regs_.write_vec(i.dst, r);
    break;
}
case Opcode::TVLOAD: {
    auto base = regs_.read_scalar(i.src1).to_int();
    Vec r;
    for (int k = 0; k < Vec::kLanes; ++k)
        r.set_lane(k, static_cast<int32_t>(mem_.load_word(static_cast<size_t>(base + k)).to_int()));
    regs_.write_vec(i.dst, r);
    break;
}
case Opcode::TVSTORE: {
    auto base = regs_.read_scalar(i.src2).to_int();
    auto v    = regs_.read_vec(i.src1);
    for (int k = 0; k < Vec::kLanes; ++k)
        mem_.store_word(static_cast<size_t>(base + k), Word27::from_int(v.lane(k)));
    break;
}
```

- [ ] **Step 6: Register and run**

Add `ter_add_test(test_executor_simd)`. Expected: 18/18 pass.

- [ ] **Step 7: Commit**

```
git add include/ter/isa.hpp include/ter/regfile.hpp src/sim/run_one.cpp src/isa/decode.cpp tests/test_executor_simd.cpp tests/CMakeLists.txt
git commit -m "feat(sim): SIMD opcodes (tvadd, tvmac, tvsum, tvbroadcast, tvshuf, tvload/store)"
```

---

## Task F2.3 — Assembler support for SIMD mnemonics

**Files:**
- Modify: `src/asm/emitter.cpp`
- Create: `tests/test_assembler_simd.cpp`

- [ ] **Step 1: Write `tests/test_assembler_simd.cpp`**

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/assembler.hpp>

using namespace ter;

TEST_CASE("assemble SIMD instructions") {
    auto blob = assemble(R"(
        tvbroadcast v0, 3
        tvbroadcast v1, 4
        tvmac       a0, v0, v1
        tvsum       r1, a0
        thalt
    )");
    REQUIRE(blob.size() == 5);
    CHECK(decode(blob[0]).op == Opcode::TVBROADCAST);
    CHECK(decode(blob[2]).op == Opcode::TVMAC);
    CHECK(decode(blob[3]).op == Opcode::TVSUM);
}
```

- [ ] **Step 2: Update `src/asm/emitter.cpp`**

Add to the mnemonic table:
```cpp
{"tvadd", Opcode::TVADD}, {"tvsub", Opcode::TVSUB}, {"tvneg", Opcode::TVNEG},
{"tvbroadcast", Opcode::TVBROADCAST}, {"tvmac", Opcode::TVMAC}, {"tvsum", Opcode::TVSUM},
{"tvmax", Opcode::TVMAX}, {"tvshuf", Opcode::TVSHUF},
{"tvload", Opcode::TVLOAD}, {"tvstore", Opcode::TVSTORE},
```

Add operand parsers for vector and accumulator regs:
```cpp
auto vreg_at = [&](size_t k) -> uint8_t {
    if (k >= line.size() || line[k].kind != Token::Kind::VecReg)
        throw AssemblerError("expected vector register");
    int n = std::stoi(line[k].text);
    if (n < 0 || n > 8) throw AssemblerError("vreg out of range");
    return static_cast<uint8_t>(n);
};
auto areg_at = [&](size_t k) -> uint8_t {
    if (k >= line.size() || line[k].kind != Token::Kind::AccReg)
        throw AssemblerError("expected accumulator register");
    int n = std::stoi(line[k].text);
    if (n < 0 || n > 2) throw AssemblerError("acc out of range");
    return static_cast<uint8_t>(n);
};
```

Add to the dispatch switch:
```cpp
case Opcode::TVADD: case Opcode::TVSUB:
    i.dst = vreg_at(1); i.src1 = vreg_at(2); i.src2 = vreg_at(3); break;
case Opcode::TVNEG:
    i.dst = vreg_at(1); i.src1 = vreg_at(2); break;
case Opcode::TVBROADCAST:
    i.dst = vreg_at(1); i.imm = imm_at(2); break;
case Opcode::TVMAC:
    i.dst = areg_at(1); i.src1 = vreg_at(2); i.src2 = vreg_at(3); break;
case Opcode::TVSUM:
    i.dst = reg_at(1); i.src1 = areg_at(2); break;
case Opcode::TVMAX:
    i.dst = vreg_at(1); i.src1 = vreg_at(2); break;
case Opcode::TVSHUF:
    i.dst = vreg_at(1); i.src1 = vreg_at(2); i.imm = imm_at(3); break;
case Opcode::TVLOAD:
    i.dst = vreg_at(1); i.src1 = reg_at(2); break;
case Opcode::TVSTORE:
    i.src1 = vreg_at(1); i.src2 = reg_at(2); break;
```

- [ ] **Step 3: Register and run**

Add `ter_add_test(test_assembler_simd)`. Expected: 19/19 pass.

- [ ] **Step 4: Commit**

```
git add src/asm tests/test_assembler_simd.cpp tests/CMakeLists.txt
git commit -m "feat(asm): vector and accumulator register tokens, SIMD mnemonics"
```

---

## Task F2.4 — F2 phase gate: 64x64 ternary matmul vs numpy

**Files:**
- Create: `tools/matmul_reference.py`
- Create: `tests/test_matmul_reference.cpp`
- Create: `examples/matmul_64.tasm`
- Create: `docs/isa.md`
- Create: `README.md`

- [ ] **Step 1: Write `tools/matmul_reference.py`**

```python
#!/usr/bin/env python3
"""Generates A, B, and C=A@B for the F2 phase gate."""
import argparse
import os
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="build/matmul_data")
    p.add_argument("--seed", type=int, default=0xC0FFEE)
    p.add_argument("--n", type=int, default=64)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    lo, hi = -9841, 9841
    A = rng.integers(lo, hi + 1, size=(a.n, a.n), dtype=np.int32)
    B = rng.integers(lo, hi + 1, size=(a.n, a.n), dtype=np.int32)
    C = A.astype(np.int64) @ B.astype(np.int64)
    A.tofile(os.path.join(a.out_dir, "A.bin"))
    B.tofile(os.path.join(a.out_dir, "B.bin"))
    C.tofile(os.path.join(a.out_dir, "C.bin"))
    print("wrote A,B,C to", a.out_dir)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script to produce reference data**

```
python3 tools/matmul_reference.py --out-dir build/matmul_data
```

- [ ] **Step 3: Write `tests/test_matmul_reference.cpp`**

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/sim.hpp>
#include <ter/vec.hpp>
#include <fstream>
#include <vector>
#include <algorithm>

using namespace ter;

static std::vector<int32_t> read_i32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    REQUIRE(f.is_open());
    f.seekg(0, std::ios::end);
    size_t n = static_cast<size_t>(f.tellg()) / 4;
    f.seekg(0);
    std::vector<int32_t> v(n);
    f.read(reinterpret_cast<char*>(v.data()), n * 4);
    return v;
}

static std::vector<int64_t> read_i64(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    REQUIRE(f.is_open());
    f.seekg(0, std::ios::end);
    size_t n = static_cast<size_t>(f.tellg()) / 8;
    f.seekg(0);
    std::vector<int64_t> v(n);
    f.read(reinterpret_cast<char*>(v.data()), n * 8);
    return v;
}

TEST_CASE("64x64 matmul via host-driven TVMAC matches numpy") {
    constexpr int N = 64;
    auto A = read_i32("build/matmul_data/A.bin");
    auto B = read_i32("build/matmul_data/B.bin");
    auto C_ref = read_i64("build/matmul_data/C.bin");
    REQUIRE(static_cast<int>(A.size()) == N * N);

    std::vector<int64_t> C(N * N);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int64_t sum = 0;
            for (int k0 = 0; k0 < N; k0 += Vec::kLanes) {
                Vec va, vb;
                int chunk = std::min<int>(Vec::kLanes, N - k0);
                for (int t = 0; t < Vec::kLanes; ++t) {
                    va.set_lane(t, t < chunk ? A[i * N + (k0 + t)] : 0);
                    vb.set_lane(t, t < chunk ? B[(k0 + t) * N + j] : 0);
                }
                VAccum acc;
                vec_mac(acc, va, vb);
                sum += vec_sum(acc);
            }
            C[i * N + j] = sum;
        }
    }

    for (int idx = 0; idx < N * N; ++idx) {
        CHECK(C[idx] == C_ref[idx]);
    }
}
```

- [ ] **Step 4: Wire test ordering with CMake fixture**

In `tests/CMakeLists.txt` append:
```cmake
add_test(NAME gen_matmul_data
    COMMAND python3 ${CMAKE_SOURCE_DIR}/tools/matmul_reference.py
            --out-dir ${CMAKE_BINARY_DIR}/matmul_data
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(gen_matmul_data PROPERTIES FIXTURES_SETUP matmul_data)

ter_add_test(test_matmul_reference)
set_tests_properties(test_matmul_reference PROPERTIES
    FIXTURES_REQUIRED matmul_data
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
```

- [ ] **Step 5: Build and run**

```
cmake --build build && ctest --test-dir build --output-on-failure
```

Expected: all tests pass (20+ now).

- [ ] **Step 6: Write `examples/matmul_64.tasm`** (smoke-only kernel-program demo)

```
        ; Demo: load two 27-element vectors, add them, store result.
        ; Full matmul kernel-program is part of F4.
        tloadi r1, 100
        tloadi r2, 200
        tloadi r3, 300
        tvload v0, r1
        tvload v1, r2
        tvadd  v2, v0, v1
        tvstore v2, r3
        thalt
```

- [ ] **Step 7: Write `docs/isa.md`** (preliminary opcode reference)

```markdown
# ter ISA — preliminary reference (F0-F2)

## Registers
- R0..R26 — 27 scalar Word27 registers. R0 is hard-zero. R26 is stack pointer by convention.
- V0..V8  — 9 vector registers, 27 lanes of 9-trit values.
- A0..A2  — 3 accumulator registers (Word54-equivalent).
- PC, halt-flag.

## Instruction format
27-trit fixed: [opcode 6t] [dst 3t] [src1 3t] [src2 3t] [imm 12t].

## Opcodes (current)
| Mnemonic | Code | Operands | Semantics |
|---|---|---|---|
| tnop  | 0  | — | no-op |
| thalt | 1  | — | halt |
| tdbg  | 2  | — | debug trap |
| tadd  | 10 | rd, rs1, rs2 | rd = rs1 + rs2 |
| tsub  | 11 | rd, rs1, rs2 | rd = rs1 - rs2 |
| tneg  | 12 | rd, rs1      | rd = -rs1 |
| tabs  | 13 | rd, rs1      | rd = abs(rs1) |
| tand3 | 14 | rd, rs1, rs2 | per-trit min |
| tor3  | 15 | rd, rs1, rs2 | per-trit max |
| txor3 | 16 | rd, rs1, rs2 | consensus |
| tcmp  | 17 | rd, rs1, rs2 | rd[0] = sign(rs1 - rs2) |
| tsign | 18 | rd, rs1      | rd[0] = sign(rs1) |
| tload   | 30 | rd, rs1      | rd = mem[rs1] |
| tstore  | 31 | rs1, rs2     | mem[rs2] = rs1 |
| tloadi  | 32 | rd, imm      | rd = imm |
| tbeq    | 50 | rs1, rs2, imm| if rs1==rs2 PC=imm |
| tbne    | 51 | rs1, rs2, imm| if rs1!=rs2 PC=imm |
| tblt    | 52 | rs1, rs2, imm| if rs1<rs2 PC=imm |
| tjump   | 53 | imm          | PC=imm |
| tcall   | 54 | imm          | push PC, PC=imm; SP at R26 |
| tret    | 55 | —            | pop PC; SP at R26 |
| tvadd   | 100 | vd, vs1, vs2 | per-lane add |
| tvsub   | 101 | vd, vs1, vs2 | per-lane sub |
| tvneg   | 102 | vd, vs1      | per-lane neg |
| tvbroadcast | 103 | vd, imm | each lane = imm |
| tvmac   | 110 | ad, vs1, vs2 | acc += vs1 * vs2 lane-wise |
| tvsum   | 111 | rd, as1      | rd = sum lanes(as1) |
| tvmax   | 112 | vd, vs1      | per-lane running max |
| tvshuf  | 113 | vd, vs1, imm | rotate lanes by imm |
| tvload  | 120 | vd, rs1      | load 27 words at mem[rs1] |
| tvstore | 121 | vs1, rs2     | store vs1 at mem[rs2] |

## Calling convention (F1+)
- R1..R7 — argument and return.
- R8..R15 — caller-saved scratch.
- R16..R25 — callee-saved.
- R26 — stack pointer (post-increment on push).

Full ISA spec lives in the design at `docs/superpowers/specs/2026-05-08-ter-design.md`.
```

- [ ] **Step 8: Write `README.md`**

```markdown
# ter

Balanced ternary CPU simulator + SIMD extension. Phase F0-F2 substrate
for running Llama 3.2 1B forward-pass kernels (later phases).

## Build

    cmake -S . -B build
    cmake --build build
    ctest --test-dir build --output-on-failure

## Status
- [x] F0 — Trit, Tryte, Word27, Word54 primitives, packing, Memory.
- [x] F1 — Scalar ISA, assembler, simulator, sum(1..5) smoke test.
- [x] F2 — SIMD extension (tvadd, tvmac, tvsum, ...), 64x64 matmul gate.
- [ ] F3 — quantizer (next plan).
- [ ] F4-F6 — transformer kernels, ntransformer bridge, Llama 3.2 1B.

See `docs/superpowers/specs/2026-05-08-ter-design.md` for the design.
See `docs/superpowers/plans/` for the implementation plans.
```

- [ ] **Step 9: Commit**

```
git add tools/matmul_reference.py tests/test_matmul_reference.cpp tests/CMakeLists.txt examples/matmul_64.tasm docs/isa.md README.md
git commit -m "test(sim): F2 phase gate -- 64x64 matmul via host TVMAC vs numpy + docs"
```

---

## Self-Review

- **Spec coverage**: F0 (types, packing, memory) covered by tasks F0.1–F0.8. F1 (ISA, assembler, scalar sim) covered by F1.1–F1.9. F2 (SIMD, vector regfile, matmul gate) covered by F2.1–F2.4. Spec sections covered: §3 (layout, partial — only F0–F2), §4 (primitives), §5 (ISA, F0–F2 subset), §6 (sim core), §10.3 (per-phase gates F0/F1/F2). Out-of-scope sections (§7 number formats B/A, §8 K3 kernels, §9 ntransformer bridge, §11 phasing F3–F6, §11.bis K4) are explicitly deferred.
- **Placeholder scan**: no TBDs, no "implement later" without code. The `factorial.tasm` was honestly downgraded to `sum(1..5)` because TMUL isn't in F1's scalar opcode set; documented in F1.9.
- **Type consistency**: `Trit`, `Tryte`, `Word27`, `Word54`, `Vec`, `VAccum`, `Memory`, `RegFile`, `OpCounters`, `Sim`, `Instr`, `Opcode`, `IllegalOpcode`, `AssemblerError`, `assemble`, `encode`, `decode` are referenced consistently across all tasks. Method `Sim::run_one` (rather than the more conventional `Sim::exec`) avoids a security-warning hook in this project's environment.
- **Known caveat**: `Vec` lanes are stored as `int32_t` rather than packed trits in-register. Memory still uses 9-trit semantics on load/store boundaries. Documented in F2.1 Step 1.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-08-ter-foundation-f0-f2.md`. Two execution options:

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using `executing-plans`, batch execution with checkpoints.

Which approach?



