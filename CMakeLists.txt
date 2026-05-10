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
target_compile_options(ter PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_include_directories(ter PUBLIC ${CMAKE_SOURCE_DIR}/include)
add_subdirectory(src)
add_subdirectory(vendor/ntransformer)

enable_testing()
add_subdirectory(tests)

add_executable(dump_gguf_tensors tools/dump_gguf_tensors.cpp)
target_link_libraries(dump_gguf_tensors PRIVATE nt_infra)
target_compile_options(dump_gguf_tensors PRIVATE -Wall -Wextra -Wpedantic -Werror)

# F10: K4 ring-0 backend (hosted-mode wrapper for now; freestanding port pending).
add_library(ter_k4 STATIC src/k4/ter_k4.cpp)
target_compile_options(ter_k4 PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_include_directories(ter_k4 PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(ter_k4 PUBLIC ter)
