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

add_library(ter INTERFACE)
target_include_directories(ter INTERFACE ${CMAKE_SOURCE_DIR}/include)

enable_testing()
add_subdirectory(tests)
