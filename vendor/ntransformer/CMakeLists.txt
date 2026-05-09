add_library(nt_infra STATIC
    core/types.cpp
    core/tensor.cpp
    model/config.cpp
    model/loader.cpp
)
target_include_directories(nt_infra PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(nt_infra PRIVATE -Wall -Wextra -Wpedantic -Werror)
