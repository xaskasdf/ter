function(ter_add_test name)
    add_executable(${name} ${name}.cpp)
    target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_link_libraries(${name} PRIVATE ter doctest::doctest)
    add_test(NAME ${name} COMMAND ${name} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endfunction()

ter_add_test(test_nt_types)
target_link_libraries(test_nt_types PRIVATE nt_infra)

ter_add_test(test_nt_tensor)
target_link_libraries(test_nt_tensor PRIVATE nt_infra)

ter_add_test(test_nt_config)
target_link_libraries(test_nt_config PRIVATE nt_infra)

ter_add_test(test_nt_loader)
target_link_libraries(test_nt_loader PRIVATE nt_infra)

ter_add_test(test_nt_tokenizer)
target_link_libraries(test_nt_tokenizer PRIVATE nt_infra)

ter_add_test(test_smoke)
ter_add_test(test_trit)
ter_add_test(test_tryte)
ter_add_test(test_word)
ter_add_test(test_word54)
ter_add_test(test_pack)
ter_add_test(test_memory)
ter_add_test(test_isa)
ter_add_test(test_regfile)
ter_add_test(test_counters)
ter_add_test(test_executor_scalar)
ter_add_test(test_executor_memory)
ter_add_test(test_executor_control)
ter_add_test(test_executor_logic)
ter_add_test(test_assembler)
ter_add_test(test_sim_smoke)
ter_add_test(test_vec)
ter_add_test(test_executor_simd)
ter_add_test(test_assembler_simd)
ter_add_test(test_numfmt_metadata)
ter_add_test(test_numfmt_quantize)
ter_add_test(test_numfmt_roundtrip)
ter_add_test(test_call_kernel)
ter_add_test(test_load_lut)

ter_add_test(test_kernel_matmul_b_small)
target_compile_definitions(test_kernel_matmul_b_small PRIVATE TER_KERNELS_DIR="${CMAKE_SOURCE_DIR}/src/kernels")

add_test(NAME gen_matmul_data
    COMMAND ${CMAKE_SOURCE_DIR}/.venv/bin/python
            ${CMAKE_SOURCE_DIR}/tools/matmul_reference.py
            --out-dir ${CMAKE_BINARY_DIR}/matmul_data
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(gen_matmul_data PROPERTIES FIXTURES_SETUP matmul_data)

ter_add_test(test_matmul_reference)
set_tests_properties(test_matmul_reference PROPERTIES
    FIXTURES_REQUIRED matmul_data
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

add_test(NAME gen_matmul_b_data
    COMMAND ${CMAKE_SOURCE_DIR}/.venv/bin/python
            ${CMAKE_SOURCE_DIR}/tools/matmul_b_reference.py
            --out-dir ${CMAKE_BINARY_DIR}/matmul_b_data
            --m 8 --n 8 --k 27
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(gen_matmul_b_data PROPERTIES FIXTURES_SETUP matmul_b_data)

ter_add_test(test_kernel_matmul_b_32)
target_compile_definitions(test_kernel_matmul_b_32 PRIVATE TER_KERNELS_DIR="${CMAKE_SOURCE_DIR}/src/kernels")
set_tests_properties(test_kernel_matmul_b_32 PROPERTIES
    FIXTURES_REQUIRED matmul_b_data
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

add_test(NAME gen_rsqrt_lut
    COMMAND ${CMAKE_SOURCE_DIR}/.venv/bin/python
            ${CMAKE_SOURCE_DIR}/tools/gen_rsqrt_lut.py
            --out-dir ${CMAKE_BINARY_DIR}/lut_data
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(gen_rsqrt_lut PROPERTIES FIXTURES_SETUP rsqrt_lut)

ter_add_test(test_kernel_rmsnorm)
target_compile_definitions(test_kernel_rmsnorm PRIVATE TER_KERNELS_DIR="${CMAKE_SOURCE_DIR}/src/kernels")
set_tests_properties(test_kernel_rmsnorm PROPERTIES
    FIXTURES_REQUIRED rsqrt_lut
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

ter_add_test(test_kernel_relocation)

add_test(NAME gen_softmax_luts
    COMMAND ${CMAKE_SOURCE_DIR}/.venv/bin/python
            ${CMAKE_SOURCE_DIR}/tools/gen_softmax_luts.py
            --out-dir ${CMAKE_BINARY_DIR}/lut_data
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(gen_softmax_luts PROPERTIES FIXTURES_SETUP softmax_luts)

ter_add_test(test_kernel_softmax)
target_compile_definitions(test_kernel_softmax PRIVATE TER_KERNELS_DIR="${CMAKE_SOURCE_DIR}/src/kernels")
set_tests_properties(test_kernel_softmax PROPERTIES
    FIXTURES_REQUIRED softmax_luts
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

add_test(NAME gen_sigmoid_lut
    COMMAND ${CMAKE_SOURCE_DIR}/.venv/bin/python
            ${CMAKE_SOURCE_DIR}/tools/gen_sigmoid_lut.py
            --out-dir ${CMAKE_BINARY_DIR}/lut_data
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(gen_sigmoid_lut PROPERTIES FIXTURES_SETUP sigmoid_lut)

ter_add_test(test_kernel_silu)
target_compile_definitions(test_kernel_silu PRIVATE TER_KERNELS_DIR="${CMAKE_SOURCE_DIR}/src/kernels")
set_tests_properties(test_kernel_silu PROPERTIES
    FIXTURES_REQUIRED sigmoid_lut
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

ter_add_test(test_kernel_rope)
target_compile_definitions(test_kernel_rope PRIVATE TER_KERNELS_DIR="${CMAKE_SOURCE_DIR}/src/kernels")

ter_add_test(test_attention)
target_compile_definitions(test_attention PRIVATE TER_KERNELS_DIR="${CMAKE_SOURCE_DIR}/src/kernels")
set_tests_properties(test_attention PROPERTIES
    FIXTURES_REQUIRED "softmax_luts"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
