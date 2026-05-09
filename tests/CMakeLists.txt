function(ter_add_test name)
    add_executable(${name} ${name}.cpp)
    target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_link_libraries(${name} PRIVATE ter doctest::doctest)
    add_test(NAME ${name} COMMAND ${name} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endfunction()

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
