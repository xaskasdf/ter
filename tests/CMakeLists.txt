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

ter_add_test(test_nt_sampler)
target_link_libraries(test_nt_sampler PRIVATE nt_infra)

ter_add_test(test_nt_dequant)
target_link_libraries(test_nt_dequant PRIVATE nt_infra)

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
ter_add_test(test_tfloat)
ter_add_test(test_bitnet)

ter_add_test(test_ter_k4)
target_link_libraries(test_ter_k4 PRIVATE ter_k4)
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

ter_add_test(test_layer_types)

ter_add_test(test_layer_forward)
target_compile_definitions(test_layer_forward PRIVATE TER_KERNELS_DIR="${CMAKE_SOURCE_DIR}/src/kernels")
set_tests_properties(test_layer_forward PROPERTIES
    FIXTURES_REQUIRED "rsqrt_lut;sigmoid_lut;softmax_luts"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

ter_add_test(test_layer_multitoken)
target_compile_definitions(test_layer_multitoken PRIVATE TER_KERNELS_DIR="${CMAKE_SOURCE_DIR}/src/kernels")
set_tests_properties(test_layer_multitoken PROPERTIES
    FIXTURES_REQUIRED "rsqrt_lut;sigmoid_lut;softmax_luts"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

ter_add_test(test_load_brandon_tiny)
target_link_libraries(test_load_brandon_tiny PRIVATE nt_infra)

ter_add_test(test_brandon_config)
target_link_libraries(test_brandon_config PRIVATE nt_infra)

ter_add_test(test_brandon_tokenizer)
target_link_libraries(test_brandon_tokenizer PRIVATE nt_infra)

ter_add_test(test_layer_brandon_shape)
target_compile_definitions(test_layer_brandon_shape PRIVATE TER_KERNELS_DIR="${CMAKE_SOURCE_DIR}/src/kernels")
set_tests_properties(test_layer_brandon_shape PROPERTIES
    FIXTURES_REQUIRED "rsqrt_lut;sigmoid_lut;softmax_luts"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

ter_add_test(test_brandon_forward)
target_link_libraries(test_brandon_forward PRIVATE nt_infra)
target_compile_definitions(test_brandon_forward PRIVATE TER_KERNELS_DIR="${CMAKE_SOURCE_DIR}/src/kernels")
set_tests_properties(test_brandon_forward PROPERTIES
    FIXTURES_REQUIRED "rsqrt_lut;sigmoid_lut;softmax_luts"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    TIMEOUT 600)

ter_add_test(test_brandon_generate)
target_link_libraries(test_brandon_generate PRIVATE nt_infra)
target_compile_definitions(test_brandon_generate PRIVATE TER_KERNELS_DIR="${CMAKE_SOURCE_DIR}/src/kernels")
set_tests_properties(test_brandon_generate PROPERTIES
    FIXTURES_REQUIRED "rsqrt_lut;sigmoid_lut;softmax_luts"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    TIMEOUT 600)

ter_add_test(test_brandon_calibrate)
target_link_libraries(test_brandon_calibrate PRIVATE nt_infra)
target_compile_definitions(test_brandon_calibrate PRIVATE TER_KERNELS_DIR="${CMAKE_SOURCE_DIR}/src/kernels")
set_tests_properties(test_brandon_calibrate PROPERTIES
    FIXTURES_REQUIRED "rsqrt_lut;sigmoid_lut;softmax_luts"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    TIMEOUT 600)

ter_add_test(test_llama_smoke)
target_link_libraries(test_llama_smoke PRIVATE nt_infra)
target_compile_definitions(test_llama_smoke PRIVATE TER_KERNELS_DIR="${CMAKE_SOURCE_DIR}/src/kernels")
set_tests_properties(test_llama_smoke PROPERTIES
    FIXTURES_REQUIRED "rsqrt_lut;sigmoid_lut;softmax_luts"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    TIMEOUT 1800)

ter_add_test(test_llama_generate)
target_link_libraries(test_llama_generate PRIVATE nt_infra)
target_compile_definitions(test_llama_generate PRIVATE TER_KERNELS_DIR="${CMAKE_SOURCE_DIR}/src/kernels")
set_tests_properties(test_llama_generate PROPERTIES
    FIXTURES_REQUIRED "rsqrt_lut;sigmoid_lut;softmax_luts"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    TIMEOUT 3600)

ter_add_test(test_tinystories_smoke)
target_link_libraries(test_tinystories_smoke PRIVATE nt_infra)
target_compile_definitions(test_tinystories_smoke PRIVATE TER_KERNELS_DIR="${CMAKE_SOURCE_DIR}/src/kernels")
set_tests_properties(test_tinystories_smoke PROPERTIES
    FIXTURES_REQUIRED "rsqrt_lut;sigmoid_lut;softmax_luts"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    TIMEOUT 600)

ter_add_test(test_bitnet_post_quant)
target_link_libraries(test_bitnet_post_quant PRIVATE nt_infra)
set_tests_properties(test_bitnet_post_quant PROPERTIES TIMEOUT 60)

ter_add_test(test_bitnet_gguf_load)
target_link_libraries(test_bitnet_gguf_load PRIVATE nt_infra)
set_tests_properties(test_bitnet_gguf_load PROPERTIES TIMEOUT 120)

ter_add_test(test_bitnet_forward)
target_link_libraries(test_bitnet_forward PRIVATE nt_infra)
target_compile_definitions(test_bitnet_forward PRIVATE TER_KERNELS_DIR="${CMAKE_SOURCE_DIR}/src/kernels")
set_tests_properties(test_bitnet_forward PROPERTIES
    FIXTURES_REQUIRED "rsqrt_lut;sigmoid_lut;softmax_luts"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    TIMEOUT 3600)

ter_add_test(test_format_a_vs_b)
target_link_libraries(test_format_a_vs_b PRIVATE nt_infra)
target_compile_definitions(test_format_a_vs_b PRIVATE TER_KERNELS_DIR="${CMAKE_SOURCE_DIR}/src/kernels")
set_tests_properties(test_format_a_vs_b PROPERTIES
    FIXTURES_REQUIRED "rsqrt_lut;sigmoid_lut;softmax_luts"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    TIMEOUT 600)
