# tersim unit tests — mirror parent ter/tests/ pattern (doctest 2.4.11 from parent FetchContent).

add_executable(test_trace_capture test_trace_capture.cpp)
target_link_libraries(test_trace_capture PRIVATE tersim ter doctest)
target_compile_options(test_trace_capture PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_test(NAME test_trace_capture COMMAND test_trace_capture)

add_executable(test_pipeline_tvmac test_pipeline_tvmac.cpp)
target_link_libraries(test_pipeline_tvmac PRIVATE tersim ter doctest)
target_compile_options(test_pipeline_tvmac PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_test(NAME test_pipeline_tvmac COMMAND test_pipeline_tvmac)

add_executable(test_pipeline_hazards test_pipeline_hazards.cpp)
target_link_libraries(test_pipeline_hazards PRIVATE tersim ter doctest)
target_compile_options(test_pipeline_hazards PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_test(NAME test_pipeline_hazards COMMAND test_pipeline_hazards)

add_executable(test_cache_hit_rates test_cache_hit_rates.cpp)
target_link_libraries(test_cache_hit_rates PRIVATE tersim ter doctest)
target_compile_options(test_cache_hit_rates PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_test(NAME test_cache_hit_rates COMMAND test_cache_hit_rates)

add_executable(test_mshr_and_bp test_mshr_and_bp.cpp)
target_link_libraries(test_mshr_and_bp PRIVATE tersim ter doctest)
target_compile_options(test_mshr_and_bp PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_test(NAME test_mshr_and_bp COMMAND test_mshr_and_bp)

# G5b: end-to-end real-kernel trace. Needs path to parent's compiled .tasm
# kernels (loaded at runtime by install_default_kernels).
get_filename_component(_TER_KERNELS_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/kernels" ABSOLUTE)
add_executable(test_forward_kernel_e2e test_forward_kernel_e2e.cpp)
target_link_libraries(test_forward_kernel_e2e PRIVATE tersim ter doctest)
target_compile_options(test_forward_kernel_e2e PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_compile_definitions(test_forward_kernel_e2e PRIVATE
    TER_KERNELS_DIR="${_TER_KERNELS_DIR}")
add_test(NAME test_forward_kernel_e2e COMMAND test_forward_kernel_e2e)

add_executable(test_gpu_roofline test_gpu_roofline.cpp)
target_link_libraries(test_gpu_roofline PRIVATE tersim doctest)
target_compile_options(test_gpu_roofline PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_test(NAME test_gpu_roofline COMMAND test_gpu_roofline)

# G7: Layer-3 ASIC energy/area/freq projection. Needs kernels dir for the
# end-to-end OpCounters → projection sanity (same as test_forward_kernel_e2e).
add_executable(test_asic_energy test_asic_energy.cpp)
target_link_libraries(test_asic_energy PRIVATE tersim ter doctest)
target_compile_options(test_asic_energy PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_compile_definitions(test_asic_energy PRIVATE
    TER_KERNELS_DIR="${_TER_KERNELS_DIR}")
add_test(NAME test_asic_energy COMMAND test_asic_energy)
