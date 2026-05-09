function(ter_add_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE ter doctest::doctest)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

ter_add_test(test_smoke)
