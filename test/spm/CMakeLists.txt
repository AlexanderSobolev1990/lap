find_package(Boost REQUIRED)

foreach(target IN ITEMS test_spm_lap test_spm_lap_sparse test_spm_lap_murty)
    add_executable(${target} ${target}.cpp)
    target_include_directories(${target} PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(${target} PRIVATE spm_lap spm_timing)
    set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE}/tests)
    add_test(NAME ${target} COMMAND ${target})
    set_tests_properties(${target} PROPERTIES TIMEOUT 60 LABELS spm)
endforeach()
target_sources(test_spm_lap_murty PRIVATE test_spm_lap_assignment_shapes.cpp)

add_custom_target(spm_tests DEPENDS test_spm_lap test_spm_lap_sparse test_spm_lap_murty)
