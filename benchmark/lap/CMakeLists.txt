find_package(Boost REQUIRED)

foreach(target IN ITEMS benchmark_lap benchmark_sparse)
    add_executable(${target} ${target}.cpp)
    target_include_directories(${target} PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(${target} PRIVATE spm_lap spm_timing)
endforeach()

if(SPM_ENABLE_PLOTS)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
    target_compile_definitions(benchmark_lap PRIVATE MATPLOTLIB)
    target_include_directories(benchmark_lap PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/3d_party ${Python3_NumPy_INCLUDE_DIRS})
    target_link_libraries(benchmark_lap PRIVATE Python3::Python)
    add_custom_target(lap_plots
        COMMAND ${CMAKE_COMMAND} -E env MPLBACKEND=Agg $<TARGET_FILE:benchmark_lap>
            --run_test=denseSmall,denseSmallSeqExtr,sparseSmall,sparseSmallSeqExtr,denseLarge,sparseLarge,sparseLargeSeqExtr,sparseLargeExtra
        DEPENDS benchmark_lap
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/doc/pictures
        COMMENT "Generating LAP benchmark plots for README"
    )
endif()
