add_library(spm_timing STATIC timing.hpp timing.cpp)
target_include_directories(spm_timing PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../..)
