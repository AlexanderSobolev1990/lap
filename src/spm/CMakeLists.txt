add_subdirectory(common)
add_subdirectory(lap)
add_subdirectory(timing)

add_library(spm INTERFACE)
target_link_libraries(spm INTERFACE spm_common spm_lap spm_timing)
