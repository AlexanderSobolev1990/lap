find_package(Armadillo REQUIRED)

add_library(spm_lap STATIC
    lap.hpp
    searchparam.hpp
    sparse.hpp sparse.cpp
    jvcdense.hpp jvcdense.cpp
    jvcsparse.hpp jvcsparse.cpp
    murty.hpp murty.cpp
    hungarian.hpp hungarian.cpp
    mack.hpp mack.cpp
    seqextr.hpp seqextr.cpp
)
set_target_properties(spm_lap PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(spm_lap PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../.. ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(spm_lap PUBLIC spm_common ${ARMADILLO_LIBRARIES})
target_compile_options(spm_lap PRIVATE -Wall -Wextra)
