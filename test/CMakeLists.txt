cmake_minimum_required(VERSION 3.7)
project(test LANGUAGES CXX)
message(STATUS "CMake version: ${CMAKE_VERSION}, Project: ${PROJECT_NAME}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE}/tests) # Директрия для тестов

enable_testing(true)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package (Boost COMPONENTS unit_test_framework REQUIRED)
#find_package(BLAS REQUIRED)
find_package(Armadillo REQUIRED)
include_directories(/usr/include/python3.8)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
#-----------------------------------------------------------------------------------------------------------------------
add_executable(test_lap
    test_lap.cpp    
    ../src/3d_party/timing.h
    ../src/3d_party/timing.cpp
    )
add_test(NAME test_lap COMMAND test_lap)

target_link_libraries(test_lap
    lap
    libpython3.8.so
    ${Boost_LIBRARIES}
    )

target_include_directories(test_lap
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/3d_party
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/lap
    )
#-----------------------------------------------------------------------------------------------------------------------
add_executable(test_sparse
    test_sparse.cpp
    ../src/3d_party/timing.h
    ../src/3d_party/timing.cpp
    )
add_test(NAME test_sparse COMMAND test_sparse)

target_link_libraries(test_sparse
    lap
    libpython3.8.so
    ${Boost_LIBRARIES}
    )

target_include_directories(test_sparse
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/3d_party
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/lap
    )
