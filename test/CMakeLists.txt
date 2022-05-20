cmake_minimum_required(VERSION 3.7)
project(test_lap LANGUAGES CXX)
message(STATUS "CMake version: ${CMAKE_VERSION}, Project: ${PROJECT_NAME}")

enable_testing(true)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package (Boost COMPONENTS unit_test_framework REQUIRED)
find_package(BLAS REQUIRED)
find_package(Armadillo REQUIRED)
include_directories(/usr/include/python3.8)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_executable(${PROJECT_NAME}
    test_lap.cpp
    ../src/3d_party/timing.h
    ../src/3d_party/timing.cpp
    )
add_test(NAME test_lap COMMAND test_lap)

target_link_libraries(${PROJECT_NAME}
    lap
    libpython3.8.so
    ${Boost_LIBRARIES}
    )

target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/3d_party
    )
