cmake_minimum_required(VERSION 3.10)
project(foo)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
find_program(dpct_bin_path NAMES dpct PATHS)
get_filename_component(bin_path_of_dpct ${dpct_bin_path} DIRECTORY)
set(dpct_cmake_file_path "${bin_path_of_dpct}/../cmake/dpct.cmake")
include(${dpct_cmake_file_path})

set(SOURCES
    ${CMAKE_SOURCE_DIR}/main.dp.cpp
)
add_executable(foo-bar ${SOURCES})
target_link_libraries(foo-bar PUBLIC -qmkl ${DNN_LIB})
