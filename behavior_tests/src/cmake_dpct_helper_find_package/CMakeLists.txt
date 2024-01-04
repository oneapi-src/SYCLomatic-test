#==---- CMakeLists.txt ---------------------------- cmake script file ----==//
#
# Copyright (C) Intel Corporation
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# See https://llvm.org/LICENSE.txt for license information.
#
#===----------------------------------------------------------------------===//

cmake_minimum_required(VERSION 3.10)

project(cmake_find_package LANGUAGES CXX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
find_program(
  dpct_bin_path
  NAMES dpct
  PATHS)
get_filename_component(bin_path_of_dpct ${dpct_bin_path} DIRECTORY)
set(dpct_cmake_file_path "${bin_path_of_dpct}/../cmake/dpct.cmake")
include(${dpct_cmake_file_path})

dpct_helper_find_package(oneAPIToolkit REQUIRED blas)
dpct_helper_find_package(oneDPL REQUIRED)

set(SOURCES
    ${CMAKE_SOURCE_DIR}/main.dp.cpp
)

add_executable(app.run ${SOURCES})
target_link_libraries(app.run PUBLIC MKL::MKL_SYCL)
