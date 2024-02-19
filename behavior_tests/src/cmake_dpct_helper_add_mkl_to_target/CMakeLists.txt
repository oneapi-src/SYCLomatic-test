#==---- CMakeLists.txt ---------------------------- cmake script file ----==//
#
# Copyright (C) Intel Corporation
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# See https://llvm.org/LICENSE.txt for license information.
#
#===----------------------------------------------------------------------===//

cmake_minimum_required(VERSION 3.10)

project(cmake_add_mkl LANGUAGES CXX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
find_program(
  dpct_bin_path
  NAMES dpct
  PATHS)
get_filename_component(bin_path_of_dpct ${dpct_bin_path} DIRECTORY)
set(dpct_cmake_file_path "${bin_path_of_dpct}/../cmake/dpct.cmake")
include(${dpct_cmake_file_path})

find_package(IntelSYCL REQUIRED)

set(CUDA_SOURCES
    fft.dp.cpp
)

add_executable(app.run ${CUDA_SOURCES})

link_directories($ENV{TBBROOT}/$ENV{TBBVER}/tbb/$ENV{TBB_MAJORVER}/lib)
link_directories(${TBBROOT}/lib)
dpct_helper_add_mkl_to_target(app.run)
