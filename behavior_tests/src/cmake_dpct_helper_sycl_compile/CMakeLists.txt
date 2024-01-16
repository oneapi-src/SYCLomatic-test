# ==---- CMakeLists.txt ---------------------------- cmake script file ----==//
#
# Copyright (C) Intel Corporation SPDX-License-Identifier: Apache-2.0 WITH
# LLVM-exception See https://llvm.org/LICENSE.txt for license information.
#
# ===----------------------------------------------------------------------===//

cmake_minimum_required(VERSION 3.20)

project(cuda_compile_sample NONE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")

find_program(
  DPCT_BIN_PATH
  NAMES dpct
  PATHS)


get_filename_component(DPCT_BIN_DIR ${DPCT_BIN_PATH} DIRECTORY)
set(DPCT_CMAKE_PATH "${DPCT_BIN_DIR}/../cmake/dpct.cmake")
message(STATUS "INCLUDING ${DPCT_CMAKE_PATH}")
include(${DPCT_CMAKE_PATH})

dpct_helper_sycl_compile(objSetOne a.dp.cpp b.dp.cpp)

set(_src_files c.dp.cpp d.dp.cpp)
dpct_helper_sycl_compile(objSetTwo ${_src_files})

set(allObjs)
list(APPEND allObjs ${objSetOne})
list(APPEND allObjs ${objSetTwo})

set(_cnt 1)
set(_allObjsLibs "")
foreach(obj ${allObjs})
  get_filename_component(libName ${obj} NAME)

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${libName}.so
    COMMAND ${CMAKE_COMMAND} -E copy ${obj}
            ${CMAKE_CURRENT_BINARY_DIR}/${libName}.so
    DEPENDS ${obj}
    COMMENT "Generating ${CMAKE_CURRENT_BINARY_DIR}/${libName}.so")
  
  list(APPEND _allObjsLibs ${CMAKE_CURRENT_BINARY_DIR}/${libName}.so)
  
  add_library(Lib${_cnt} SHARED IMPORTED)
  set_target_properties(Lib${_cnt} PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/${libName}.so)
  
  MATH(EXPR _cnt "${_cnt}+1")
endforeach()

add_executable(app.run main.dp.cpp)
target_link_libraries(app.run PRIVATE Lib1 Lib2 Lib3 Lib4)

add_custom_target(tdc ALL DEPENDS ${_allObjsLibs})
add_dependencies(app.run tdc)
install(TARGETS app.run)
