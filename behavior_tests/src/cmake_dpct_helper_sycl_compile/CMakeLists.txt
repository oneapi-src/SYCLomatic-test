# ==---- CMakeLists.txt ---------------------------- cmake script file ----==//
#
# Copyright (C) Intel Corporation SPDX-License-Identifier: Apache-2.0 WITH
# LLVM-exception See https://llvm.org/LICENSE.txt for license information.
#
# ===----------------------------------------------------------------------===//

cmake_minimum_required(VERSION 3.25)

project(cuda_compile_sample)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")

find_package(IntelSYCL REQUIRED)
find_program(
  DPCT_BIN_PATH
  NAMES dpct
  PATHS)

get_filename_component(DPCT_BIN_DIR ${DPCT_BIN_PATH} DIRECTORY)
set(DPCT_CMAKE_PATH "${DPCT_BIN_DIR}/../cmake/dpct.cmake")
include(${DPCT_CMAKE_PATH})

dpct_helper_sycl_compile(objSetOne a.dp.cpp b.dp.cpp)

set(_src_files c.dp.cpp d.dp.cpp)
dpct_helper_sycl_compile(objSetTwo ${_src_files})

set(allObjs)
list(APPEND allObjs ${objSetOne})
list(APPEND allObjs ${objSetTwo})

if(WIN32)
  set(SHARED_LIB_EXTN ".lib")
elseif(UNIX AND NOT APPLE)
  set(SHARED_LIB_EXTN ".so")
else()
  message(FATAL "Unsupported platform")
endif()

set(_cnt 1)
set(_allObjsLibs "")
foreach(objPath ${allObjs})
  get_filename_component(libName ${objPath} NAME)
  string(REGEX REPLACE "${CMAKE_CXX_OUTPUT_EXTENSION}$" ${SHARED_LIB_EXTN} libPath ${objPath})
  message(STATUS "LibPath: ${libPath}")

if (UNIX AND NOT APPLE)
  add_custom_command(
    OUTPUT  ${libPath}
    COMMAND ${CMAKE_COMMAND} -E copy ${objPath} ${libPath}
    DEPENDS ${objPath}
    COMMENT "Generating ${libPath}")
elseif(WIN32)
   # This is a dummy command to make cmake do something for the target $libPath
   add_custom_command(
    OUTPUT  ${libPath}
    COMMAND ${CMAKE_COMMAND} -E echo "Generating ${libPath}"
    DEPENDS ${objPath})
endif()

  list(APPEND _allObjsLibs ${libPath})
  
  add_library(Lib${_cnt} SHARED IMPORTED)

if (WIN32)
  set_target_properties(Lib${_cnt} PROPERTIES IMPORTED_IMPLIB ${libPath})
elseif (UNIX AND NOT APPLE)
  set_target_properties(Lib${_cnt} PROPERTIES IMPORTED_LOCATION ${libPath})
endif()
  
  MATH(EXPR _cnt "${_cnt}+1")
endforeach()

add_executable(run main.dp.cpp)
target_link_libraries(run PRIVATE Lib1 Lib2 Lib3 Lib4)

add_custom_target(tdc ALL DEPENDS ${_allObjsLibs})
add_dependencies(run tdc)
