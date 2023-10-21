# Install script for directory: /home/local_user/sandbox/Velocity-Bench/cudaSift/CUDA

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/." TYPE FILE FILES
    "/home/local_user/sandbox/Velocity-Bench/cudaSift/CUDA/cudaImage.cu"
    "/home/local_user/sandbox/Velocity-Bench/cudaSift/CUDA/cudaImage.h"
    "/home/local_user/sandbox/Velocity-Bench/cudaSift/CUDA/cudaSiftH.cu"
    "/home/local_user/sandbox/Velocity-Bench/cudaSift/CUDA/cudaSiftH.h"
    "/home/local_user/sandbox/Velocity-Bench/cudaSift/CUDA/matching.cu"
    "/home/local_user/sandbox/Velocity-Bench/cudaSift/CUDA/cudaSiftD.h"
    "/home/local_user/sandbox/Velocity-Bench/cudaSift/CUDA/cudaSift.h"
    "/home/local_user/sandbox/Velocity-Bench/cudaSift/CUDA/cudautils.h"
    "/home/local_user/sandbox/Velocity-Bench/cudaSift/CUDA/../common/Utility.cpp"
    "/home/local_user/sandbox/Velocity-Bench/cudaSift/CUDA/geomFuncs.cpp"
    "/home/local_user/sandbox/Velocity-Bench/cudaSift/CUDA/mainSift.cpp"
    "/home/local_user/sandbox/Velocity-Bench/cudaSift/CUDA/cudaSiftD.cu"
    "/home/local_user/sandbox/Velocity-Bench/cudaSift/CUDA/CMakeLists.txt"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/data" TYPE FILE FILES
    "/home/local_user/sandbox/Velocity-Bench/cudaSift/CUDA/data/left.pgm"
    "/home/local_user/sandbox/Velocity-Bench/cudaSift/CUDA/data/righ.pgm"
    )
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/local_user/sandbox/Velocity-Bench/cudaSift/CUDA/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
