// ====------ get_library_version.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <iostream>

#include "cublas.h"

void foo1() {
  int version = 0;
  cublasGetVersion(&version);
  std::cout << "foo1" << std::endl;
  std::cout << version << std::endl;
}

#include "cublas_v2.h"

void foo2() {
  cublasHandle_t handle;
  int version = 0;
  cublasGetVersion(handle, &version);
  std::cout << "foo2" << std::endl;
  std::cout << version << std::endl;
}

#include "cufft.h"

void foo3() {
  int version = 0;
  cufftGetVersion(&version);
  libraryPropertyType major_t = MAJOR_VERSION;
  libraryPropertyType minor_t = MINOR_VERSION;
  libraryPropertyType patch_t = PATCH_LEVEL;
  int major = 0, minor = 0, patch = 0;
  cufftGetProperty(major_t, &major);
  cufftGetProperty(minor_t, &minor);
  cufftGetProperty(patch_t, &patch);
  std::cout << "foo3" << std::endl;
  std::cout << version << std::endl;
  std::cout << major << std::endl;
  std::cout << minor << std::endl;
  std::cout << patch << std::endl;
}

int main() {
  foo1();
  foo2();
  foo3();
  return 0;
}
