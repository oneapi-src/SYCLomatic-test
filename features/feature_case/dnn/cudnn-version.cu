// ====------ cudnn-version.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include "cudnn.h"
#include <iostream>

int main() {

  size_t version = cudnnGetVersion();
  std::cout << "version = " << version << std::endl;
  if((version >= 3000) && (version < 10000)) {
    std::cout << "passed" << std::endl;
  } else {
    std::cout << "failed" << std::endl;
    return 1;
  }
  return 0;
}