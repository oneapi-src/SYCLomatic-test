// ====------ LibCommonUtils_api_test2.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// TEST_FEATURE: LibCommonUtils_mkl_get_version

#include "cufft.h"

int main() {
  int ver;
  cufftGetVersion(&ver);
  return 0;
}

