// ====------ test.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include "header1/header1.h"
#include "header2/header2.h"
int main() {
  helloFromGPU<<<1, 1>>>();
  helloFromGPU2<<<1, 1>>>();
  return 0;
}