// ====------ warp.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include "cuda.h"

__global__ void foo() {
  unsigned mask;
  int predicate;
  int val = 0;
  int srcLane;

  __all_sync(mask, predicate);

  __any_sync(mask, predicate);

  __shfl_sync(mask, val, srcLane);
}

int main() {
  return 0;
}