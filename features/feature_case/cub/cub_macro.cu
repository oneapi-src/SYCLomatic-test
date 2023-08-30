// ====------ cub_macro.cu --------------------------------- *- CUDA -* ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <cub/cub.cuh>

CUB_RUNTIME_FUNCTION void foo() {}

__global__ void test() {
  foo();
  (void) CUB_MIN(1, 2);
  (void) CUB_MAX(1, 2);
}

int main() {
  test<<<1, 1>>>();

  return 0;
}
