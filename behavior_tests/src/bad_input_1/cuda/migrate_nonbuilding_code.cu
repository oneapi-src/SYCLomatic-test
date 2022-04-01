// ====------ migrate_nonbuilding_code.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <cuda.h>
#include <cmath>

//typedef struct { int eps,sig; } unknown_t;

template <int EFLAG>
__global__ void foo(const unknown_t *ljd_in) {

  __shared__ unknown_t ljd[128];

}

int main() {
  unknown_t *data;
  foo<3><<<1, 1>>>(data);
}
