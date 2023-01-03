// ====------ test.cu -------------------------------------- *- CUDA -* ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

__global__ void k(float f, float* f_p) {
  atomicAdd(f_p, f);
}

int main() {
  return 0;
}
