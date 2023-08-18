// ===-------------- math-half-raw.cu -------=--------- *- CUDA -* --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <cuda_fp16.h>
#include <cuda_runtime.h>
int main() {
  __half_raw one_h{0x3C00};
  __half_raw zero_h{0};
  __half_raw *ptr = new __half_raw{0};
  ptr->x = 0x3C00;
  zero_h.x = 0x3C00;
  half alpha = one_h;
  if(float(alpha) != 1.0) return 1;
  alpha = zero_h;
  if(float(alpha) != 1.0) return 1;
  return 0;
}