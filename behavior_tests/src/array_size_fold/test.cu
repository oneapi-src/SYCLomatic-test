// ====------ test.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#define S 3
class C {};
__global__ void f() {
  const int x = 2;
  __shared__ int fold1[S];
  __shared__ int fold2[x];
  __shared__ int fold3[sizeof(C) * 3];
  __shared__ int fold4[sizeof(x) * 3];
  __shared__ int fold5[sizeof(x * 3) * 3];
  __shared__ int fold6[S][S+1+S];
  __shared__ int unfold1[1 + 1];
  __shared__ int unfold2[sizeof(float3) * 3];
}
int main() {
  f<<<1, 1>>>();
  return 0;
}
