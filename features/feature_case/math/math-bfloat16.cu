// ====------ math-bfloat16.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <cuda.h>
#include <cuda_bf16.h>
#include <iostream>

__global__ void testMathFunctions(char *const TestResults) {
  const auto bf16 = __float2bfloat16(3.14f);
  const float f32 = __bfloat162float(bf16);
  TestResults[0] = (fabs(3.14f - f32) < 0.1f);
}

int main() {
  constexpr int NumberOfTests = 1;
  char *TestResults;
  cudaMallocManaged(&TestResults, NumberOfTests * sizeof(*TestResults));
  testMathFunctions<<<1, 1>>>(TestResults);
  cudaDeviceSynchronize();
  for (int i = 0; i < NumberOfTests; i++) {
    if (TestResults[i] == 0) {
      std::cout << "Test " << i << " failed" << std::endl;
      return 1;
    }
  }
  return 0;
}
