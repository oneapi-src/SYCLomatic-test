// ====------ math-saturate.cu---------------------------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include <cassert>
#include <iostream>
#include <limits>

#include <cuda.h>

__global__ void test_saturate(char *const TestResults) {
  TestResults[0] = (__saturatef(-0.1f) == 0.0f);
  TestResults[1] = (__saturatef(1.1f) == 1.0f);
  TestResults[2] = (__saturatef(0.5f) == 0.5f);
  TestResults[3] =
      (__saturatef(std::numeric_limits<float>::quiet_NaN()) == 0.0f);

  TestResults[4] = (saturate(-0.1f) == 0.0f);
  TestResults[5] = (saturate(1.1f) == 1.0f);
  TestResults[6] = (saturate(0.5f) == 0.5f);
  TestResults[7] =
      (saturate(std::numeric_limits<float>::quiet_NaN()) == 0.0f);
}

int main() {
  constexpr int NumberOfTests = 8;
  char *TestResults;
  cudaMallocManaged(&TestResults, NumberOfTests * sizeof(*TestResults));
  test_saturate<<<1, 1>>>(TestResults);
  cudaDeviceSynchronize();
  for (int i = 0; i < NumberOfTests; i++) {
    if (TestResults[i] == 0) {
      std::cerr << "saturate test " << i << " failed" << std::endl;
      return 1;
    }
  }
  return 0;
}
