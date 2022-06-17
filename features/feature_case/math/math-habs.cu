// ====------ math-habs.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <cassert>
#include <iostream>

#include <cuda.h>
#include <cuda_fp16.h>

__global__ void testMathFunctions(char *const TestResults) {
  {
    __half h = -3.14;
    h = __habs(h);
    TestResults[0] = (h == half(3.14));
  }

  {
    __half2 h2 = __halves2half2(-1.1, -1.1);
    h2 = __habs2(h2);
    TestResults[1] =
        (__low2half(h2) == half(1.1) && __high2half(h2) == half(1.1));
  }
}

int main() {
  constexpr int NumberOfTests = 2;
  char *TestResults;
  cudaMallocManaged(&TestResults, NumberOfTests * sizeof(*TestResults));
  testMathFunctions<<<1, 1>>>(TestResults);
  cudaDeviceSynchronize();
  for (int i = 0; i < NumberOfTests; i++) {
    if (TestResults[i] == 0) {
      std::cout << "Test " << i << " failed" << std::endl;
    }
    assert(TestResults[i] != 0);
  }
}
