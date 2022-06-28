// ====------ math-exec.cu---------- *- CUDA -* ----===////
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
    __half2 h2 = __halves2half2(1.1, 1.1);
    h2 = __hadd2(h2, h2);
    TestResults[0] =
        (__low2half(h2) == __half(2.2) && __high2half(h2) == __half(2.2));
  }

  {
    __half h = __half(0.5);
    h = hrcp(h);
    TestResults[1] = (h == __half(2));
  }

  {
    long l = -5;
    l = labs(l);
    TestResults[2] = (l == long(5l));
  }

  {
    long long ll = -10;
    ll = llabs(ll);
    TestResults[3] = (ll == 10ll);
  }
}

int main() {
  constexpr int NumberOfTests = 4;
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
