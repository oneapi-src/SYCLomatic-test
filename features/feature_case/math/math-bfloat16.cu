// ====-------------- math-bfloat16.cu---------- *- CUDA -* -------------===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <cuda.h>
#include <cuda_bf16.h>
#include <iostream>

__device__ uint16_t convertToU16(__nv_bfloat16 value) {
  union {
    __nv_bfloat16 bf16;
    uint16_t u16;
  } TypePun;
  TypePun.bf16 = value;
  return TypePun.u16;
}

__device__ bool valuesAreClose(float a, float b, float epsilon = 0.05f) {
  return (fabsf(a - b) < epsilon);
}

__global__ void testMathFunctions(char *const TestResults) {
  const __nv_bfloat16 bf16 = __float2bfloat16(3.14f);
  const float f32 = __bfloat162float(bf16);

  // Check that the intermediate bfloat16 value has the expected byte
  // representation. The CUDA and SYCL values both use round-to-nearest-even
  // rounding mode.
  TestResults[0] = (convertToU16(bf16) == 0x4049);

  // Check that the converted value is close to the original. The two values
  // may differ slightly due to the loss of precision during type conversion.
  TestResults[1] = valuesAreClose(f32, 3.14f);
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
      return 1;
    }
  }
  return 0;
}
