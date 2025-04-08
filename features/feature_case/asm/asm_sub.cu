// ====------ asm_sub.cu ------------------------- *- CUDA -* -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>

#define TEST(FN)                                                               \
  {                                                                            \
    if (FN()) {                                                                \
      printf("Test " #FN " PASS\n");                                           \
    } else {                                                                   \
      printf("Test " #FN " FAIL\n");                                           \
      return 1;                                                                \
    }                                                                          \
  }

inline __device__ uint32_t sub(uint32_t a, uint32_t b) {
  uint32_t c;
  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
  return c;
}

__global__ void test_sub_kernel(uint32_t *d_out, uint32_t a, uint32_t b) {
  *d_out = sub(a, b);
}

// Function to convert a float to a 16-bit half-float (IEEE 754)
uint16_t floatToHalf(float value) {
  uint32_t f = *(uint32_t *)&value;
  uint32_t sign = (f & 0x80000000) >> 16;
  uint32_t exponent =
      ((f & 0x7F800000) >> 23) - 112;         // Adjust exponent bias (127-15)
  uint32_t mantissa = (f & 0x007FFFFF) >> 13; // Reduce mantissa to 10 bits

  if (exponent <= 0) { // Handle underflow
    exponent = 0;
    mantissa = 0;
  } else if (exponent >= 31) { // Handle overflow
    exponent = 31;
    mantissa = 0;
  }

  return (uint16_t)(sign | (exponent << 10) | mantissa);
}

// Function to convert a 16-bit half-float back to float
float halfToFloat(uint16_t half) {
  uint32_t sign = (half & 0x8000) << 16;
  uint32_t exponent =
      ((half & 0x7C00) >> 10) + 112; // Adjust exponent bias (15+127-127)
  uint32_t mantissa = (half & 0x03FF) << 13; // Extend mantissa to 23 bits

  uint32_t f = sign | (exponent << 23) | mantissa;
  return *(float *)&f;
}

// Function to pack two half-floats into a 32-bit integer
uint32_t packHalfFloats(float a, float b) {
  return (uint32_t(floatToHalf(a)) << 16) | floatToHalf(b);
}

bool run_test() {
  uint32_t h_out;
  uint32_t *d_out;
  uint32_t a = packHalfFloats(
      1.5, 1.5); // Example packed half-precision floats (1.5, 1.5)
  uint32_t b = packHalfFloats(
      1.0, 1.0); // Example packed half-precision floats (1.0, 1.0)
  uint32_t expected = packHalfFloats(0.5, 0.5); // Expected result (0.5, 0.5)

  cudaMalloc(&d_out, sizeof(uint32_t));
  test_sub_kernel<<<1, 1>>>(d_out, a, b);
  cudaMemcpy(&h_out, d_out, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaFree(d_out);

  if (h_out == expected) {
    return true;
  } else {
    std::cout << "Computed: 0x" << std::hex << h_out << " Expected: 0x"
              << expected << std::endl;
    return false;
  }
}

int main() {
  TEST(run_test);
  return 0;
}
