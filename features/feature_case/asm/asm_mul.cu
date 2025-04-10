// ====------ asm_mul.cu ------------------------- *- CUDA -* -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <cuda.h>
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

// Inline device function using PTX assembly for f16x2 multiplication
inline __device__ uint32_t mul(uint32_t a, uint32_t b) {
  uint32_t c;
  asm volatile("mul.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
  return c;
}

// CUDA kernel to test the mul function
__global__ void test_kernel(uint32_t *d_out, uint32_t a, uint32_t b) {
  *d_out = mul(a, b);
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

// Function to pack two half-floats into a 32-bit integer
uint32_t packHalfFloats(float a, float b) {
  return (uint32_t(floatToHalf(a)) << 16) | floatToHalf(b);
}

bool run_test() {
  // Test inputs (packed FP16 format)
  uint32_t a = packHalfFloats(1.5, 1.5); // (1.5, 1.5) in FP16 packed format
  uint32_t b = packHalfFloats(2.0, 2.0); // (2.0, 2.0) in FP16 packed format

  // Expected output: (1.5 * 2.0, 1.5 * 2.0) = (3.0, 3.0) in FP16 packed format
  uint32_t expected =
      packHalfFloats(3.0, 3.0); // (3.0, 3.0) in FP16 packed format

  // Allocate device memory
  uint32_t *d_out;
  cudaMalloc(&d_out, sizeof(uint32_t));

  // Launch kernel
  test_kernel<<<1, 1>>>(d_out, a, b);
  cudaDeviceSynchronize();

  // Copy result back to host
  uint32_t h_out;
  cudaMemcpy(&h_out, d_out, sizeof(uint32_t), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_out);

  // Validate result
  if (h_out == expected) {
    return true;
  } else {
    std::cerr << "Test Failed! Expected: 0x" << std::hex << expected
              << ", but got: 0x" << h_out << std::endl;
    return false;
  }
}

int main() {
  TEST(run_test);
  return 0;
}
