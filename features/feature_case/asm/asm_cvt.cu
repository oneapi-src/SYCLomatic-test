// ====------ asm_cvt.cu ----------------------------------- *- CUDA -* ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <cmath>
#include <cuda_fp16.h>
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

inline __device__ uint32_t float2_to_half2(float2 f) {
  union {
    uint32_t u32;
    uint16_t u16[2];
  } tmp;
  asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n"
               : "=r"(tmp.u32)
               : "f"(f.y), "f"(f.x));

  return tmp.u32;
}

// Kernel function to test float2_to_half2
__global__ void testFloat2ToHalf2(float2 *inputs, uint32_t *outputs,
                                  int numElements) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < numElements) {
    outputs[idx] = float2_to_half2(inputs[idx]);
  }
}

// CPU function to convert float to half-precision manually
uint16_t floatToHalf(float f) {
  __half h = __float2half(f); // CUDA built-in function
  return *(reinterpret_cast<uint16_t *>(&h));
}

// CPU function to compute expected half2 representation
uint32_t computeExpectedHalf2(float2 f) {
  uint16_t h1 = floatToHalf(f.x);
  uint16_t h2 = floatToHalf(f.y);
  return (static_cast<uint32_t>(h2) << 16) | h1;
}

bool cvt_rn_f16x2_f32_test() {
  // Define test inputs
  float2 hostInputs[] = {
      {1.0f, 2.0f},      // Normal case
      {0.0f, 0.0f},      // Zero case
      {-1.0f, -2.0f},    // Negative values
      {65504.0f, 1e-8f}, // Max half, small value
      {INFINITY, NAN},   // Special cases
  };

  const int numTests = sizeof(hostInputs) / sizeof(hostInputs[0]);
  uint32_t hostOutputs[numTests];
  uint32_t expectedOutputs[numTests];

  // Compute expected outputs
  for (int i = 0; i < numTests; i++) {
    expectedOutputs[i] = computeExpectedHalf2(hostInputs[i]);
  }

  // Allocate GPU memory
  float2 *d_inputs;
  uint32_t *d_outputs;
  cudaMalloc(&d_inputs, numTests * sizeof(float2));
  cudaMalloc(&d_outputs, numTests * sizeof(uint32_t));

  // Copy inputs to device
  cudaMemcpy(d_inputs, hostInputs, numTests * sizeof(float2),
             cudaMemcpyHostToDevice);

  // Launch kernel
  testFloat2ToHalf2<<<1, numTests>>>(d_inputs, d_outputs, numTests);

  // Copy results back to host
  cudaMemcpy(hostOutputs, d_outputs, numTests * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);

  // Validate results
  bool testPassed = true;
  for (int i = 0; i < numTests; i++) {
    if (hostOutputs[i] != expectedOutputs[i]) {
      std::cerr << "Test failed for input (" << hostInputs[i].x << ", "
                << hostInputs[i].y << ")\n"
                << "Expected: " << std::hex << expectedOutputs[i]
                << ", Got: " << hostOutputs[i] << std::dec << "\n";
      testPassed = false;
    }
  }

  // Cleanup
  cudaFree(d_inputs);
  cudaFree(d_outputs);

  return testPassed;
}

int main() {
  TEST(cvt_rn_f16x2_f32_test);
  return 0;
}
