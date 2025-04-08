// ====------ asm_st.cu ----------------------------------- *- CUDA -* ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <cuda_runtime.h>
#include <iostream>

#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
#define __PTR "l"
#else
#define __PTR "r"
#endif

#define TEST(FN)                                                               \
  {                                                                            \
    if (FN()) {                                                                \
      printf("Test " #FN " PASS\n");                                           \
    } else {                                                                   \
      printf("Test " #FN " FAIL\n");                                           \
      return 1;                                                                \
    }                                                                          \
  }

__device__ inline void store_streaming_short4(short4 *addr, short x, short y,
                                              short z, short w) {
  asm("st.cs.global.v4.s16 [%0+0], {%1, %2, %3, %4};" ::__PTR(addr), "h"(x),
      "h"(y), "h"(z), "h"(w));
}

__global__ void test_store_streaming_short4(short4 *d_output) {
  store_streaming_short4(d_output, 1, 2, 3, 4);
}

__device__ inline void store_streaming_short2(short2 *addr, short x, short y) {
  asm("st.cs.global.v2.s16 [%0+0], {%1, %2};" ::__PTR(addr), "h"(x), "h"(y));
}

__global__ void test_store_streaming_short2(short2 *d_output) {
  store_streaming_short2(d_output, 1, 2);
}

bool test_1() {
  short4 *d_output;
  short4 h_output;

  // Allocate memory on GPU
  cudaMalloc(&d_output, sizeof(short4));

  // Launch kernel
  test_store_streaming_short4<<<1, 1>>>(d_output);
  cudaDeviceSynchronize();

  // Copy result back to host
  cudaMemcpy(&h_output, d_output, sizeof(short4), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_output);

  // Validate results
  if (h_output.x == 1 && h_output.y == 2 && h_output.z == 3 &&
      h_output.w == 4) {
    return true;
  }
  return false;
}

bool test_2() {
  short2 *d_output;
  short2 h_output;

  // Allocate memory on GPU
  cudaMalloc(&d_output, sizeof(short4));

  // Launch kernel
  test_store_streaming_short2<<<1, 1>>>(d_output);
  cudaDeviceSynchronize();

  // Copy result back to host
  cudaMemcpy(&h_output, d_output, sizeof(short2), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_output);

  // Validate results
  if (h_output.x == 1 && h_output.y == 2) {
    return true;
  }

  return false;
}

int main() {
  TEST(test_1);
  TEST(test_2);
  return 0;
}
