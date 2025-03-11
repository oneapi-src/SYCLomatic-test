// ====------ asm_cp.cu ----------------------------------- *- CUDA -* ---===//
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

__device__ inline void cp_async4_pred(void *smem_ptr, const void *glob_ptr,
                                      bool pred = true) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("{\n"
               "   .reg .pred p;\n"
               "   setp.ne.b32 p, %0, 0;\n"
               "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
               "}\n"
               :
               : "r"((int)pred), "r"(smem), "l"(glob_ptr), "n"(BYTES));
}

__global__ void test_cp_async4_pred(int4 *d_out, int4 *d_in) {
  extern __shared__ int4 smem[];
  int tid = threadIdx.x;

  if (tid % 2) {
    cp_async4_pred(&smem[tid], &d_in[tid], true);

    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_all;" ::: "memory");

    __syncthreads();
    d_out[tid] = smem[tid];
  }
}

bool cp_async4_pred_test() {
  const int N = 256;
  size_t size = N * sizeof(int4);

  // Allocate host memory
  int4 *h_in = (int4 *)malloc(size);
  int4 *h_out = (int4 *)malloc(size);

  for (int i = 0; i < N; i++) {
    h_in[i] = make_int4(i, i + 1, i + 2, i + 3);
  }

  int4 *d_in, *d_out;
  cudaMalloc(&d_in, size);
  cudaMalloc(&d_out, size);

  // Copy input data to device
  cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

  // Launch kernel
  test_cp_async4_pred<<<1, N, size>>>(d_out, d_in);
  cudaDeviceSynchronize();

  // Copy output data back to host
  cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

  bool passed = true;
  for (int i = 0; i < N; i++) {
    if (i % 2 && (h_out[i].x != h_in[i].x || h_out[i].y != h_in[i].y ||
                  h_out[i].z != h_in[i].z || h_out[i].w != h_in[i].w)) {

      passed = false;
      std::cout << "Mismatch at index " << i << "\n";
      break;
    }
  }

  free(h_in);
  free(h_out);
  cudaFree(d_in);
  cudaFree(d_out);
  return passed;
}

__device__ inline void cp_async4(void *smem_ptr, const void *glob_ptr) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("{\n"
               "   cp.async.cg.shared.global [%0], [%1], %2;\n"
               "}\n" ::"r"(smem),
               "l"(glob_ptr), "n"(BYTES));
}

__global__ void test_cp_async4(int4 *d_out, int4 *d_in) {
  extern __shared__ int4 smem[];
  int tid = threadIdx.x;

  // Perform async copy
  cp_async4(&smem[tid], &d_in[tid]);

  // Ensure all async copies are completed before reading
  asm volatile("cp.async.commit_group;" ::: "memory");
  asm volatile("cp.async.wait_all;" ::: "memory");
  __syncthreads();

  // Store the result back to global memory for verification
  d_out[tid] = smem[tid];
}

bool cp_async4_test() {
  const int N = 256;
  size_t size = N * sizeof(int4);

  // Allocate host memory
  int4 *h_in = (int4 *)malloc(size);
  int4 *h_out = (int4 *)malloc(size);

  for (int i = 0; i < N; i++) {
    h_in[i] = make_int4(i, i + 1, i + 2, i + 3);
  }

  int4 *d_in, *d_out;
  cudaMalloc(&d_in, size);
  cudaMalloc(&d_out, size);

  // Copy input data to device
  cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

  // Launch kernel
  test_cp_async4<<<1, N, size>>>(d_out, d_in);
  cudaDeviceSynchronize();

  // Copy output data back to host
  cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

  bool passed = true;
  for (int i = 0; i < N; i++) {
    if (h_out[i].x != h_in[i].x || h_out[i].y != h_in[i].y ||
        h_out[i].z != h_in[i].z || h_out[i].w != h_in[i].w) {
      passed = false;
      std::cout << "Mismatch at index " << i << "\n";
      break;
    }
  }

  free(h_in);
  free(h_out);
  cudaFree(d_in);
  cudaFree(d_out);
  return passed;
}

int main() {
  TEST(cp_async4_pred_test);
  TEST(cp_async4_test);
  return 0;
}
