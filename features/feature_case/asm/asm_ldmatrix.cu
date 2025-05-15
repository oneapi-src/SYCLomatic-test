// ====------------- asm_ldmatrix.cu ------------- *- CUDA -* -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===---------------------------------------------------------------------===//

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>

#define NO_HALVES_PER_BLOCK 1024

#define TEST(FN)                                                               \
  {                                                                            \
    if (FN()) {                                                                \
      printf("Test " #FN " PASS\n");                                           \
    } else {                                                                   \
      printf("Test " #FN " FAIL\n");                                           \
      return 1;                                                                \
    }                                                                          \
  }

__device__ void ldmatrix_x1(void *addr, int *r) {
    unsigned int addr_int = __cvta_generic_to_shared(addr);

    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                : "=r"(r[0])
                : "r"(addr_int));
}

__device__ void ldmatrix_x2(void *addr, int *r) {
    unsigned int addr_int = __cvta_generic_to_shared(addr);

    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                  : "=r"(r[0]), "=r"(r[1])
                  : "r"(addr_int));
}

__device__ void ldmatrix_x4(void *addr, int *r) {
    unsigned int addr_int = __cvta_generic_to_shared(addr);

    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                  : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                  : "r"(addr_int));
}

__device__ void ldmatrix_x1_trans(void *addr, int *r) {
    unsigned int addr_int = __cvta_generic_to_shared(addr);

    asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
                  : "=r"(r[0])
                  : "r"(addr_int));
}

__device__ void ldmatrix_x2_trans(void *addr, int *r) {
    unsigned int addr_int = __cvta_generic_to_shared(addr);

    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                  : "=r"(r[0]), "=r"(r[1])
                  : "r"(addr_int));
}

__device__ void ldmatrix_x4_trans(void *addr, int *r) {
    unsigned int addr_int = __cvta_generic_to_shared(addr);

    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                  : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                  : "r"(addr_int));
}

template <bool TRANS = false, int X = 1>
__global__ void ldmatrix_kernel(half *input, half *output, const int ELEMENTS_PER_BLOCK) {
  const int MATRIX_SIZE = 8 * 8;

  __shared__ half shared_data[NO_HALVES_PER_BLOCK];

  int lane_id = threadIdx.x % 32;
  int warp_id = threadIdx.x / 32;

  for (int i = threadIdx.x; i < ELEMENTS_PER_BLOCK; i += blockDim.x) {
    shared_data[i] = input[blockIdx.x * ELEMENTS_PER_BLOCK + i];
  }

  __syncthreads();

  int row_offset = MATRIX_SIZE * X * warp_id;
  if (lane_id < X * 8)
    row_offset += (8 * lane_id);

  void *addr = shared_data + row_offset;
  int r[X];

  if (TRANS) {
    if (X == 1)
      ldmatrix_x1_trans(addr, r);
    else if (X == 2)
      ldmatrix_x2_trans(addr, r);
    else if (X == 4)
      ldmatrix_x4_trans(addr, r);
  } else {
    if (X == 1)
      ldmatrix_x1(addr, r);
    else if (X == 2)
      ldmatrix_x2(addr, r);
    else if (X == 4)
      ldmatrix_x4(addr, r);
  }

  for (int i = 0; i < X; i++) {
    int d_ind = i * MATRIX_SIZE + 2 * lane_id;

    if (d_ind + 1 < ELEMENTS_PER_BLOCK) {
      output[blockIdx.x * ELEMENTS_PER_BLOCK + MATRIX_SIZE * X * warp_id + d_ind]     = ((half *)(&r[i]))[0];
      output[blockIdx.x * ELEMENTS_PER_BLOCK + MATRIX_SIZE * X * warp_id + d_ind + 1] = ((half *)(&r[i]))[1];
    }
  }
}

template <bool TRANS = false, int X = 1>
bool run_test(const int ROWS, const int COLS, const int NUM_MATRICES) {
  const int MATRIX_SIZE = ROWS * COLS;
  const int TOTAL_ELEMENTS = NUM_MATRICES * MATRIX_SIZE;

  // Allocate host memory for matrices
  half *h_input = new half[TOTAL_ELEMENTS];
  half *h_output = new half[TOTAL_ELEMENTS];
  half *exp_output = new half[TOTAL_ELEMENTS];

  // Allocate device memory for matrices
  half *d_input;
  half *d_output;
  cudaMalloc(&d_input, TOTAL_ELEMENTS * sizeof(half));
  cudaMalloc(&d_output, TOTAL_ELEMENTS * sizeof(half));
  cudaMemset(d_output, 0, TOTAL_ELEMENTS * sizeof(half));

  // Initialize input matrix with some values
  for (int i = 0; i < TOTAL_ELEMENTS; i++) {
      h_input[i] = static_cast<half>(i);
  }

  // Initialize expected matrix with some values
  if (TRANS) {
    int val = 0;

    for (int k = 0; k < NUM_MATRICES; k++) {
      for (int c = 0; c < COLS; c++) {
        for (int r = 0; r < ROWS; r++) {
          exp_output[k * MATRIX_SIZE + r * COLS + c] = static_cast<half>(val++);
        }
      }
    }
  } else {
    int val = 0;

    for (int k = 0; k < NUM_MATRICES; k++) {
      for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
          exp_output[k * MATRIX_SIZE + r * COLS + c] = static_cast<half>(val++);
        }
      }
    }
  }

  // Copy input matrix to device
  cudaMemcpy(d_input, h_input, TOTAL_ELEMENTS * sizeof(half), cudaMemcpyHostToDevice);

  int no_mat_block = NO_HALVES_PER_BLOCK / (8 * 8);
  int no_blocks = NUM_MATRICES / no_mat_block;
  int no_threads;
  if (no_blocks) {
    no_threads = 32 * (no_mat_block / X);
  } else {
    no_blocks = 1;
    no_threads = 32 * (NUM_MATRICES / X);
  }
  const int ELEMENTS_PER_BLOCK = no_threads * X * 2;

  // Launch kernel
  ldmatrix_kernel<TRANS, X><<<no_blocks, no_threads>>>(d_input, d_output, ELEMENTS_PER_BLOCK);

  cudaDeviceSynchronize();

  // Copy output matrix back to host
  cudaMemcpy(h_output, d_output, TOTAL_ELEMENTS * sizeof(half), cudaMemcpyDeviceToHost);

  // Compare input & expected matrices data
  bool pass = true;
  for (int k = 0; k < NUM_MATRICES; k++) {
    for (int r = 0; r < ROWS; r++) {
      for (int c = 0; c < COLS; c++) {
        int index = k * MATRIX_SIZE + r * COLS + c;

        float out = __half2float(h_output[index]);
        float exp_out = __half2float(exp_output[index]);

        if (out != exp_out)
          pass = false;
      }
    }
  }

  // Cleanup
  delete[] h_input;
  delete[] h_output;
  cudaFree(d_input);
  cudaFree(d_output);

  return pass;
}

bool ldmatrix_m8n8_b16_x1() {
  // Matrix dimensions
  const int ROWS = 8;
  const int COLS = 8;
  const int NUM_MATRICES = 1;

  return run_test<false, 1>(ROWS, COLS, NUM_MATRICES);
}

bool ldmatrix_m8n8_b16_x2() {
  // Matrix dimensions
  const int ROWS = 8;
  const int COLS = 8;
  const int NUM_MATRICES = 2;

  return run_test<false, 2>(ROWS, COLS, NUM_MATRICES);
}

bool ldmatrix_m8n8_b16_x4() {
  // Matrix dimensions
  const int ROWS = 8;
  const int COLS = 8;
  const int NUM_MATRICES = 4;

  return run_test<false, 4>(ROWS, COLS, NUM_MATRICES);
}

bool ldmatrix_m8n8_b16_x1_trans() {
  // Matrix dimensions
  const int ROWS = 8;
  const int COLS = 8;
  const int NUM_MATRICES = 1;

  return run_test<true, 1>(ROWS, COLS, NUM_MATRICES);
}

bool ldmatrix_m8n8_b16_x2_trans() {
  // Matrix dimensions
  const int ROWS = 8;
  const int COLS = 8;
  const int NUM_MATRICES = 2;

  return run_test<true, 2>(ROWS, COLS, NUM_MATRICES);
}

bool ldmatrix_m8n8_b16_x4_trans() {
  // Matrix dimensions
  const int ROWS = 8;
  const int COLS = 8;
  const int NUM_MATRICES = 4;

  return run_test<true, 4>(ROWS, COLS, NUM_MATRICES);
}

int main() {
  TEST(ldmatrix_m8n8_b16_x1);
  TEST(ldmatrix_m8n8_b16_x2);
  TEST(ldmatrix_m8n8_b16_x4);

  TEST(ldmatrix_m8n8_b16_x1_trans);
  TEST(ldmatrix_m8n8_b16_x2_trans);
  TEST(ldmatrix_m8n8_b16_x4_trans);

  return 0;
}
