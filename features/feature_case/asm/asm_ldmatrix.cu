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

#define LAUNCH_TEST(TEST)                                                      \
  if (!launch_test_##TEST) {                                                   \
    return false;                                                              \
  }


template <int Shape_M, int Shape_N>
void calculate_num_matrices(int M, int N, int &NUM_MATRICES) {
  NUM_MATRICES = 0;

  if (M % Shape_M == 0 && N % Shape_N == 0) {
    NUM_MATRICES = (M * N) / (Shape_M * Shape_N);
  }
}

__device__ void ldmatrix_x1(void *addr, volatile int *r) {
    unsigned int addr_int = __cvta_generic_to_shared(addr);

    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                : "=r"(r[0])
                : "r"(addr_int));
}

__device__ void ldmatrix_x2(void *addr, volatile int *r) {
    unsigned int addr_int = __cvta_generic_to_shared(addr);

    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                  : "=r"(r[0]), "=r"(r[1])
                  : "r"(addr_int));
}

__device__ void ldmatrix_x4(void *addr, volatile int *r) {
    unsigned int addr_int = __cvta_generic_to_shared(addr);

    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                  : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                  : "r"(addr_int));
}

__device__ void ldmatrix_x1_trans(void *addr, volatile int *r) {
    unsigned int addr_int = __cvta_generic_to_shared(addr);

    asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
                  : "=r"(r[0])
                  : "r"(addr_int));
}

__device__ void ldmatrix_x2_trans(void *addr, volatile int *r) {
    unsigned int addr_int = __cvta_generic_to_shared(addr);

    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                  : "=r"(r[0]), "=r"(r[1])
                  : "r"(addr_int));
}

__device__ void ldmatrix_x4_trans(void *addr, volatile int *r) {
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
  volatile int r[X];

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

template <int Shape_M, int Shape_N, bool TRANS = false, int X = 1>
bool run_test_ldmatrix_b16(const int ROWS, const int COLS, const int NUM_MATRICES) {
  const int MATRIX_SIZE = Shape_M * Shape_N;
  const int TOTAL_ELEMENTS = ROWS * COLS;

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
      for (int c = 0; c < Shape_N; c++) {
        for (int r = 0; r < Shape_M; r++) {
          exp_output[k * MATRIX_SIZE + r * Shape_N + c] = static_cast<half>(val++);
        }
      }
    }
  } else {
    int val = 0;

    for (int k = 0; k < NUM_MATRICES; k++) {
      for (int r = 0; r < Shape_M; r++) {
        for (int c = 0; c < Shape_N; c++) {
          exp_output[k * MATRIX_SIZE + r * Shape_N + c] = static_cast<half>(val++);
        }
      }
    }
  }

  // Copy input matrix to device
  cudaMemcpy(d_input, h_input, TOTAL_ELEMENTS * sizeof(half), cudaMemcpyHostToDevice);

  int no_mat_block = NO_HALVES_PER_BLOCK / (Shape_M * Shape_N);
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
  for (int r = 0; r < ROWS; r++) {
    for (int c = 0; c < COLS; c++) {
      int index = r * COLS + c;

      float out = __half2float(h_output[index]);
      float exp_out = __half2float(exp_output[index]);

      if (out != exp_out) {
        std::cout << "Mismatch at index " << index << ": expected " << exp_out << ", got " << out << std::endl;
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

bool launch_test_ldmatrix_m8n8_b16_x1(const int M, const int N) {
  int NUM_MATRICES;
  calculate_num_matrices<8, 8>(M, N, NUM_MATRICES);

  if (NUM_MATRICES == 0) {
    std::cerr << "Matrix dimensions are not compatible with m8n8.x1 (b16): " << M << ", " << N << std::endl;
    return false;
  }

  bool correct;
  correct = run_test_ldmatrix_b16<8, 8, false, 1>(M, N, NUM_MATRICES);
  if (!correct) {
    std::cerr << "m8n8.x1 (b16) failed for dims: " << M << ", " << N << std::endl;
    return false;
  }

  correct = run_test_ldmatrix_b16<8, 8, true, 1>(M, N, NUM_MATRICES);
  if (!correct) {
    std::cerr << "m8n8.x1.trans (b16) failed for dims: " << M << ", " << N << std::endl;
    return false;
  }

  return true;
}

bool launch_test_ldmatrix_m8n8_b16_x2(const int M, const int N) {
  int NUM_MATRICES;
  calculate_num_matrices<8, 8>(M, N, NUM_MATRICES);

  if (NUM_MATRICES == 0) {
    std::cerr << "Matrix dimensions are not compatible with m8n8.x2 (b16): " << M << ", " << N << std::endl;
    return false;
  }

  bool correct;
  correct = run_test_ldmatrix_b16<8, 8, false, 2>(M, N, NUM_MATRICES);
  if (!correct) {
    std::cerr << "m8n8.x2 (b16) failed for dims: " << M << ", " << N << std::endl;
    return false;
  }

  correct = run_test_ldmatrix_b16<8, 8, true, 2>(M, N, NUM_MATRICES);
  if (!correct) {
    std::cerr << "m8n8.x2.trans (b16) failed for dims: " << M << ", " << N << std::endl;
    return false;
  }

  return true;
}

bool launch_test_ldmatrix_m8n8_b16_x4(const int M, const int N) {
  int NUM_MATRICES;
  calculate_num_matrices<8, 8>(M, N, NUM_MATRICES);

  if (NUM_MATRICES == 0) {
    std::cerr << "Matrix dimensions are not compatible with m8n8.x4 (b16): " << M << ", " << N << std::endl;
    return false;
  }

  bool correct;
  correct = run_test_ldmatrix_b16<8, 8, false, 4>(M, N, NUM_MATRICES);
  if (!correct) {
    std::cerr << "m8n8.x4 (b16) failed for dims: " << M << ", " << N << std::endl;
    return false;
  }

  correct = run_test_ldmatrix_b16<8, 8, true, 4>(M, N, NUM_MATRICES);
  if (!correct) {
    std::cerr << "m8n8.x4.trans (b16) failed for dims: " << M << ", " << N << std::endl;
    return false;
  }

  return true;
}

bool ldmatrix_m8n8_b16_x1() {
  LAUNCH_TEST(ldmatrix_m8n8_b16_x1(8, 8));
  LAUNCH_TEST(ldmatrix_m8n8_b16_x1(16, 16));
  LAUNCH_TEST(ldmatrix_m8n8_b16_x1(8, 16));
  LAUNCH_TEST(ldmatrix_m8n8_b16_x1(16, 8));
  LAUNCH_TEST(ldmatrix_m8n8_b16_x1(32, 32));

  return true;
}

bool ldmatrix_m8n8_b16_x2() {
  LAUNCH_TEST(ldmatrix_m8n8_b16_x2(8, 16));
  LAUNCH_TEST(ldmatrix_m8n8_b16_x2(16, 32));
  LAUNCH_TEST(ldmatrix_m8n8_b16_x2(8, 32));
  LAUNCH_TEST(ldmatrix_m8n8_b16_x2(16, 8));
  LAUNCH_TEST(ldmatrix_m8n8_b16_x2(32, 32));

  return true;
}

bool ldmatrix_m8n8_b16_x4() {
  LAUNCH_TEST(ldmatrix_m8n8_b16_x4(8, 32));
  LAUNCH_TEST(ldmatrix_m8n8_b16_x4(16, 64));
  LAUNCH_TEST(ldmatrix_m8n8_b16_x4(8, 64));
  LAUNCH_TEST(ldmatrix_m8n8_b16_x4(16, 32));
  LAUNCH_TEST(ldmatrix_m8n8_b16_x4(32, 32));

  return true;
}

int main() {
  TEST(ldmatrix_m8n8_b16_x1);
  TEST(ldmatrix_m8n8_b16_x2);
  TEST(ldmatrix_m8n8_b16_x4);

  return 0;
}
