// ====--------------- asm_mma.cu --------------- *- CUDA -* --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===---------------------------------------------------------------------===//

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
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

#define LAUNCH_TEST(TEST) \
  if (!launch_test_##TEST) { \
    return false; \
  }

template <typename ABType, typename CDType>
__host__ void initialize_matrices(ABType *A, ABType *B, CDType *C, CDType *D, int M, int N, int K, int A_NUM_MAT, int B_NUM_MAT, int CD_NUM_MAT) {
  for (int N_MAT = 0; N_MAT < A_NUM_MAT; N_MAT++) {
    int A_OFFSET = N_MAT * M * K;

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < K; ++j) {
        A[A_OFFSET + i * K + j] = i * K + j;
      }
    }
  }

  for (int N_MAT = 0; N_MAT < B_NUM_MAT; N_MAT++) {
    int B_OFFSET = N_MAT * K * N;

    for (int i = 0; i < K; ++i) {
      for (int j = 0; j < N; ++j) {
        B[B_OFFSET + i * N + j] = i * N + j;
      }
    }
  }


  for (int N_MAT = 0; N_MAT < CD_NUM_MAT; N_MAT++) {
    int CD_OFFSET = N_MAT * M * N;

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        C[CD_OFFSET + i * N + j] = i * N + j;
        D[CD_OFFSET + i * N + j] = 0.0;
      }
    }
  }
}


template <typename ABType, typename CDType>
void matrix_multiplication_cpu(ABType *A, ABType *B, CDType *C, CDType *D, int M, int N, int K, int A_NUM_MAT, int B_NUM_MAT, int CD_NUM_MAT) {
  for (int N_MAT = 0; N_MAT < CD_NUM_MAT; N_MAT++) {
    int A_OFFSET = (N_MAT % A_NUM_MAT) * (M * K);
    int B_OFFSET = (N_MAT % B_NUM_MAT) * (K * N);
    int CD_OFFSET = N_MAT * M * N;

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        CDType sum = C[CD_OFFSET + i * N + j];
        for (int k = 0; k < K; ++k) {
          sum += static_cast<CDType>(A[A_OFFSET + i * K + k]) * static_cast<CDType>(B[B_OFFSET + k * N + j]);
        }
        D[CD_OFFSET + i * N + j] = sum;
      }
    }
  }
}

template <typename T>
bool check_result(int M, int N, T *D, T *D_ref) {
  bool correct = true;

  for (int i = 0; i < M * N; i++) {
    if (fabs(static_cast<float>(D[i]) - static_cast<float>(D_ref[i])) > 1e-3) {
      std::cout << "Mismatch at index " << i << ": "
                << "Expected: " << static_cast<float>(D_ref[i]) << ", "
                << "Got: " << static_cast<float>(D[i]) << std::endl;
      correct = false;
      break;
    }
  }

  return correct;
}

template <int Shape_M, int Shape_N, int Shape_K>
void calculate_num_matrices(int M, int N, int K, int &A_NUM_MAT, int &B_NUM_MAT, int &CD_NUM_MAT) {
  A_NUM_MAT = B_NUM_MAT = CD_NUM_MAT = 0;

  if (M % Shape_M == 0 && N % Shape_N == 0 && K % Shape_K == 0) {
    A_NUM_MAT = (M * K) / (Shape_M * Shape_K);
    B_NUM_MAT = (K * N) / (Shape_K * Shape_N);
    CD_NUM_MAT = (M * N) / (Shape_M * Shape_N);
  }
}

#define WARP_SIZE 32
#define OFFSET(ROW, COL, ld) ((ROW) * (ld) + (COL))
#define IN_BOUND_A(OFFSET) ((OFFSET) >= 0 && (OFFSET) < (M * K))
#define IN_BOUND_B(OFFSET) ((OFFSET) >= 0 && (OFFSET) < (K * N))
#define IN_BOUND_CD(OFFSET) ((OFFSET) >= 0 && (OFFSET) < (M * N))

template<int Shape_M, int Shape_N, int Shape_K>
__global__ void mma_kernel_m16n8k8_ptx_bf16_f32(__nv_bfloat16 *A, __nv_bfloat16 *B, float *C, float *D, int M, int N, int K, int A_NUM_MAT, int B_NUM_MAT, int CD_NUM_MAT) {
  const int THREAD_IDX = threadIdx.x + blockIdx.x * blockDim.x;
  const int WARP_ID = THREAD_IDX / WARP_SIZE;
  const int LANE_ID = THREAD_IDX % WARP_SIZE;

  const int THREAD_ROW = LANE_ID / 4;
  const int THREAD_COL = LANE_ID % 4;

  int A_OFFSET = (WARP_ID % A_NUM_MAT) * Shape_M * Shape_K;
  int B_OFFSET = (WARP_ID % B_NUM_MAT) * Shape_K * Shape_N;
  int CD_OFFSET = (WARP_ID % CD_NUM_MAT) * Shape_M * Shape_N;

  __nv_bfloat162 a[2];
  __nv_bfloat162 b;
  float c[4];
  float d[4];

  auto ra = reinterpret_cast<__nv_bfloat16 *>(a);
  auto rb = reinterpret_cast<__nv_bfloat16 *>(&b);

  for (int i = 0; i < 4; i++) {
    int load_offset = A_OFFSET + OFFSET(THREAD_ROW + 8 * (i >> 1), (THREAD_COL * 2) + (i & 0x1), Shape_K);

    if (IN_BOUND_A(load_offset)) {
      ra[i] = A[load_offset];
    }
  }

  for (int i = 0; i < 2; i++) {
    int load_offset = B_OFFSET + OFFSET((THREAD_COL * 2) + i, THREAD_ROW, Shape_N);

    if (IN_BOUND_B(load_offset)) {
      rb[i] = B[load_offset];
    }
  }

  for (int i = 0; i < 4; i++) {
    int load_offset = CD_OFFSET + OFFSET(THREAD_ROW + 8 * (i >> 1), (THREAD_COL * 2) + (i & 0x1), Shape_N);

    if (IN_BOUND_CD(load_offset)) {
      c[i] = C[load_offset];
    }
  }

  asm("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
      " { %0, %1, %2, %3 }, "
      " { %4, %5 }, "
      " { %6 }, "
      " { %7, %8, %9, %10 };"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(*(reinterpret_cast<int *>(&a[0]))),
        "r"(*(reinterpret_cast<int *>(&a[1]))),
        "r"(*(reinterpret_cast<int *>(&b))),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));

  for (int i = 0; i < 4; i++) {
    int load_offset = CD_OFFSET + OFFSET(THREAD_ROW + 8 * (i >> 1), (THREAD_COL * 2) + (i & 0x1), Shape_N);

    if (IN_BOUND_CD(load_offset)) {
      D[load_offset] = d[i];
    }
  }
}

bool run_test_mma_m16n8k8_bf16_f32(const int M, const int N, const int K) {
  int A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT;
  calculate_num_matrices<16, 8, 8>(M, N, K, A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT);

  if (A_NUM_MAT == 0 || B_NUM_MAT == 0 || CD_NUM_MAT == 0) {
    std::cerr << "Matrix dimensions are not compatible with m16n8k8" << std::endl;
    return false;
  }

  __nv_bfloat16 *d_A, *d_B;
  float *d_C, *d_D;
  __nv_bfloat16 h_A[M * K], h_B[K * N];
  float h_C[M * N], h_D[M * N];
  float h_D_ref[M * N];

  initialize_matrices(h_A, h_B, h_C, h_D, 16, 8, 8, A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT);

  matrix_multiplication_cpu(h_A, h_B, h_C, h_D_ref, 16, 8, 8, A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT);

  cudaMalloc(&d_A, M * K * sizeof(__nv_bfloat16));
  cudaMalloc(&d_B, K * N * sizeof(__nv_bfloat16));
  cudaMalloc(&d_C, M * N * sizeof(float));
  cudaMalloc(&d_D, M * N * sizeof(float));

  cudaMemcpy(d_A, h_A, M * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, K * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_D, h_D, M * N * sizeof(float), cudaMemcpyHostToDevice);

  int no_mat_blocks = 4;
  int no_blocks = CD_NUM_MAT / no_mat_blocks;
  int no_threads;
  if (no_blocks) {
    no_threads = WARP_SIZE * no_mat_blocks;
  } else {
    no_blocks = 1;
    no_threads = WARP_SIZE * CD_NUM_MAT;
  }

  mma_kernel_m16n8k8_ptx_bf16_f32<16, 8, 8><<<no_blocks, no_threads>>>(d_A, d_B, d_C, d_D, M, N, K, A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT);
  cudaDeviceSynchronize();
  cudaMemcpy(h_D, d_D, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  bool correct = check_result(M, N, h_D, h_D_ref);

  std::cout << "m16n8k8 (f32.bf16.bf16.f32): " << (correct ? "PASSED" : "FAILED") << std::endl;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_D);

  return correct;
}

bool mma_m16n8k8_bf16_f32() {
  LAUNCH_TEST(mma_m16n8k8_bf16_f32(16, 8, 8));
  LAUNCH_TEST(mma_m16n8k8_bf16_f32(32, 16, 16));
  LAUNCH_TEST(mma_m16n8k8_bf16_f32(16, 16, 8));
  LAUNCH_TEST(mma_m16n8k8_bf16_f32(16, 16, 16));
  LAUNCH_TEST(mma_m16n8k8_bf16_f32(32, 32, 32));

  return true;
}

int main() {
  TEST(mma_m16n8k8_bf16_f32);

  return 0;
}
