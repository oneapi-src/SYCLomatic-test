// ====--------------- asm_mma.cu --------------- *- CUDA -* --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===---------------------------------------------------------------------===//

#include <cuda_runtime.h>
#include <cuda_fp16.h>
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

template <typename ABType, typename CDType>
__host__ void initialize_matrices(ABType *A, ABType *B, CDType *C, CDType *D, int M, int N, int K, int NUM_MATRICES = 1) {
  for (int N_MAT = 0; N_MAT < NUM_MATRICES; N_MAT++) {
    int A_OFFSET = N_MAT * M * K;
    int B_OFFSET = N_MAT * K * N;
    int C_OFFSET = N_MAT * M * N;
    int D_OFFSET = N_MAT * M * N;

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < K; ++j) {
        A[A_OFFSET + i * K + j] = i * K + j;
      }
    }

    for (int i = 0; i < K; ++i) {
      for (int j = 0; j < N; ++j) {
        B[B_OFFSET + i * N + j] = i * N + j;
      }
    }

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        C[C_OFFSET + i * N + j] = i * N + j;
      }
    }

    for (int i = 0; i < M * N; i++) {
      D[D_OFFSET + i] = 0.0;
    }
  }
}

__host__ void initialize_matrices(int8_t *A, int8_t *B, int *C, int *D, int M, int N, int K, int NUM_MATRICES = 1) {
  for (int N_MAT = 0; N_MAT < NUM_MATRICES; N_MAT++) {
    int A_OFFSET = N_MAT * M * K;
    int B_OFFSET = N_MAT * K * N;
    int C_OFFSET = N_MAT * M * N;
    int D_OFFSET = N_MAT * M * N;

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < K; ++j) {
        A[A_OFFSET + i * K + j] = (i * K + j) % 8;
      }
    }

    for (int i = 0; i < K; ++i) {
      for (int j = 0; j < N; ++j) {
        B[B_OFFSET + i * N + j] = (i * N + j) % 8;
      }
    }

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        C[C_OFFSET + i * N + j] = i * N + j;
      }
    }

    for (int i = 0; i < M * N; i++) {
      D[D_OFFSET + i] = 0;
    }
  }
}


template <typename ABType, typename CDType>
void matrix_multiplication_cpu(ABType *A, ABType *B, CDType *C, CDType *D, int M, int N, int K, int NUM_MATRICES = 1) {
  for (int N_MAT = 0; N_MAT < NUM_MATRICES; N_MAT++) {
    int A_OFFSET = N_MAT * M * K;
    int B_OFFSET = N_MAT * K * N;
    int C_OFFSET = N_MAT * M * N;
    int D_OFFSET = N_MAT * M * N;

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        CDType sum = C[C_OFFSET + i * N + j];
        for (int k = 0; k < K; ++k) {
          sum += static_cast<CDType>(A[A_OFFSET + i * K + k]) * static_cast<CDType>(B[B_OFFSET + k * N + j]);
        }
        D[D_OFFSET + i * N + j] = sum;
      }
    }
  }
}

void matrix_multiplication_cpu(half *A, half *B, half *C, half *D, int M, int N, int K, int NUM_MATRICES = 1) {
  for (int N_MAT = 0; N_MAT < NUM_MATRICES; N_MAT++) {
    int A_OFFSET = N_MAT * M * K;
    int B_OFFSET = N_MAT * K * N;
    int C_OFFSET = N_MAT * M * N;
    int D_OFFSET = N_MAT * M * N;

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        float sum = __half2float(C[C_OFFSET + i * N + j]);
        for (int k = 0; k < K; ++k) {
          sum += __half2float(A[A_OFFSET + i * K + k]) * __half2float(B[B_OFFSET + k * N + j]);
        }
        D[D_OFFSET + i * N + j] = __float2half(sum);
      }
    }
  }
}

template <typename T>
bool check_result(int M, int N, T *D, T *D_ref, int NUM_MATRICES = 1) {
  bool correct = true;

  for (int N_MAT = 0; N_MAT < NUM_MATRICES; N_MAT++) {
    int D_OFFSET = N_MAT * M * N;

    for (int i = 0; i < M * N; i++) {
      if (fabs(static_cast<float>(D[D_OFFSET + i]) - static_cast<float>(D_ref[D_OFFSET + i])) > 1e-3) {
        std::cout << "Mismatch at index " << i << ": "
                  << "Expected: " << static_cast<float>(D_ref[D_OFFSET + i]) << ", "
                  << "Got: " << static_cast<float>(D[D_OFFSET + i]) << std::endl;
        correct = false;
        break;
      }
    }
  }

  return correct;
}

#define WARP_SIZE 32
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

__global__ void mma_kernel_m16n8k16_ptx_f16_f32(half *A, half *B, float *C, float *D, int M, int N, int K) {
  const int WARP_ID = threadIdx.x / WARP_SIZE;
  const int LANE_ID = threadIdx.x % WARP_SIZE;

  const int C_THREAD_ROW = LANE_ID / 4;
  const int C_THREAD_COL = LANE_ID % 4;

  int A_OFFSET = WARP_ID * M * K;
  int B_OFFSET = WARP_ID * K * N;
  int CD_OFFSET = WARP_ID * M * N;

  half2 ra[4];
  half2 rb[2];
  float c[4] = {0};
  float d[4] = {0};

  half *a = reinterpret_cast<half *>(ra);
  for (int i = 0; i < 8; i++) {
    int r_off = 8;
    if (i < 2 || (i >= 4 && i < 6)) {
      r_off = 0;
    }

    int c_off = 0;
    if (i >= 4) {
      c_off = 8;
    }

    a[i] = A[A_OFFSET + OFFSET(C_THREAD_ROW + r_off, (C_THREAD_COL * 2) + (i & 0x1) + c_off, K)];
  }

  half *b = reinterpret_cast<half *>(rb);
  for (int i = 0; i < 4; i++) {
    int r_off = 0;
    if (i >= 2) {
      r_off = 8;
    }

    b[i] = B[B_OFFSET + OFFSET((C_THREAD_COL * 2) + (i & 0x1) + r_off, C_THREAD_ROW, N)];
  }

  for (int i = 0; i < 4; i++) {
    if (i < 2) {
      c[i] = C[CD_OFFSET + OFFSET(C_THREAD_ROW, (C_THREAD_COL * 2) + (i & 0x1), N)];
    } else {
      c[i] = C[CD_OFFSET + OFFSET(C_THREAD_ROW + 8, (C_THREAD_COL * 2) + (i & 0x1), N)];
    }
  }

  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      " { %0, %1, %2, %3 }, "
      " { %4, %5, %6, %7 }, "
      " { %8, %9 }, "
      " { %10, %11, %12, %13 };"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(*(reinterpret_cast<int *>(&ra[0]))),
        "r"(*(reinterpret_cast<int *>(&ra[1]))),
        "r"(*(reinterpret_cast<int *>(&ra[2]))),
        "r"(*(reinterpret_cast<int *>(&ra[3]))),
        "r"(*(reinterpret_cast<int *>(&rb[0]))),
        "r"(*(reinterpret_cast<int *>(&rb[1]))),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));

  for (int i = 0; i < 4; i++) {
    if (i < 2) {
      D[CD_OFFSET + OFFSET(C_THREAD_ROW, (C_THREAD_COL * 2) + (i & 0x1), N)] = d[i];
    } else {
      D[CD_OFFSET + OFFSET(C_THREAD_ROW + 8, (C_THREAD_COL * 2) + (i & 0x1), N)] = d[i];
    }
  }
}

__global__ void mma_kernel_m16n8k16_s8_s32(int8_t *A, int8_t *B, int *C, int *D, int M, int N, int K) {
  const int WARP_ID = threadIdx.x / WARP_SIZE;
  const int LANE_ID = threadIdx.x % WARP_SIZE;

  const int C_THREAD_ROW = LANE_ID / 4;
  const int C_THREAD_COL = LANE_ID % 4;

  int A_OFFSET = WARP_ID * M * K;
  int B_OFFSET = WARP_ID * K * N;
  int CD_OFFSET = WARP_ID * M * N;

  int a[2] = {0};
  int b = 0;
  int c[4] = {0};
  int d[4] = {0};

  auto *ra = reinterpret_cast<int8_t *>(a);
  auto *rb = reinterpret_cast<int8_t *>(&b);

  for (int i = 0; i < 8; i++) {
    int r_off = 0;
    if (i >= 4) {
      r_off = 8;
    }
    ra[i] = A[A_OFFSET + OFFSET(C_THREAD_ROW + r_off, (C_THREAD_COL * 4) + (i & 0x3), K)];
  }

  for (int i = 0; i < 4; i++) {
    rb[i] = B[B_OFFSET + OFFSET((C_THREAD_COL * 4) + i, C_THREAD_ROW, N)];
  }

  for (int i = 0; i < 4; i++) {
    if (i < 2)
      c[i] = C[CD_OFFSET + OFFSET(C_THREAD_ROW, (C_THREAD_COL * 2) + (i & 0x1), N)];
    else
      c[i] = C[CD_OFFSET + OFFSET(C_THREAD_ROW + 8, (C_THREAD_COL * 2) + (i & 0x1), N)];
  }

  asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
      " { %0, %1, %2, %3 }, "
      " { %4, %5 }, "
      " { %6 }, "
      " { %7, %8, %9, %10 };"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a[0]), "r"(a[1]),
        "r"(b),
        "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));

  for (int i = 0; i < 4; i++) {
    if (i < 2)
      D[CD_OFFSET + OFFSET(C_THREAD_ROW, (C_THREAD_COL * 2) + (i & 0x1), N)] = d[i];
    else
      D[CD_OFFSET + OFFSET(C_THREAD_ROW + 8, (C_THREAD_COL * 2) + (i & 0x1), N)] = d[i];
  }
}

bool mma_m16n8k16_f16_f32() {
  const int M = 16;
  const int N = 8;
  const int K = 16;
  const int NUM_MATRICES = 2;

  half *d_A, *d_B;
  float *d_C, *d_D;
  half h_A[NUM_MATRICES * M * K], h_B[NUM_MATRICES * K * N];
  float h_C[NUM_MATRICES * M * N], h_D[NUM_MATRICES * M * N];
  float h_D_ref[NUM_MATRICES * M * N];

  initialize_matrices(h_A, h_B, h_C, h_D, M, N, K, NUM_MATRICES);

  matrix_multiplication_cpu(h_A, h_B, h_C, h_D_ref, M, N, K, NUM_MATRICES);

  cudaMalloc(&d_A, NUM_MATRICES * M * K * sizeof(half));
  cudaMalloc(&d_B, NUM_MATRICES * K * N * sizeof(half));
  cudaMalloc(&d_C, NUM_MATRICES * M * N * sizeof(float));
  cudaMalloc(&d_D, NUM_MATRICES * M * N * sizeof(float));

  cudaMemcpy(d_A, h_A, NUM_MATRICES * M * K * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, NUM_MATRICES * K * N * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, h_C, NUM_MATRICES * M * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_D, h_D, NUM_MATRICES * M * N * sizeof(float), cudaMemcpyHostToDevice);

  int no_mat_blocks = 4;
  int no_blocks = NUM_MATRICES / no_mat_blocks;
  int no_threads;
  if (no_blocks) {
    no_threads = WARP_SIZE * no_mat_blocks;
  } else {
    no_blocks = 1;
    no_threads = WARP_SIZE * NUM_MATRICES;
  }

  mma_kernel_m16n8k16_ptx_f16_f32<<<no_blocks, no_threads>>>(d_A, d_B, d_C, d_D, M, N, K);
  cudaDeviceSynchronize();
  cudaMemcpy(h_D, d_D, NUM_MATRICES * M * N * sizeof(float), cudaMemcpyDeviceToHost);

  bool correct = check_result(M, N, h_D, h_D_ref, NUM_MATRICES);

  std::cout << "m16n8k16 (f32.f16.f16.f32): " << (correct ? "PASSED" : "FAILED") << std::endl;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_D);

  return correct;
}

bool mma_m16n8k16_s8_s32() {
  const int M = 16;
  const int N = 8;
  const int K = 16;
  const int NUM_MATRICES = 2;

  int8_t *d_A, *d_B;
  int *d_C, *d_D;
  int8_t h_A[NUM_MATRICES * M * K], h_B[NUM_MATRICES * K * N];
  int h_C[NUM_MATRICES * M * N], h_D[NUM_MATRICES * M * N];
  int h_D_ref[NUM_MATRICES * M * N];

  initialize_matrices(h_A, h_B, h_C, h_D, M, N, K, NUM_MATRICES);

  matrix_multiplication_cpu(h_A, h_B, h_C, h_D_ref, M, N, K, NUM_MATRICES);

  cudaMalloc(&d_A, NUM_MATRICES * M * K * sizeof(int8_t));
  cudaMalloc(&d_B, NUM_MATRICES * K * N * sizeof(int8_t));
  cudaMalloc(&d_C, NUM_MATRICES * M * N * sizeof(int));
  cudaMalloc(&d_D, NUM_MATRICES * M * N * sizeof(int));

  cudaMemcpy(d_A, h_A, NUM_MATRICES * M * K * sizeof(int8_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, NUM_MATRICES * K * N * sizeof(int8_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, h_C, NUM_MATRICES * M * N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_D, h_D, NUM_MATRICES * M * N * sizeof(int), cudaMemcpyHostToDevice);

  int no_mat_blocks = 4;
  int no_blocks = NUM_MATRICES / no_mat_blocks;
  int no_threads;
  if (no_blocks) {
    no_threads = WARP_SIZE * no_mat_blocks;
  } else {
    no_blocks = 1;
    no_threads = WARP_SIZE * NUM_MATRICES;
  }

  mma_kernel_m16n8k16_s8_s32<<<no_blocks, no_threads>>>(d_A, d_B, d_C, d_D, M, N, K);
  cudaDeviceSynchronize();
  cudaMemcpy(h_D, d_D, NUM_MATRICES * M * N * sizeof(int), cudaMemcpyDeviceToHost);

  bool correct = check_result(M, N, h_D, h_D_ref, NUM_MATRICES);

  std::cout << "m16n8k16 (s32.s8.s8.s32): " << (correct ? "PASSED" : "FAILED") << std::endl;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_D);

  return correct;
}

int main() {
  TEST(mma_m16n8k16_f16_f32);
  TEST(mma_m16n8k16_s8_s32);

  return 0;
}
