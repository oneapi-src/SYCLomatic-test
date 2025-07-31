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
#include <stdint.h>

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

__host__ void initialize_matrices(int8_t *A, int8_t *B, int *C, int *D, int M, int N, int K, int A_NUM_MAT, int B_NUM_MAT, int CD_NUM_MAT) {
  for (int N_MAT = 0; N_MAT < A_NUM_MAT; N_MAT++) {
    int A_OFFSET = N_MAT * M * K;

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < K; ++j) {
        A[A_OFFSET + i * K + j] = (i * K + j) % 8;
      }
    }
  }

  for (int N_MAT = 0; N_MAT < B_NUM_MAT; N_MAT++) {
    int B_OFFSET = N_MAT * K * N;

    for (int i = 0; i < K; ++i) {
      for (int j = 0; j < N; ++j) {
        B[B_OFFSET + i * N + j] = (i * N + j) % 8;
      }
    }
  }

  for (int N_MAT = 0; N_MAT < CD_NUM_MAT; N_MAT++) {
    int CD_OFFSET = N_MAT * M * N;

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        C[CD_OFFSET + i * N + j] = i * N + j;
        D[CD_OFFSET + i * N + j] = 0;
      }
    }
  }
}


__host__ void initialize_matrices(half *A, half *B, half *C, half *D, int M,
                                  int N, int K, int A_NUM_MAT, int B_NUM_MAT,
                                  int CD_NUM_MAT) {
  for (int N_MAT = 0; N_MAT < A_NUM_MAT; N_MAT++) {
    int A_OFFSET = N_MAT * M * K;

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < K; ++j) {
        A[A_OFFSET + i * K + j] = __float2half(((i * K + j) / 4.0f));
      }
    }
  }

  for (int N_MAT = 0; N_MAT < B_NUM_MAT; N_MAT++) {
    int B_OFFSET = N_MAT * K * N;

    for (int i = 0; i < K; ++i) {
      for (int j = 0; j < N; ++j) {
        B[B_OFFSET + i * N + j] = __float2half(((i * N + j) / 4.0f));
      }
    }
  }

  for (int N_MAT = 0; N_MAT < CD_NUM_MAT; N_MAT++) {
    int CD_OFFSET = N_MAT * M * N;

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        C[CD_OFFSET + i * N + j] = __float2half((i * N + j) * 1.0f);
        D[CD_OFFSET + i * N + j] = 0.0f;
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
/*
1. Properties of half (float16)
Limited precision: float16 has only 5 bits for the exponent and 10 bits for the
mantissa, giving roughly 3 decimal digits of precision.

Non-uniform precision: The closer to zero, the higher the precision; the closer
to the maximum value, the lower the precision.

2. Difference of CPU vs. GPU on half support
* On the GPU (e.g., Tensor Core or native FP16 ALUs):
Many GPUs, especially NVIDIA ones, support native half x half operations,
either:

half x half -> half or half x half -> float

Using Tensor Cores or PTX instructions like mma.sync, often:

half x half -> float (accumulate) -> rounded to half at the end

So the GPU may retain intermediate precision in float, which can differ from a
float-roundtrip on the CPU.

* On the CPU:
CPUs usually don't have native half-precision units, so the operation goes
like:

Convert half to float -> multiply as float -> convert result back to half

This involves two rounding steps:

From half to float (no loss)

From float to half (can introduce significant rounding errors)

For example, if you multiply 60000 x 2 as float, you get 120000. But float16
can't represent 120000, so it becomes inf.
*/
// Note: The results of cpu vs. gpu is different, especially for larger number.
void matrix_multiplication_cpu(half *A, half *B, half *C, half *D, int M, int N, int K, int A_NUM_MAT, int B_NUM_MAT, int CD_NUM_MAT) {
  for (int N_MAT = 0; N_MAT < CD_NUM_MAT; N_MAT++) {
    int A_OFFSET = (N_MAT % A_NUM_MAT) * (M * K);
    int B_OFFSET = (N_MAT % B_NUM_MAT) * (K * N);
    int CD_OFFSET = N_MAT * M * N;

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        float sum = __half2float(C[CD_OFFSET + i * N + j]);
        for (int k = 0; k < K; ++k) {
          sum += __half2float(A[A_OFFSET + i * K + k]) * __half2float(B[B_OFFSET + k * N + j]);
        }
        D[CD_OFFSET + i * N + j] = __float2half(sum);
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
__global__ void mma_kernel_m8n8k4_ptx_f16_f32(half *A, half *B, float *C, float *D, int M, int N, int K, int A_NUM_MAT, int B_NUM_MAT, int CD_NUM_MAT) {
  const int THREAD_IDX = threadIdx.x + blockIdx.x * blockDim.x;
  const int WARP_ID = THREAD_IDX / WARP_SIZE;
  const int LANE_ID = THREAD_IDX % WARP_SIZE;

  const int THREAD_ROW = LANE_ID % 4;
  const int THREAD_COL = LANE_ID % 4;

  int A_OFFSET = (WARP_ID % A_NUM_MAT) * Shape_M * Shape_K;
  int B_OFFSET = (WARP_ID % B_NUM_MAT) * Shape_K * Shape_N;
  int CD_OFFSET = (WARP_ID % CD_NUM_MAT) * Shape_M * Shape_N;

  half2 a[2];
  half2 b[2];
  float c[8];
  float d[8];

  auto ra = reinterpret_cast<half *>(a);
  auto rb = reinterpret_cast<half *>(b);

  for (int i = 0; i < 4; i++) {
    int load_offset;

    if (LANE_ID < 16) {
      load_offset = A_OFFSET + OFFSET(i, THREAD_COL, Shape_K);
    } else {
      load_offset = A_OFFSET + OFFSET(i + 4, THREAD_COL, Shape_K);
    }

    if (IN_BOUND_A(load_offset)) {
      ra[i] = A[load_offset];
    }
  }

  for (int i = 0; i < 4; i++) {
    int load_offset;

    if (LANE_ID < 16) {
      load_offset = B_OFFSET + OFFSET(THREAD_ROW, i, Shape_N);
    } else {
      load_offset = B_OFFSET + OFFSET(THREAD_ROW, i + 4, Shape_N);
    }

    if (IN_BOUND_B(load_offset)) {
      rb[i] = B[load_offset];
    }
  }

  for (int i = 0; i < 8; i++) {
    int load_offset;

    if (LANE_ID < 16) {
      load_offset = CD_OFFSET + OFFSET((LANE_ID & 0b1) + (i & 0b10), (i & 0b100) + (LANE_ID & 0b10) + (i & 0b1), Shape_N);
    } else {
      load_offset = CD_OFFSET + OFFSET((LANE_ID & 0b1) + (i & 0b10) + 4, (i & 0b100) + (LANE_ID & 0b10) + (i & 0b1), Shape_N);
    }

    if (IN_BOUND_CD(load_offset)) {
      c[i] = C[load_offset];
    }
  }

  asm("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 "
      " { %0, %1, %2, %3, %4, %5, %6, %7 }, "
      " { %8, %9 }, "
      " { %10, %11 }, "
      " { %12, %13, %14, %15, %16, %17, %18, %19 };"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "r"(*(reinterpret_cast<int *>(&a[0]))),
        "r"(*(reinterpret_cast<int *>(&a[1]))),
        "r"(*(reinterpret_cast<int *>(&b[0]))),
        "r"(*(reinterpret_cast<int *>(&b[1]))),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "f"(c[4]), "f"(c[5]), "f"(c[6]), "f"(c[7]));

  for (int i = 0; i < 8; i++) {
    int load_offset;

    if (LANE_ID < 16) {
      load_offset = CD_OFFSET + OFFSET((LANE_ID & 0b1) + (i & 0b10), (i & 0b100) + (LANE_ID & 0b10) + (i & 0b1), Shape_N);
    } else {
      load_offset = CD_OFFSET + OFFSET((LANE_ID & 0b1) + (i & 0b10) + 4, (i & 0b100) + (LANE_ID & 0b10) + (i & 0b1), Shape_N);
    }

    if (IN_BOUND_CD(load_offset)) {
      D[load_offset] = d[i];
    }
  }
}

template<int Shape_M, int Shape_N, int Shape_K>
__global__ void mma_kernel_m8n8k16_s8_s32(int8_t *A, int8_t *B, int *C, int *D, int M, int N, int K, int A_NUM_MAT, int B_NUM_MAT, int CD_NUM_MAT) {
  const int THREAD_IDX = threadIdx.x + blockIdx.x * blockDim.x;
  const int WARP_ID = THREAD_IDX / WARP_SIZE;
  const int LANE_ID = THREAD_IDX % WARP_SIZE;

  const int THREAD_ROW = LANE_ID / 4;
  const int THREAD_COL = LANE_ID % 4;

  int A_OFFSET = (WARP_ID % A_NUM_MAT) * Shape_M * Shape_K;
  int B_OFFSET = (WARP_ID % B_NUM_MAT) * Shape_K * Shape_N;
  int CD_OFFSET = (WARP_ID % CD_NUM_MAT) * Shape_M * Shape_N;

  int a;
  int b;
  int c[2];
  int d[2];

  auto ra = reinterpret_cast<int8_t *>(&a);
  auto rb = reinterpret_cast<int8_t *>(&b);

  for (int i = 0; i < 4; i++) {
    int load_offset = A_OFFSET + OFFSET(THREAD_ROW, (THREAD_COL * 4) + i, Shape_K);

    if (IN_BOUND_A(load_offset)) {
      ra[i] = A[load_offset];
    }
  }

  for (int i = 0; i < 4; i++) {
    int load_offset = B_OFFSET + OFFSET((THREAD_COL * 4) + i, THREAD_ROW, Shape_N);

    if (IN_BOUND_B(load_offset)) {
      rb[i] = B[load_offset];
    }
  }

  for (int i = 0; i < 2; i++) {
    int load_offset = CD_OFFSET + OFFSET(THREAD_ROW, (THREAD_COL * 2) + i, Shape_N);

    if (IN_BOUND_CD(load_offset)) {
      c[i] = C[load_offset];
    }
  }

  asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
      " { %0, %1 }, "
      " { %2 }, "
      " { %3 }, "
      " { %4, %5 };"
      : "=r"(d[0]), "=r"(d[1])
      : "r"(a),
        "r"(b),
        "r"(c[0]), "r"(c[1]));

  for (int i = 0; i < 2; i++) {
    int load_offset = CD_OFFSET + OFFSET(THREAD_ROW, (THREAD_COL * 2) + i, Shape_N);

    if (IN_BOUND_CD(load_offset)) {
      D[load_offset] = d[i];
    }
  }
}

template<int Shape_M, int Shape_N, int Shape_K>
__global__ void mma_kernel_m16n8k8_ptx_f16_f32(half *A, half *B, float *C, float *D, int M, int N, int K, int A_NUM_MAT, int B_NUM_MAT, int CD_NUM_MAT) {
  const int THREAD_IDX = threadIdx.x + blockIdx.x * blockDim.x;
  const int WARP_ID = THREAD_IDX / WARP_SIZE;
  const int LANE_ID = THREAD_IDX % WARP_SIZE;

  const int THREAD_ROW = LANE_ID / 4;
  const int THREAD_COL = LANE_ID % 4;

  int A_OFFSET = (WARP_ID % A_NUM_MAT) * Shape_M * Shape_K;
  int B_OFFSET = (WARP_ID % B_NUM_MAT) * Shape_K * Shape_N;
  int CD_OFFSET = (WARP_ID % CD_NUM_MAT) * Shape_M * Shape_N;

  half2 a[2];
  half2 b;
  float c[4];
  float d[4];

  auto ra = reinterpret_cast<half *>(a);
  auto rb = reinterpret_cast<half *>(&b);

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

  asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
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

template<int Shape_M, int Shape_N, int Shape_K>
__global__ void mma_kernel_m16n8k16_ptx_f16_f32(half *A, half *B, float *C, float *D, int M, int N, int K, int A_NUM_MAT, int B_NUM_MAT, int CD_NUM_MAT) {
  const int THREAD_IDX = threadIdx.x + blockIdx.x * blockDim.x;
  const int WARP_ID = THREAD_IDX / WARP_SIZE;
  const int LANE_ID = THREAD_IDX % WARP_SIZE;

  const int THREAD_ROW = LANE_ID / 4;
  const int THREAD_COL = LANE_ID % 4;

  int A_OFFSET = (WARP_ID % A_NUM_MAT) * Shape_M * Shape_K;
  int B_OFFSET = (WARP_ID % B_NUM_MAT) * Shape_K * Shape_N;
  int CD_OFFSET = (WARP_ID % CD_NUM_MAT) * Shape_M * Shape_N;

  half2 a[4];
  half2 b[2];
  float c[4];
  float d[4];

  auto ra = reinterpret_cast<half *>(a);
  auto rb = reinterpret_cast<half *>(b);

  for (int i = 0; i < 8; i++) {
    int r_off = 8;
    if (i < 2 || (i >= 4 && i < 6)) {
      r_off = 0;
    }

    int c_off = 0;
    if (i >= 4) {
      c_off = 8;
    }

    int load_offset = A_OFFSET + OFFSET(THREAD_ROW + r_off, (THREAD_COL * 2) + (i & 0x1) + c_off, Shape_K);
    if (IN_BOUND_A(load_offset)) {
      ra[i] = A[load_offset];
    }
  }

  for (int i = 0; i < 4; i++) {
    int r_off = 0;
    if (i >= 2) {
      r_off = 8;
    }

    int load_offset = B_OFFSET + OFFSET((THREAD_COL * 2) + (i & 0x1) + r_off, THREAD_ROW, Shape_N);
    if (IN_BOUND_B(load_offset)) {
      rb[i] = B[load_offset];
    }
  }

  for (int i = 0; i < 4; i++) {
    int load_offset;

    if (i < 2) {
      load_offset = CD_OFFSET + OFFSET(THREAD_ROW, (THREAD_COL * 2) + (i & 0x1), Shape_N);
    } else {
      load_offset = CD_OFFSET + OFFSET(THREAD_ROW + 8, (THREAD_COL * 2) + (i & 0x1), Shape_N);
    }

    if (IN_BOUND_CD(load_offset)) {
      c[i] = C[load_offset];
    }
  }

  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      " { %0, %1, %2, %3 }, "
      " { %4, %5, %6, %7 }, "
      " { %8, %9 }, "
      " { %10, %11, %12, %13 };"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(*(reinterpret_cast<int *>(&a[0]))),
        "r"(*(reinterpret_cast<int *>(&a[1]))),
        "r"(*(reinterpret_cast<int *>(&a[2]))),
        "r"(*(reinterpret_cast<int *>(&a[3]))),
        "r"(*(reinterpret_cast<int *>(&b[0]))),
        "r"(*(reinterpret_cast<int *>(&b[1]))),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));

  for (int i = 0; i < 4; i++) {
    int load_offset;

    if (i < 2) {
      load_offset = CD_OFFSET + OFFSET(THREAD_ROW, (THREAD_COL * 2) + (i & 0x1), Shape_N);
    } else {
      load_offset = CD_OFFSET + OFFSET(THREAD_ROW + 8, (THREAD_COL * 2) + (i & 0x1), Shape_N);
    }

    if (IN_BOUND_CD(load_offset)) {
      D[load_offset] = d[i];
    }
  }
}


template <int Shape_M, int Shape_N, int Shape_K>
__global__ void mma_kernel_m16n8k16_ptx_f16_f16(half *A, half *B, half *C,
                                                half *D, int M, int N, int K,
                                                int A_NUM_MAT, int B_NUM_MAT,
                                                int CD_NUM_MAT) {
  const int THREAD_IDX = threadIdx.x + blockIdx.x * blockDim.x;
  const int WARP_ID = THREAD_IDX / WARP_SIZE;
  const int LANE_ID = THREAD_IDX % WARP_SIZE;

  const int THREAD_ROW = LANE_ID / 4;
  const int THREAD_COL = LANE_ID % 4;

  int A_OFFSET = (WARP_ID % A_NUM_MAT) * Shape_M * Shape_K;
  int B_OFFSET = (WARP_ID % B_NUM_MAT) * Shape_K * Shape_N;
  int CD_OFFSET = (WARP_ID % CD_NUM_MAT) * Shape_M * Shape_N;

  uint32_t a[4];
  uint32_t b[2];
  uint32_t c[2];
  uint32_t d[2];

  auto ra = reinterpret_cast<half *>(a);
  auto rb = reinterpret_cast<half *>(b);
  auto rc = reinterpret_cast<half *>(c);
  auto rd = reinterpret_cast<half *>(d);

  for (int i = 0; i < 8; i++) {
    int r_off = 8;
    if (i < 2 || (i >= 4 && i < 6)) {
      r_off = 0;
    }

    int c_off = 0;
    if (i >= 4) {
      c_off = 8;
    }

    int load_offset =
        A_OFFSET + OFFSET(THREAD_ROW + r_off,
                          (THREAD_COL * 2) + (i & 0x1) + c_off, Shape_K);
    if (IN_BOUND_A(load_offset)) {
      ra[i] = A[load_offset];

    }
  }

  for (int i = 0; i < 4; i++) {
    int r_off = 0;
    if (i >= 2) {
      r_off = 8;
    }

    int load_offset = B_OFFSET + OFFSET((THREAD_COL * 2) + (i & 0x1) + r_off,
                                        THREAD_ROW, Shape_N);
    if (IN_BOUND_B(load_offset)) {
      rb[i] = B[load_offset];
    }
  }

  for (int i = 0; i < 4; i++) {
    int load_offset;

    if (i < 2) {
      load_offset =
          CD_OFFSET + OFFSET(THREAD_ROW, (THREAD_COL * 2) + (i & 0x1), Shape_N);
    } else {
      load_offset = CD_OFFSET + OFFSET(THREAD_ROW + 8,
                                       (THREAD_COL * 2) + (i & 0x1), Shape_N);
    }

    if (IN_BOUND_CD(load_offset)) {
      rc[i] = C[load_offset];
    }
  }


  asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
      " { %0, %1}, "
      " { %2, %3, %4, %5 }, "
      " { %6, %7 }, "
      " { %8, %9 };"
      : "+r"(d[0]), "+r"(d[1])
      : "r"(*(reinterpret_cast<int *>(&a[0]))),
        "r"(*(reinterpret_cast<int *>(&a[1]))),
        "r"(*(reinterpret_cast<int *>(&a[2]))),
        "r"(*(reinterpret_cast<int *>(&a[3]))),
        "r"(*(reinterpret_cast<int *>(&b[0]))),
        "r"(*(reinterpret_cast<int *>(&b[1]))),
        "r"(c[0]), "r"(c[1]));

  for (int i = 0; i < 4; i++) {
    int load_offset;

    if (i < 2) {
      load_offset =
          CD_OFFSET + OFFSET(THREAD_ROW, (THREAD_COL * 2) + (i & 0x1), Shape_N);
    } else {
      load_offset = CD_OFFSET + OFFSET(THREAD_ROW + 8,
                                       (THREAD_COL * 2) + (i & 0x1), Shape_N);
    }

    if (IN_BOUND_CD(load_offset)) {
      D[load_offset] = rd[i];
    }
  }
}


template <int Shape_M, int Shape_N, int Shape_K>
__global__ void mma_kernel_m16n8k16_ptx_bf16_f32(__nv_bfloat16 *A, __nv_bfloat16 *B, float *C, float *D, int M, int N, int K, int A_NUM_MAT, int B_NUM_MAT, int CD_NUM_MAT) {
  const int THREAD_IDX = threadIdx.x + blockIdx.x * blockDim.x;
  const int WARP_ID = THREAD_IDX / WARP_SIZE;
  const int LANE_ID = THREAD_IDX % WARP_SIZE;

  const int THREAD_ROW = LANE_ID / 4;
  const int THREAD_COL = LANE_ID % 4;

  int A_OFFSET = (WARP_ID % A_NUM_MAT) * Shape_M * Shape_K;
  int B_OFFSET = (WARP_ID % B_NUM_MAT) * Shape_K * Shape_N;
  int CD_OFFSET = (WARP_ID % CD_NUM_MAT) * Shape_M * Shape_N;

  uint32_t a[4];
  uint32_t b[2];
  float c[4];
  float d[4];

  unsigned char indx = 0;
  for (unsigned char i = 0; i < 8; i++) {
    int r_off = 8;
    if (i < 2 || (i >= 4 && i < 6)) {
      r_off = 0;
    }

    int c_off = 0;
    if (i >= 4) {
      c_off = 8;
    }

    int load_offset =
        A_OFFSET + OFFSET(THREAD_ROW + r_off,
                          (THREAD_COL * 2) + (i & 0x1) + c_off, Shape_K);
    if (IN_BOUND_A(load_offset)) {
      __nv_bfloat16 val = A[load_offset];
      if ((i & 0x01) == 0) {
        // First value of pair, put to high
        a[indx] = uint32_t(*reinterpret_cast<uint16_t *>(&val) << 16);
      } else {
        // Second value of pair, put to low
        a[indx] = a[indx] | *reinterpret_cast<uint16_t *>(&val);
        indx++;
      }
    }
  }

  indx = 0;
  for (int i = 0; i < 4; i++) {
    int r_off = 0;
    if (i >= 2) {
      r_off = 8;
    }

    int load_offset = B_OFFSET + OFFSET((THREAD_COL * 2) + (i & 0x1) + r_off,
                                        THREAD_ROW, Shape_N);
    if (IN_BOUND_B(load_offset)) {
      __nv_bfloat16 val = B[load_offset];
      if ((i & 0x01) == 0) {
        // First value of pair, put to high
        b[indx] = uint32_t(*reinterpret_cast<uint16_t *>(&val) << 16);
      } else {
        // Second value of pair, put to low
        b[indx] = b[indx] | *reinterpret_cast<uint16_t *>(&val);
        indx++;
      }
    }
  }

  for (int i = 0; i < 4; i++) {
    int load_offset;

    if (i < 2) {
      load_offset =
          CD_OFFSET + OFFSET(THREAD_ROW, (THREAD_COL * 2) + (i & 0x1), Shape_N);
    } else {
      load_offset = CD_OFFSET + OFFSET(THREAD_ROW + 8,
                                       (THREAD_COL * 2) + (i & 0x1), Shape_N);
    }

    if (IN_BOUND_CD(load_offset)) {
      c[i] = C[load_offset];
    }
  }

  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      " { %0, %1, %2, %3 }, "
      " { %4, %5, %6, %7 }, "
      " { %8, %9 }, "
      " { %10, %11, %12, %13 };"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
        "r"(b[0]), "r"(b[1]), 
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));


  for (int i = 0; i < 4; i++) {
    int load_offset;

    if (i < 2) {
      load_offset =
          CD_OFFSET + OFFSET(THREAD_ROW, (THREAD_COL * 2) + (i & 0x1), Shape_N);
    } else {
      load_offset = CD_OFFSET + OFFSET(THREAD_ROW + 8,
                                       (THREAD_COL * 2) + (i & 0x1), Shape_N);
    }

    if (IN_BOUND_CD(load_offset)) {
      D[load_offset] = d[i];
    }
  }
}

template<int Shape_M, int Shape_N, int Shape_K>
__global__ void mma_kernel_m16n8k16_s8_s32(int8_t *A, int8_t *B, int *C, int *D, int M, int N, int K, int A_NUM_MAT, int B_NUM_MAT, int CD_NUM_MAT) {
  const int THREAD_IDX = threadIdx.x + blockIdx.x * blockDim.x;
  const int WARP_ID = THREAD_IDX / WARP_SIZE;
  const int LANE_ID = THREAD_IDX % WARP_SIZE;

  const int THREAD_ROW = LANE_ID / 4;
  const int THREAD_COL = LANE_ID % 4;

  int A_OFFSET = (WARP_ID % A_NUM_MAT) * Shape_M * Shape_K;
  int B_OFFSET = (WARP_ID % B_NUM_MAT) * Shape_K * Shape_N;
  int CD_OFFSET = (WARP_ID % CD_NUM_MAT) * Shape_M * Shape_N;

  int a[2];
  int b;
  int c[4];
  int d[4];

  auto ra = reinterpret_cast<int8_t *>(a);
  auto rb = reinterpret_cast<int8_t *>(&b);

  for (int i = 0; i < 8; i++) {
    int r_off = 0;
    if (i >= 4) {
      r_off = 8;
    }

    int load_offset = A_OFFSET + OFFSET(THREAD_ROW + r_off, (THREAD_COL * 4) + (i & 0x3), Shape_K);
    if (IN_BOUND_A(load_offset)) {
      ra[i] = A[load_offset];
    }
  }

  for (int i = 0; i < 4; i++) {
    int load_offset = B_OFFSET + OFFSET((THREAD_COL * 4) + i, THREAD_ROW, Shape_N);

    if (IN_BOUND_B(load_offset)) {
      rb[i] = B[load_offset];
    }
  }

  for (int i = 0; i < 4; i++) {
    int load_offset;

    if (i < 2) {
      load_offset = CD_OFFSET + OFFSET(THREAD_ROW, (THREAD_COL * 2) + (i & 0x1), Shape_N);
    } else {
      load_offset = CD_OFFSET + OFFSET(THREAD_ROW + 8, (THREAD_COL * 2) + (i & 0x1), Shape_N);
    }

    if (IN_BOUND_CD(load_offset)) {
      c[i] = C[load_offset];
    }
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
    int load_offset;

    if (i < 2) {
      load_offset = CD_OFFSET + OFFSET(THREAD_ROW, (THREAD_COL * 2) + (i & 0x1), Shape_N);
    } else {
      load_offset = CD_OFFSET + OFFSET(THREAD_ROW + 8, (THREAD_COL * 2) + (i & 0x1), Shape_N);
    }

    if (IN_BOUND_CD(load_offset)) {
      D[load_offset] = d[i];
    }
  }
}

template<int Shape_M, int Shape_N, int Shape_K>
__global__ void mma_kernel_m16n8k32_s8_s32(int8_t *A, int8_t *B, int *C, int *D, int M, int N, int K, int A_NUM_MAT, int B_NUM_MAT, int CD_NUM_MAT) {
  const int THREAD_IDX = threadIdx.x + blockIdx.x * blockDim.x;
  const int WARP_ID = THREAD_IDX / WARP_SIZE;
  const int LANE_ID = THREAD_IDX % WARP_SIZE;

  const int THREAD_ROW = LANE_ID / 4;
  const int THREAD_COL = LANE_ID % 4;

  int A_OFFSET = (WARP_ID % A_NUM_MAT) * Shape_M * Shape_K;
  int B_OFFSET = (WARP_ID % B_NUM_MAT) * Shape_K * Shape_N;
  int CD_OFFSET = (WARP_ID % CD_NUM_MAT) * Shape_M * Shape_N;

  int a[4];
  int b[2];
  int c[4];
  int d[4];

  auto ra = reinterpret_cast<int8_t *>(a);
  auto rb = reinterpret_cast<int8_t *>(b);

  for (int i = 0; i < 16; i++) {
    int r_off = 0;
    if (i < 4 || (i >= 8 && i < 12)) {
      r_off = 8;
    }

    int c_off = 0;
    if (i >= 8) {
      c_off = 16;
    }

    int load_offset = A_OFFSET + OFFSET(THREAD_ROW + r_off, (THREAD_COL * 4) + (i & 0x3) + c_off, Shape_K);
    if (IN_BOUND_A(load_offset)) {
      ra[i] = A[load_offset];
    }
  }

  for (int i = 0; i < 8; i++) {
    int r_off = 0;
    if (i >= 4) {
      r_off = 16;
    }

    int load_offset = B_OFFSET + OFFSET((THREAD_COL * 4) + (i & 0x3) + r_off, THREAD_ROW, Shape_N);
    if (IN_BOUND_B(load_offset)) {
      rb[i] = B[load_offset];
    }
  }

  for (int i = 0; i < 4; i++) {
    int load_offset;

    if (i < 2)
      load_offset = CD_OFFSET + OFFSET(THREAD_ROW, (THREAD_COL * 2) + (i & 0x1), Shape_N);
    else
      load_offset = CD_OFFSET + OFFSET(THREAD_ROW + 8, (THREAD_COL * 2) + (i & 0x1), Shape_N);
    
    if (IN_BOUND_CD(load_offset)) {
      c[i] = C[load_offset];
    }
  }

  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      " { %0, %1, %2, %3 }, "
      " { %4, %5, %6, %7 }, "
      " { %8, %9 }, "
      " { %10, %11, %12, %13 };"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
        "r"(b[0]), "r"(b[1]),
        "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));

  for (int i = 0; i < 4; i++) {
    int load_offset;

    if (i < 2)
      load_offset = CD_OFFSET + OFFSET(THREAD_ROW, (THREAD_COL * 2) + (i & 0x1), Shape_N);
    else
      load_offset = CD_OFFSET + OFFSET(THREAD_ROW + 8, (THREAD_COL * 2) + (i & 0x1), Shape_N);
    
    if (IN_BOUND_CD(load_offset)) {
      D[load_offset] = d[i];
    }
  }
}

bool run_test_mma_m8n8k4_f16_f32(const int M, const int N, const int K) {
  int A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT;
  calculate_num_matrices<8, 8, 4>(M, N, K, A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT);

  if (A_NUM_MAT == 0 || B_NUM_MAT == 0 || CD_NUM_MAT == 0) {
    std::cerr << "Matrix dimensions are not compatible with m8n8k4" << std::endl;
    return false;
  }

  half *d_A, *d_B;
  float *d_C, *d_D;
  half h_A[M * K], h_B[K * N];
  float h_C[M * N], h_D[M * N];
  float h_D_ref[M * N];

  initialize_matrices(h_A, h_B, h_C, h_D, 8, 8, 4, A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT);

  matrix_multiplication_cpu(h_A, h_B, h_C, h_D_ref, 8, 8, 4, A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT);

  cudaMalloc(&d_A, M * K * sizeof(half));
  cudaMalloc(&d_B, K * N * sizeof(half));
  cudaMalloc(&d_C, M * N * sizeof(float));
  cudaMalloc(&d_D, M * N * sizeof(float));

  cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);
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

  mma_kernel_m8n8k4_ptx_f16_f32<8, 8, 4><<<no_blocks, no_threads>>>(d_A, d_B, d_C, d_D, M, N, K, A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT);
  cudaDeviceSynchronize();
  cudaMemcpy(h_D, d_D, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  bool correct = check_result(M, N, h_D, h_D_ref);

  std::cout << "m8n8k4 (f32.f16.f16.f32): " << (correct ? "PASSED" : "FAILED") << std::endl;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_D);

  return correct;
}

bool run_test_mma_m8n8k16_s8_s32(const int M, const int N, const int K) {
  int A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT;
  calculate_num_matrices<8, 8, 16>(M, N, K, A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT);

  if (A_NUM_MAT == 0 || B_NUM_MAT == 0 || CD_NUM_MAT == 0) {
    std::cerr << "Matrix dimensions are not compatible with m8n8k16" << std::endl;
    return false;
  }

  int8_t *d_A, *d_B;
  int *d_C, *d_D;
  int8_t h_A[M * K], h_B[K * N];
  int h_C[M * N], h_D[M * N];
  int h_D_ref[M * N];

  initialize_matrices(h_A, h_B, h_C, h_D, 8, 8, 16, A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT);

  matrix_multiplication_cpu(h_A, h_B, h_C, h_D_ref, 8, 8, 16, A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT);

  cudaMalloc(&d_A, M * K * sizeof(int8_t));
  cudaMalloc(&d_B, K * N * sizeof(int8_t));
  cudaMalloc(&d_C, M * N * sizeof(int));
  cudaMalloc(&d_D, M * N * sizeof(int));

  cudaMemcpy(d_A, h_A, M * K * sizeof(int8_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, K * N * sizeof(int8_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, h_C, M * N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_D, h_D, M * N * sizeof(int), cudaMemcpyHostToDevice);

  int no_mat_blocks = 4;
  int no_blocks = CD_NUM_MAT / no_mat_blocks;
  int no_threads;
  if (no_blocks) {
    no_threads = WARP_SIZE * no_mat_blocks;
  } else {
    no_blocks = 1;
    no_threads = WARP_SIZE * CD_NUM_MAT;
  }

  mma_kernel_m8n8k16_s8_s32<8, 8, 16><<<no_blocks, no_threads>>>(d_A, d_B, d_C, d_D, M, N, K, A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT);
  cudaDeviceSynchronize();
  cudaMemcpy(h_D, d_D, M * N * sizeof(int), cudaMemcpyDeviceToHost);

  bool correct = check_result(M, N, h_D, h_D_ref);

  std::cout << "m8n8k16 (s32.s8.s8.s32): " << (correct ? "PASSED" : "FAILED") << std::endl;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_D);

  return correct;
}

bool run_test_mma_m16n8k8_f16_f32(const int M, const int N, const int K) {
  int A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT;
  calculate_num_matrices<16, 8, 8>(M, N, K, A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT);

  if (A_NUM_MAT == 0 || B_NUM_MAT == 0 || CD_NUM_MAT == 0) {
    std::cerr << "Matrix dimensions are not compatible with m16n8k8" << std::endl;
    return false;
  }

  half *d_A, *d_B;
  float *d_C, *d_D;
  half h_A[M * K], h_B[K * N];
  float h_C[M * N], h_D[M * N];
  float h_D_ref[M * N];

  initialize_matrices(h_A, h_B, h_C, h_D, 16, 8, 8, A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT);

  matrix_multiplication_cpu(h_A, h_B, h_C, h_D_ref, 16, 8, 8, A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT);

  cudaMalloc(&d_A, M * K * sizeof(half));
  cudaMalloc(&d_B, K * N * sizeof(half));
  cudaMalloc(&d_C, M * N * sizeof(float));
  cudaMalloc(&d_D, M * N * sizeof(float));

  cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);
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

  mma_kernel_m16n8k8_ptx_f16_f32<16, 8, 8><<<no_blocks, no_threads>>>(d_A, d_B, d_C, d_D, M, N, K, A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT);
  cudaDeviceSynchronize();
  cudaMemcpy(h_D, d_D, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  bool correct = check_result(M, N, h_D, h_D_ref);

  std::cout << "m16n8k8 (f32.f16.f16.f32): " << (correct ? "PASSED" : "FAILED") << std::endl;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_D);

  return correct;
}

bool run_test_mma_m16n8k16_f16_f32(const int M, const int N, const int K) {
  int A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT;
  calculate_num_matrices<16, 8, 16>(M, N, K, A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT);

  if (A_NUM_MAT == 0 || B_NUM_MAT == 0 || CD_NUM_MAT == 0) {
    std::cerr << "Matrix dimensions are not compatible with m16n8k16" << std::endl;
    return false;
  }

  half *d_A, *d_B;
  float *d_C, *d_D;
  half h_A[M * K], h_B[K * N];
  float h_C[M * N], h_D[M * N];
  float h_D_ref[M * N];

  initialize_matrices(h_A, h_B, h_C, h_D, 16, 8, 16, A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT);

  matrix_multiplication_cpu(h_A, h_B, h_C, h_D_ref, 16, 8, 16, A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT);

  cudaMalloc(&d_A, M * K * sizeof(half));
  cudaMalloc(&d_B, K * N * sizeof(half));
  cudaMalloc(&d_C, M * N * sizeof(float));
  cudaMalloc(&d_D, M * N * sizeof(float));

  cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);
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

  mma_kernel_m16n8k16_ptx_f16_f32<16, 8, 16><<<no_blocks, no_threads>>>(d_A, d_B, d_C, d_D, M, N, K, A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT);
  cudaDeviceSynchronize();
  cudaMemcpy(h_D, d_D, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  bool correct = check_result(M, N, h_D, h_D_ref);

  std::cout << "m16n8k16 (f32.f16.f16.f32): " << (correct ? "PASSED" : "FAILED") << std::endl;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_D);

  return correct;
}

bool run_test_mma_m16n8k16_f16_f16(const int M, const int N, const int K) {
  int A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT;
  calculate_num_matrices<16, 8, 16>(M, N, K, A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT);

  if (A_NUM_MAT == 0 || B_NUM_MAT == 0 || CD_NUM_MAT == 0) {
    std::cerr << "Matrix dimensions are not compatible with m16n8k16" << std::endl;
    return false;
  }

  half *d_A, *d_B;
  half *d_C, *d_D;
  half h_A[M * K], h_B[K * N];
  half h_C[M * N], h_D[M * N], h_D_ref[M * N];

  initialize_matrices(h_A, h_B, h_C, h_D, 16, 8, 16, A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT);

  matrix_multiplication_cpu(h_A, h_B, h_C, h_D_ref, 16, 8, 16, A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT);

  cudaMalloc(&d_A, M * K * sizeof(half));
  cudaMalloc(&d_B, K * N * sizeof(half));
  cudaMalloc(&d_C, M * N * sizeof(half));
  cudaMalloc(&d_D, M * N * sizeof(half));

  cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, h_C, M * N * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_D, h_D, M * N * sizeof(half), cudaMemcpyHostToDevice);

  int no_mat_blocks = 4;
  int no_blocks = CD_NUM_MAT / no_mat_blocks;
  int no_threads;
  if (no_blocks) {
    no_threads = WARP_SIZE * no_mat_blocks;
  } else {
    no_blocks = 1;
    no_threads = WARP_SIZE * CD_NUM_MAT;
  }

  mma_kernel_m16n8k16_ptx_f16_f16<16, 8, 16><<<no_blocks, no_threads>>>(d_A, d_B, d_C, d_D, M, N, K, A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT);
  cudaDeviceSynchronize();
  cudaMemcpy(h_D, d_D, M * N * sizeof(half), cudaMemcpyDeviceToHost);

  bool correct = check_result(M, N, h_D, h_D_ref);

  std::cout << "m16n8k16 (f16.f16.f16.f16): " << (correct ? "PASSED" : "FAILED") << std::endl;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_D);

  return correct;
}


bool run_test_mma_m16n8k16_bf16_f32(const int M, const int N, const int K) {
  int A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT;
  calculate_num_matrices<16, 8, 16>(M, N, K, A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT);

  if (A_NUM_MAT == 0 || B_NUM_MAT == 0 || CD_NUM_MAT == 0) {
    std::cerr << "Matrix dimensions are not compatible with m16n8k16"
              << std::endl;
    return false;
  }

  __nv_bfloat16 *d_A, *d_B;
  float *d_C, *d_D;
  __nv_bfloat16 h_A[M * K], h_B[K * N];
  float h_C[M * N], h_D[M * N];
  float h_D_ref[M * N];

  initialize_matrices(h_A, h_B, h_C, h_D, 16, 8, 16, A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT);

  matrix_multiplication_cpu(h_A, h_B, h_C, h_D_ref, 16, 8, 16, A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT);

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

  mma_kernel_m16n8k16_ptx_bf16_f32<16, 8, 16><<<no_blocks, no_threads>>>(
      d_A, d_B, d_C, d_D, M, N, K, A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT);
  cudaDeviceSynchronize();
  cudaMemcpy(h_D, d_D, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  bool correct = check_result(M, N, h_D, h_D_ref);

  std::cout << "m16n8k16 (f32.bf16.bf16.f32): " << (correct ? "PASSED" : "FAILED")
            << std::endl;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_D);

  return correct;
}

bool run_test_mma_m16n8k16_s8_s32(const int M, const int N, const int K) {
  int A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT;
  calculate_num_matrices<16, 8, 16>(M, N, K, A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT);

  if (A_NUM_MAT == 0 || B_NUM_MAT == 0 || CD_NUM_MAT == 0) {
    std::cerr << "Matrix dimensions are not compatible with m16n8k16" << std::endl;
    return false;
  }

  int8_t *d_A, *d_B;
  int *d_C, *d_D;
  int8_t h_A[M * K], h_B[K * N];
  int h_C[M * N], h_D[M * N];
  int h_D_ref[M * N];

  initialize_matrices(h_A, h_B, h_C, h_D, 16, 8, 16, A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT);

  matrix_multiplication_cpu(h_A, h_B, h_C, h_D_ref, 16, 8, 16, A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT);

  cudaMalloc(&d_A, M * K * sizeof(int8_t));
  cudaMalloc(&d_B, K * N * sizeof(int8_t));
  cudaMalloc(&d_C, M * N * sizeof(int));
  cudaMalloc(&d_D, M * N * sizeof(int));

  cudaMemcpy(d_A, h_A, M * K * sizeof(int8_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, K * N * sizeof(int8_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, h_C, M * N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_D, h_D, M * N * sizeof(int), cudaMemcpyHostToDevice);

  int no_mat_blocks = 4;
  int no_blocks = CD_NUM_MAT / no_mat_blocks;
  int no_threads;
  if (no_blocks) {
    no_threads = WARP_SIZE * no_mat_blocks;
  } else {
    no_blocks = 1;
    no_threads = WARP_SIZE * CD_NUM_MAT;
  }

  mma_kernel_m16n8k16_s8_s32<16, 8, 16><<<no_blocks, no_threads>>>(d_A, d_B, d_C, d_D, M, N, K, A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT);
  cudaDeviceSynchronize();
  cudaMemcpy(h_D, d_D, M * N * sizeof(int), cudaMemcpyDeviceToHost);

  bool correct = check_result(M, N, h_D, h_D_ref);

  std::cout << "m16n8k16 (s32.s8.s8.s32): " << (correct ? "PASSED" : "FAILED") << std::endl;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_D);

  return correct;
}

bool run_test_mma_m16n8k32_s8_s32(const int M, const int N, const int K) {
  int A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT;
  calculate_num_matrices<16, 8, 32>(M, N, K, A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT);

  if (A_NUM_MAT == 0 || B_NUM_MAT == 0 || CD_NUM_MAT == 0) {
    std::cerr << "Matrix dimensions are not compatible with m16n8k16" << std::endl;
    return false;
  }

  int8_t *d_A, *d_B;
  int *d_C, *d_D;
  int8_t h_A[M * K], h_B[K * N];
  int h_C[M * N], h_D[M * N];
  int h_D_ref[M * N];

  initialize_matrices(h_A, h_B, h_C, h_D, 16, 8, 32, A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT);

  matrix_multiplication_cpu(h_A, h_B, h_C, h_D_ref, 16, 8, 32, A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT);

  cudaMalloc(&d_A, M * K * sizeof(int8_t));
  cudaMalloc(&d_B, K * N * sizeof(int8_t));
  cudaMalloc(&d_C, M * N * sizeof(int));
  cudaMalloc(&d_D, M * N * sizeof(int));

  cudaMemcpy(d_A, h_A, M * K * sizeof(int8_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, K * N * sizeof(int8_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, h_C, M * N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_D, h_D, M * N * sizeof(int), cudaMemcpyHostToDevice);

  int no_mat_blocks = 4;
  int no_blocks = CD_NUM_MAT / no_mat_blocks;
  int no_threads;
  if (no_blocks) {
    no_threads = WARP_SIZE * no_mat_blocks;
  } else {
    no_blocks = 1;
    no_threads = WARP_SIZE * CD_NUM_MAT;
  }

  mma_kernel_m16n8k32_s8_s32<16, 8, 32><<<no_blocks, no_threads>>>(d_A, d_B, d_C, d_D, M, N, K, A_NUM_MAT, B_NUM_MAT, CD_NUM_MAT);
  cudaDeviceSynchronize();
  cudaMemcpy(h_D, d_D, M * N * sizeof(int), cudaMemcpyDeviceToHost);

  bool correct = check_result(M, N, h_D, h_D_ref);

  std::cout << "m16n8k32 (s32.s8.s8.s32): " << (correct ? "PASSED" : "FAILED") << std::endl;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_D);

  return correct;
}

bool launch_test_mma_m8n8k4_f16_f32(const int M, const int N, const int K) {
  bool correct = run_test_mma_m8n8k4_f16_f32(M, N, K);

  if (!correct) {
    std::cerr << "m8n8k4 (f32.f16.f16.f32) failed for dims: " << M << ", " << N << ", " << K << std::endl;
    return false;
  }

  return true;
}

bool launch_test_mma_m8n8k16_s8_s32(const int M, const int N, const int K) {
  bool correct = run_test_mma_m8n8k16_s8_s32(M, N, K);

  if (!correct) {
    std::cerr << "m8n8k16 (s32.s8.s8.s32) failed for dims: " << M << ", " << N << ", " << K << std::endl;
    return false;
  }

  return true;
}

bool launch_test_mma_m16n8k8_f16_f32(const int M, const int N, const int K) {
  bool correct = run_test_mma_m16n8k8_f16_f32(M, N, K);

  if (!correct) {
    std::cerr << "m16n8k8 (f32.f16.f16.f32) failed for dims: " << M << ", " << N << ", " << K << std::endl;
    return false;
  }

  return true;
}

bool launch_test_mma_m16n8k16_f16_f16(const int M, const int N, const int K) {
  bool correct = run_test_mma_m16n8k16_f16_f16(M, N, K);

  if (!correct) {
    std::cerr << "m16n8k16 (f16.f16.f16.f16) failed for dims: " << M << ", " << N
              << ", " << K << std::endl;
    return false;
  }

  return true;
}

bool launch_test_mma_m16n8k16_f16_f32(const int M, const int N, const int K) {
  bool correct = run_test_mma_m16n8k16_f16_f32(M, N, K);

  if (!correct) {
    std::cerr << "m16n8k16 (f32.f16.f16.f32) failed for dims: " << M << ", " << N << ", " << K << std::endl;
    return false;
  }

  return true;
}

bool launch_test_mma_m16n8k16_bf16_f32(const int M, const int N, const int K) {
  bool correct = run_test_mma_m16n8k16_bf16_f32(M, N, K);

  if (!correct) {
    std::cerr << "m16n8k16 (f32.bf16.bf16.f32) failed for dims: " << M << ", "
              << N << ", " << K << std::endl;
    return false;
  }

  return true;
}

bool launch_test_mma_m16n8k16_s8_s32(const int M, const int N, const int K) {
  bool correct = run_test_mma_m16n8k16_s8_s32(M, N, K);

  if (!correct) {
    std::cerr << "m16n8k16 (s32.s8.s8.s32) failed for dims: " << M << ", " << N << ", " << K << std::endl;
    return false;
  }

  return true;
}

bool launch_test_mma_m16n8k32_s8_s32(const int M, const int N, const int K) {
  bool correct = run_test_mma_m16n8k32_s8_s32(M, N, K);

  if (!correct) {
    std::cerr << "m16n8k32 (s32.s8.s8.s32) failed for dims: " << M << ", " << N << ", " << K << std::endl;
    return false;
  }

  return true;
}

bool mma_m8n8k4_f16_f32() {
  LAUNCH_TEST(mma_m8n8k4_f16_f32(8, 8, 4));
  LAUNCH_TEST(mma_m8n8k4_f16_f32(16, 16, 8));
  LAUNCH_TEST(mma_m8n8k4_f16_f32(8, 16, 4));
  LAUNCH_TEST(mma_m8n8k4_f16_f32(8, 16, 8));
  LAUNCH_TEST(mma_m8n8k4_f16_f32(32, 32, 32));

  return true;
}

bool mma_m8n8k16_s8_s32() {
  LAUNCH_TEST(mma_m8n8k16_s8_s32(8, 8, 16));
  LAUNCH_TEST(mma_m8n8k16_s8_s32(16, 16, 32));
  LAUNCH_TEST(mma_m8n8k16_s8_s32(8, 16, 16));
  LAUNCH_TEST(mma_m8n8k16_s8_s32(8, 16, 32));
  LAUNCH_TEST(mma_m8n8k16_s8_s32(32, 32, 32));

  return true;
}

bool mma_m16n8k8_f16_f32() {
  LAUNCH_TEST(mma_m16n8k8_f16_f32(16, 8, 8));
  LAUNCH_TEST(mma_m16n8k8_f16_f32(32, 16, 16));
  LAUNCH_TEST(mma_m16n8k8_f16_f32(16, 16, 8));
  LAUNCH_TEST(mma_m16n8k8_f16_f32(16, 16, 16));
  LAUNCH_TEST(mma_m16n8k8_f16_f32(32, 32, 32));

  return true;
}

bool mma_m16n8k16_f16_f32() {
  LAUNCH_TEST(mma_m16n8k16_f16_f32(16, 8, 16));
  LAUNCH_TEST(mma_m16n8k16_f16_f32(32, 16, 32));
  LAUNCH_TEST(mma_m16n8k16_f16_f32(16, 16, 16));
  LAUNCH_TEST(mma_m16n8k16_f16_f32(16, 16, 32));
  LAUNCH_TEST(mma_m16n8k16_f16_f32(32, 32, 32));

  return true;
}

bool mma_m16n8k16_f16_f16() {
  LAUNCH_TEST(mma_m16n8k16_f16_f16(16, 8, 16));
  LAUNCH_TEST(mma_m16n8k16_f16_f16(32, 16, 32));
  LAUNCH_TEST(mma_m16n8k16_f16_f16(16, 16, 16));
  LAUNCH_TEST(mma_m16n8k16_f16_f16(16, 16, 32));
  LAUNCH_TEST(mma_m16n8k16_f16_f16(32, 32, 32));

  return true;
}

bool mma_m16n8k16_bf16_f32() {
  LAUNCH_TEST(mma_m16n8k16_bf16_f32(16, 8, 16));
  LAUNCH_TEST(mma_m16n8k16_bf16_f32(32, 16, 32));
  LAUNCH_TEST(mma_m16n8k16_bf16_f32(16, 16, 16));
  LAUNCH_TEST(mma_m16n8k16_bf16_f32(16, 16, 32));
  LAUNCH_TEST(mma_m16n8k16_bf16_f32(32, 32, 32));

  return true;
}

bool mma_m16n8k16_s8_s32() {
  LAUNCH_TEST(mma_m16n8k16_s8_s32(16, 8, 16));
  LAUNCH_TEST(mma_m16n8k16_s8_s32(32, 16, 32));
  LAUNCH_TEST(mma_m16n8k16_s8_s32(16, 16, 16));
  LAUNCH_TEST(mma_m16n8k16_s8_s32(16, 16, 32));
  LAUNCH_TEST(mma_m16n8k16_s8_s32(32, 32, 32));

  return true;
}

bool mma_m16n8k32_s8_s32() {
  LAUNCH_TEST(mma_m16n8k32_s8_s32(16, 8, 32));
  LAUNCH_TEST(mma_m16n8k32_s8_s32(32, 16, 64));
  LAUNCH_TEST(mma_m16n8k32_s8_s32(16, 16, 32));
  LAUNCH_TEST(mma_m16n8k32_s8_s32(16, 16, 64));
  LAUNCH_TEST(mma_m16n8k32_s8_s32(32, 32, 32));

  return true;
}

int main() {
  TEST(mma_m8n8k4_f16_f32);
  TEST(mma_m8n8k16_s8_s32);
  TEST(mma_m16n8k8_f16_f32);
  TEST(mma_m16n8k16_f16_f32);
  TEST(mma_m16n8k16_bf16_f32);
  TEST(mma_m16n8k16_s8_s32);
  TEST(mma_m16n8k32_s8_s32);
  TEST(mma_m16n8k16_f16_f16);
  return 0;
}
