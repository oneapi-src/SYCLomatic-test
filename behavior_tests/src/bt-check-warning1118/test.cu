// ====------ test.cu ------------------------------------- *- CUDA -* ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

__device__ void test1(int *a) {
  unsigned tid =
      ((blockIdx.x + (blockIdx.y * gridDim.x)) * (blockDim.x * blockDim.y)) +
      (threadIdx.x + (threadIdx.y * blockDim.x));

  __syncthreads();

  switch (tid) {
  case 0:
    a[0] = 1;
    __syncthreads();
    break;
  case 1:
    a[1] = 1;
    __syncthreads();
    break;
  default:
    a[tid] = a[tid - 1] + a[tid - 2];
  }

  if (tid > 32) {
    __syncthreads();
  } else {
    a[tid] = 0;
    __syncthreads();
  }

  // do
  do {
    a[tid] = a[tid - 1] + a[tid - 2];
    __syncthreads();
  } while (a[tid]);

  // while
  while (a[tid])
    __syncthreads();

  // early return
  if (tid < 32)
    return;

  __syncthreads();
  a[tid] = a[tid - 1] + a[tid - 2];
}

__device__ void test2(int *a) {
  if (threadIdx.x > 10)
    test1(a);

  if (threadIdx.x < 5)
    return;

  test1(a);
}

constexpr int const_expr() {
  return 10;
}

__device__ void test3(int *a) {

  test2(a);

  if (threadIdx.x > 10)
    test2(a);

  if (threadIdx.x < 5)
    return;
  test2(a);

  for (int i = 0; i < 10; ++i) {
    test2(a);
  }

  do {
    test2(a);
  } while (*a < 10);
}
