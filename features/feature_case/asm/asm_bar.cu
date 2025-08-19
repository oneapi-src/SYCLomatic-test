// ====------ asm_bar.cu ----------------------------------- *- CUDA -* ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <cuda_runtime.h>

__global__ void bar_warp_sync(int *arr, int *brr) {
  arr[threadIdx.x] = threadIdx.x + 10;
  if (threadIdx.x % 2 == 0) {
    for (int i = 0; i < 1000; ++i)
      arr[threadIdx.x] += arr[threadIdx.x] - 1 * arr[threadIdx.x] - 3;
    if (arr[threadIdx.x] < 0)
      arr[threadIdx.x] = 0;
  }

  asm volatile("bar.warp.sync %0;" ::"r"(0b1010101010));
  if (threadIdx.x == 1) {
    for (int i = 0; i < 10; ++i) {
      brr[i] = arr[i];
    }
  }
}

int test_bar_warp_sync() {
  int *arr, *brr;
  cudaMallocManaged(&arr, sizeof(int) * 10);
  cudaMemset(arr, 0, sizeof(int) * 10);
  cudaMallocManaged(&brr, sizeof(int) * 10);
  cudaMemset(brr, 0, sizeof(int) * 10);

  bar_warp_sync<<<1, 10>>>(arr, brr);
  cudaDeviceSynchronize();
  cudaFree(arr);
  int res = 0;
  for (int i = 1; i < 10; i += 2)
    if (brr[i] != i + 10 || brr[i - 1] != 0)
      res = 1;
  cudaFree(brr);
  return res;
}

constexpr unsigned int WIDTH = 10;
constexpr unsigned int HEIGHT = 32 * 4;

__global__ void bar_sync_bar_arrive(int *arr, int *results) {
  __shared__ int buffer[HEIGHT / 2];
  int value;

  // Initialize provider threads
  if (threadIdx.x < HEIGHT / 2)
    value = arr[threadIdx.x * WIDTH] + arr[(threadIdx.x + HEIGHT / 2) * WIDTH];
  else
    value = 0;

  for (unsigned int i = 1; i < WIDTH + 1; ++i) {
    if (threadIdx.x < HEIGHT / 2) {
      // provider
      buffer[threadIdx.x] = value;
      asm volatile("bar.arrive 1, %0;" ::"r"(HEIGHT));
      value = arr[threadIdx.x * WIDTH + i] +
              arr[(threadIdx.x + HEIGHT / 2) * WIDTH + i];
      asm volatile("bar.sync 2, %0;" ::"r"(HEIGHT));
    } else {
      // consumer
      asm volatile("bar.sync 1, %0;" ::"r"(HEIGHT));
      value = buffer[threadIdx.x - HEIGHT / 2];
      asm volatile("bar.arrive 2, %0;" ::"r"(HEIGHT));
      results[threadIdx.x - HEIGHT / 2] += value;
    }
  }
}

int test_bar_sync_bar_arrive() {
  int *arr, *results;

  cudaMallocManaged(&arr, sizeof(int) * WIDTH * HEIGHT);
  for (unsigned int i = 0; i < WIDTH * HEIGHT; ++i) {
    arr[i] = i;
  }

  cudaMallocManaged(&results, sizeof(int) * HEIGHT / 2);
  cudaMemset(results, 0, sizeof(int) * HEIGHT / 2);

  bar_sync_bar_arrive<<<1, HEIGHT>>>(arr, results);
  cudaDeviceSynchronize();

  int expected = 0;
  for (unsigned int i = 0; i < WIDTH * HEIGHT; ++i) {
    expected += i;
  }
  int result = 0;
  for (unsigned int i = 0; i < HEIGHT / 2; ++i) {
    result += results[i];
  }

  int res = result != expected;

  cudaFree(arr);
  cudaFree(results);
  return res;
}

int main() {
  int res = 0;
  res |= test_bar_warp_sync();
  res |= test_bar_sync_bar_arrive();
  return res;
}
