// ===------- cub_block_exchange.cu---------------------- *- CUDA -* ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <iostream>

__global__ void Kernel1(int *d_data) {
  using BlockLoadT = cub::BlockLoad<int, 128, 4, cub::BLOCK_LOAD_STRIPED>;
  using BlockStoreT = cub::BlockStore<int, 128, 4, cub::BLOCK_STORE_STRIPED>;
  __shared__ typename BlockLoadT::TempStorage load_temp_storage;
  __shared__ typename BlockStoreT::TempStorage store_temp_storage;
  int thread_data[4];
  BlockLoadT(load_temp_storage).Load(d_data, thread_data);
  BlockStoreT(store_temp_storage).Store(d_data, thread_data);
}

__global__ void Kernel2(int *d_data) {
  using BlockLoadT = cub::BlockLoad<int, 128, 4, cub::BLOCK_LOAD_STRIPED>;
  using BlockStoreT = cub::BlockStore<int, 128, 4, cub::BLOCK_STORE_STRIPED>;
  __shared__ typename BlockLoadT::TempStorage load_temp_storage;
  __shared__ typename BlockStoreT::TempStorage store_temp_storage;
  int thread_data[4];
  BlockLoadT(load_temp_storage).Load(d_data, thread_data, 100);
  BlockStoreT(store_temp_storage).Store(d_data, thread_data, 100);
}

__global__ void Kernel3(int *d_data) {
  using BlockLoadT = cub::BlockLoad<int, 128, 4, cub::BLOCK_LOAD_STRIPED>;
  using BlockStoreT = cub::BlockStore<int, 128, 4, cub::BLOCK_STORE_STRIPED>;
  __shared__ typename BlockLoadT::TempStorage load_temp_storage;
  __shared__ typename BlockStoreT::TempStorage store_temp_storage;
  int thread_data[4];
  BlockLoadT(load_temp_storage).Load(d_data, thread_data, 100, 0);
  BlockStoreT(store_temp_storage).Store(d_data, thread_data, 100);
}

__global__ void Kernel4(int *d_data) {
  using BlockLoadT = cub::BlockLoad<int, 128, 4, cub::BLOCK_LOAD_DIRECT>;
  using BlockStoreT = cub::BlockStore<int, 128, 4, cub::BLOCK_STORE_DIRECT>;
  __shared__ typename BlockLoadT::TempStorage load_temp_storage;
  __shared__ typename BlockStoreT::TempStorage store_temp_storage;
  int thread_data[4];
  BlockLoadT(load_temp_storage).Load(d_data, thread_data);
  BlockStoreT(store_temp_storage).Store(d_data, thread_data);
}

__global__ void Kernel5(int *d_data) {
  using BlockLoadT = cub::BlockLoad<int, 128, 4, cub::BLOCK_LOAD_DIRECT>;
  using BlockStoreT = cub::BlockStore<int, 128, 4, cub::BLOCK_STORE_DIRECT>;
  __shared__ typename BlockLoadT::TempStorage load_temp_storage;
  __shared__ typename BlockStoreT::TempStorage store_temp_storage;
  int thread_data[4];
  BlockLoadT(load_temp_storage).Load(d_data, thread_data, 100);
  BlockStoreT(store_temp_storage).Store(d_data, thread_data, 100);
}

__global__ void Kernel6(int *d_data) {
  using BlockLoadT = cub::BlockLoad<int, 128, 4, cub::BLOCK_LOAD_DIRECT>;
  using BlockStoreT = cub::BlockStore<int, 128, 4, cub::BLOCK_STORE_DIRECT>;
  __shared__ typename BlockLoadT::TempStorage load_temp_storage;
  __shared__ typename BlockStoreT::TempStorage store_temp_storage;
  int thread_data[4];
  BlockLoadT(load_temp_storage).Load(d_data, thread_data, 100, 0);
  BlockStoreT(store_temp_storage).Store(d_data, thread_data, 100);
}

__global__ void Kernel7(int *d_data) {
  using BlockLoadT = cub::BlockLoad<int, 128, 4, cub::BLOCK_LOAD_VECTORIZE>;
  using BlockStoreT = cub::BlockStore<int, 128, 4, cub::BLOCK_STORE_VECTORIZE>;
  __shared__ typename BlockLoadT::TempStorage load_temp_storage;
  __shared__ typename BlockStoreT::TempStorage store_temp_storage;
  int thread_data[4];
  BlockLoadT(load_temp_storage).Load(d_data, thread_data);
  BlockStoreT(store_temp_storage).Store(d_data, thread_data);
}

__global__ void Kernel8(int *d_data) {
  using BlockLoadT = cub::BlockLoad<int, 128, 4, cub::BLOCK_LOAD_VECTORIZE>;
  using BlockStoreT = cub::BlockStore<int, 128, 4, cub::BLOCK_STORE_VECTORIZE>;
  __shared__ typename BlockLoadT::TempStorage load_temp_storage;
  __shared__ typename BlockStoreT::TempStorage store_temp_storage;
  int thread_data[4];
  BlockLoadT(load_temp_storage).Load(d_data, thread_data, 100);
  BlockStoreT(store_temp_storage).Store(d_data, thread_data, 100);
}

__global__ void Kernel9(int *d_data) {
  using BlockLoadT = cub::BlockLoad<int, 128, 4, cub::BLOCK_LOAD_VECTORIZE>;
  using BlockStoreT = cub::BlockStore<int, 128, 4, cub::BLOCK_STORE_VECTORIZE>;
  __shared__ typename BlockLoadT::TempStorage load_temp_storage;
  __shared__ typename BlockStoreT::TempStorage store_temp_storage;
  int thread_data[4];
  BlockLoadT(load_temp_storage).Load(d_data, thread_data, 100, 0);
  BlockStoreT(store_temp_storage).Store(d_data, thread_data, 100);
}

__global__ void Kernel10(int *d_data) {
  using BlockLoadT = cub::BlockLoad<int, 128, 4, cub::BLOCK_LOAD_TRANSPOSE>;
  using BlockStoreT = cub::BlockStore<int, 128, 4, cub::BLOCK_STORE_TRANSPOSE>;
  __shared__ typename BlockLoadT::TempStorage load_temp_storage;
  __shared__ typename BlockStoreT::TempStorage store_temp_storage;
  int thread_data[4];
  BlockLoadT(load_temp_storage).Load(d_data, thread_data);
  BlockStoreT(store_temp_storage).Store(d_data, thread_data);
}

__global__ void Kernel11(int *d_data) {
  using BlockLoadT = cub::BlockLoad<int, 128, 4, cub::BLOCK_LOAD_TRANSPOSE>;
  using BlockStoreT = cub::BlockStore<int, 128, 4, cub::BLOCK_STORE_TRANSPOSE>;
  __shared__ typename BlockLoadT::TempStorage load_temp_storage;
  __shared__ typename BlockStoreT::TempStorage store_temp_storage;
  int thread_data[4];
  BlockLoadT(load_temp_storage).Load(d_data, thread_data, 100);
  BlockStoreT(store_temp_storage).Store(d_data, thread_data, 100);
}

__global__ void Kernel12(int *d_data) {
  using BlockLoadT = cub::BlockLoad<int, 128, 4, cub::BLOCK_LOAD_TRANSPOSE>;
  using BlockStoreT = cub::BlockStore<int, 128, 4, cub::BLOCK_STORE_TRANSPOSE>;
  __shared__ typename BlockLoadT::TempStorage load_temp_storage;
  __shared__ typename BlockStoreT::TempStorage store_temp_storage;
  int thread_data[4];
  BlockLoadT(load_temp_storage).Load(d_data, thread_data, 100, 0);
  BlockStoreT(store_temp_storage).Store(d_data, thread_data, 100);
}

__global__ void Kernel13(int *d_data) {
  using BlockLoadT = cub::BlockLoad<int, 128, 4, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  using BlockStoreT = cub::BlockStore<int, 128, 4, cub::BLOCK_STORE_WARP_TRANSPOSE>;
  __shared__ typename BlockLoadT::TempStorage load_temp_storage;
  __shared__ typename BlockStoreT::TempStorage store_temp_storage;
  int thread_data[4];
  BlockLoadT(load_temp_storage).Load(d_data, thread_data);
  BlockStoreT(store_temp_storage).Store(d_data, thread_data);
}

__global__ void Kernel14(int *d_data) {
  using BlockLoadT = cub::BlockLoad<int, 128, 4, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  using BlockStoreT = cub::BlockStore<int, 128, 4, cub::BLOCK_STORE_WARP_TRANSPOSE>;
  __shared__ typename BlockLoadT::TempStorage load_temp_storage;
  __shared__ typename BlockStoreT::TempStorage store_temp_storage;
  int thread_data[4];
  BlockLoadT(load_temp_storage).Load(d_data, thread_data, 100);
  BlockStoreT(store_temp_storage).Store(d_data, thread_data, 100);
}

__global__ void Kernel15(int *d_data) {
  using BlockLoadT = cub::BlockLoad<int, 128, 4, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  using BlockStoreT = cub::BlockStore<int, 128, 4, cub::BLOCK_STORE_WARP_TRANSPOSE>;
  __shared__ typename BlockLoadT::TempStorage load_temp_storage;
  __shared__ typename BlockStoreT::TempStorage store_temp_storage;
  int thread_data[4];
  BlockLoadT(load_temp_storage).Load(d_data, thread_data, 100, 0);
  BlockStoreT(store_temp_storage).Store(d_data, thread_data, 100);
}

bool test1() {
  int *d_data;
  cudaMallocManaged(&d_data, sizeof(int) * 512);
  for (int i = 0; i < 512; i++) {
    d_data[i] = i;
  }

  Kernel1<<<1, 128>>>(d_data);
  Kernel2<<<1, 128>>>(d_data);
  Kernel3<<<1, 128>>>(d_data);
  Kernel4<<<1, 128>>>(d_data);
  Kernel5<<<1, 128>>>(d_data);
  Kernel6<<<1, 128>>>(d_data);
  Kernel7<<<1, 128>>>(d_data);
  Kernel8<<<1, 128>>>(d_data);
  Kernel9<<<1, 128>>>(d_data);
  cudaDeviceSynchronize();

  for (int i = 0; i < 512; ++i) {
    if (d_data[i] != i) {
      std::cout << "test1 failed\n";
      std::ostream_iterator<int> Iter(std::cout, ", ");
      std::copy(d_data, d_data + 512, Iter);
      std::cout << std::endl;
      cudaFree(d_data);
      return false;
    }
  }
  cudaFree(d_data);
  std::cout << "test1 pass\n";
  return true;
}

bool test2() {
  int *d_data;
  cudaMallocManaged(&d_data, sizeof(int) * 512);
  for (int i = 0; i < 512; i++) {
    d_data[i] = i;
  }

  Kernel10<<<1, 128>>>(d_data);
  Kernel11<<<1, 128>>>(d_data);
  Kernel12<<<1, 128>>>(d_data);
  Kernel13<<<1, 128>>>(d_data);
  Kernel14<<<1, 128>>>(d_data);
  Kernel15<<<1, 128>>>(d_data);
  cudaDeviceSynchronize();

  for (int i = 0; i < 512; ++i) {
    if (d_data[i] != i) {
      std::cout << "test2 failed\n";
      std::ostream_iterator<int> Iter(std::cout, ", ");
      std::copy(d_data, d_data + 512, Iter);
      std::cout << std::endl;
      cudaFree(d_data);
      return false;
    }
  }
  cudaFree(d_data);
  std::cout << "test2 pass\n";
  return true;
}

int main() {
  return !(test1() && test2());
}
