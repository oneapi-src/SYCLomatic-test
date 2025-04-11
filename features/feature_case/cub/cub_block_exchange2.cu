// ===------- cub_block_exchange2.cu---------------------- *- CUDA -* ------===//
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

__global__ void BlockedToWarpStripedKernel(int *d_data) {

  typedef cub::BlockExchange<int, 128, 4> BlockExchange;
  __shared__ typename BlockExchange::TempStorage temp_storage;
  int thread_data[4];
  cub::LoadDirectBlocked(threadIdx.x, d_data, thread_data);
  BlockExchange(temp_storage).BlockedToWarpStriped(thread_data, thread_data);
  cub::StoreDirectBlocked(threadIdx.x, d_data, thread_data);
}

__global__ void WarpStripedToBlockedKernel(int *d_data) {

  typedef cub::BlockExchange<int, 128, 4> BlockExchange;
  __shared__ typename BlockExchange::TempStorage temp_storage;
  int thread_data[4];
  cub::LoadDirectBlocked(threadIdx.x, d_data, thread_data);
  BlockExchange(temp_storage).WarpStripedToBlocked(thread_data, thread_data);
  cub::StoreDirectBlocked(threadIdx.x, d_data, thread_data);
}

bool test_blocked_to_warp_striped() {
  int *d_data, expected[512];
  cudaMallocManaged(&d_data, sizeof(int) * 512);
  for (int i = 0; i < 512; ++i)
    d_data[i] = i;

  BlockedToWarpStripedKernel<<<1, 128>>>(d_data);
  cudaDeviceSynchronize();
  size_t warp_id = 0, warp_offset = 0, lane_id = 0;
  for (int i = 0; i < 128; i++) {
    warp_id = i / 32;
    lane_id = i % 32;
    warp_offset = warp_id * 32 * 4;
    expected[4 * i + 0] = warp_offset + lane_id + 0 * 32;
    expected[4 * i + 1] = warp_offset + lane_id + 1 * 32;
    expected[4 * i + 2] = warp_offset + lane_id + 2 * 32;
    expected[4 * i + 3] = warp_offset + lane_id + 3 * 32;
  }

  for (int i = 0; i < 512; ++i) {
    if (expected[i] != d_data[i]) {
      std::cout << "test_blocked_to_warp_striped failed\n";
      std::ostream_iterator<int> Iter(std::cout, ", ");
      std::copy(d_data, d_data + 512, Iter);
      std::cout << std::endl;
      std::copy(expected, expected + 512, Iter);
      std::cout << std::endl;
      return false;
    }
  }
  std::cout << "test_blocked_to_warp_striped pass\n";
  return true;
}

bool test_warp_striped_to_blocked() {
  int *d_data, expected[512];
  cudaMallocManaged(&d_data, sizeof(int) * 512);
  size_t warp_id = 0, warp_offset = 0, lane_id = 0;
  for (int i = 0; i < 128; i++) {
    warp_id = i / 32;
    lane_id = i % 32;
    warp_offset = warp_id * 32 * 4;
    d_data[4 * i + 0] = warp_offset + lane_id + 0 * 32;
    d_data[4 * i + 1] = warp_offset + lane_id + 1 * 32;
    d_data[4 * i + 2] = warp_offset + lane_id + 2 * 32;
    d_data[4 * i + 3] = warp_offset + lane_id + 3 * 32;
  }

  WarpStripedToBlockedKernel<<<1, 128>>>(d_data);
  cudaDeviceSynchronize();

  for (int i = 0; i < 512; i++) {
    expected[i] = i;
  }

  for (int i = 0; i < 512; ++i) {
    if (expected[i] != d_data[i]) {
      std::cout << "test_warp_striped_to_blocked failed\n";
      std::ostream_iterator<int> Iter(std::cout, ", ");
      std::copy(d_data, d_data + 512, Iter);
      std::cout << std::endl;
      return false;
    }
  }
  std::cout << "test_warp_striped_to_blocked pass\n";
  return true;
}

int main() {
  return !(test_blocked_to_warp_striped() && test_warp_striped_to_blocked());
}
