// ====------ sync_warp_p1.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <iostream>
#include <cuda_runtime.h>
#include <vector>

#define WARP_SIZE 32
#define DATA_NUM 128

template<typename T = int>
void init_data(T* data, int num) {
  std::vector<T> host_data(num);
  for(int i = 0; i < num; i++)
    host_data[i] = i;
  cudaMemcpy(data, host_data.data(), num * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T = int>
bool verify_data(T* data, T* expect, int num, int step = 1) {
  std::vector<T> host_data(num);
  cudaMemcpy(host_data.data(), data, num * sizeof(T), cudaMemcpyDeviceToHost);
  for(int i = 0; i < num; i = i + step) {
    if(host_data[i] != expect[i]) {
      return false;
    }
  }
  return true;
}

template<typename T = int>
void print_data(T* data, int num) {
  std::vector<T> host_data(num);
  cudaMemcpy(host_data.data(), data, num * sizeof(T), cudaMemcpyDeviceToHost);
  for (int i = 0; i < num; i++) {
    std::cout << host_data[i] << ", ";
    if((i+1)%32 == 0)
        std::cout << std::endl;
  }
  std::cout << std::endl;
}


//sync API
__global__ void SyncWarpKernel(int* data) {
  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  int output = 0;
  __syncwarp();
  output = 1;
  data[threadid] = output;
}
__global__ void SyncAndKernel(int* data) {
  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  int output = 0;
  output = __syncthreads_and(threadid >= 16);
  data[threadid] = output;
}
__global__ void SyncOrKernel(int* data) {
  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  int output = 0;
  output = __syncthreads_or(threadid >= 16);
  data[threadid] = output;
}
__global__ void SyncCountKernel(int* data) {
  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  int output = 0;
  output = __syncthreads_count(threadid >= 16);
  data[threadid] = output;
}

// Warp API
__global__ void AnyKernel(int* data) {
  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  int output = 0;
  output = __any(threadid >= 16);
  data[threadid] = output;
}
__global__ void AllKernel(int* data) {
  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  int output = 0;
  output = __all(threadid >= 16);
  data[threadid] = output;
}
__global__ void BallotKernel(unsigned int* data) {
  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  int output = 0;
  output = __ballot(threadid >= 16);
  data[threadid] = output;
}
__global__ void ShuffleKernel1(unsigned int* data) {
  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  int output = 0;
  output = __shfl(threadid, threadid + 1);
  data[threadid] = output;
}
__global__ void ShuffleUpKernel1(unsigned int* data) {
  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  int output = 0;
  output = __shfl_up(threadid, 1);
  data[threadid] = output;
}
__global__ void ShuffleDownKernel1(unsigned int* data) {
  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  int output = 0;
  output = __shfl_down(threadid, 1);
  data[threadid] = output;
}
__global__ void ShuffleXorKernel1(unsigned int* data) {
  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  int output = 0;
  output = __shfl_xor(threadid, 2);
  data[threadid] = output;
}

__global__ void ShuffleKernel2(unsigned int* data) {
  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  int output = 0;
  output = __shfl(threadid, threadid + 1, 8);
  data[threadid] = output;
}
__global__ void ShuffleUpKernel2(unsigned int* data) {
  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  int output = 0;
  output = __shfl_up(threadid, 1, 8);
  data[threadid] = output;
}
__global__ void ShuffleDownKernel2(unsigned int* data) {
  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  int output = 0;
  output = __shfl_down(threadid, 1, 8);
  data[threadid] = output;
}
__global__ void ShuffleXorKernel2(unsigned int* data) {
  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  int output = 0;
  output = __shfl_xor(threadid, 1, 8);
  data[threadid] = output;
}

__global__ void AnySyncKernel1(int* data) {
  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  int output = 0;
  unsigned int mask = 0x55555555;
  output = __any_sync(mask, threadid >= 16);
  data[threadid] = output;
}
__global__ void AllSyncKernel1(int* data) {
  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  int output = 0;
  unsigned int mask = 0x55555555;
  output = __all_sync(mask, threadid >= 16);
  data[threadid] = output;
}
__global__ void BallotSyncKernel1(unsigned int* data) {
  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  int output = 0;
  unsigned int mask = 0x55555555;
  output = __ballot_sync(mask, threadid >= 16);
  data[threadid] = output;
}
__global__ void ShuffleSyncKernel1(unsigned int* data) {
  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  int output = 0;
  unsigned int mask = 0x55555555;
  output = __shfl_sync(mask, threadid, threadid + 1);
  data[threadid] = output;
}
__global__ void ShuffleUpSyncKernel1(unsigned int* data) {
  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  int output = 0;
  unsigned int mask = 0x55555555;
  output = __shfl_up_sync(mask, threadid, 1);
  data[threadid] = output;
}
__global__ void ShuffleDownSyncKernel1(unsigned int* data) {
  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  int output = 0;
  unsigned int mask = 0x55555555;
  output = __shfl_down_sync(mask, threadid, 1);
  data[threadid] = output;
}
__global__ void ShuffleXorSyncKernel1(unsigned int* data) {
  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  int output = 0;
  unsigned int mask = 0x55555555;
  output = __shfl_xor_sync(mask, threadid, 2);
  data[threadid] = output;
}


int main() {
  bool Result = true;
  int* dev_data = nullptr;
  unsigned int *dev_data_u = nullptr;
  dim3 GridSize;
  dim3 BlockSize;
  cudaMalloc(&dev_data, DATA_NUM * sizeof(int));
  cudaMalloc(&dev_data_u, DATA_NUM * sizeof(unsigned int));

  GridSize = {2};
  BlockSize = {32, 2, 1};
  int expect1[DATA_NUM] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  };
  init_data<int>(dev_data, DATA_NUM);

  SyncWarpKernel<<<GridSize, BlockSize>>>(dev_data);

  cudaDeviceSynchronize();
  if(!verify_data<int>(dev_data, expect1, DATA_NUM)) {
    std::cout << "SyncWarpKernel" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data<int>(expect1, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data<int>(dev_data, DATA_NUM);
  }

  GridSize = {2};
  BlockSize = {32, 2, 1};
  int expect2[DATA_NUM] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  };
  init_data<int>(dev_data, DATA_NUM);

  SyncAndKernel<<<GridSize, BlockSize>>>(dev_data);

  cudaDeviceSynchronize();
  if(!verify_data<int>(dev_data, expect2, DATA_NUM)) {
    std::cout << "SyncAndKernel" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data<int>(expect2, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data<int>(dev_data, DATA_NUM);
  }
  
  GridSize = {2};
  BlockSize = {32, 2, 1};
  int expect3[DATA_NUM] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  };
  init_data<int>(dev_data, DATA_NUM);

  SyncOrKernel<<<GridSize, BlockSize>>>(dev_data);

  cudaDeviceSynchronize();
  if(!verify_data<int>(dev_data, expect3, DATA_NUM)) {
    std::cout << "SyncOrKernel" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data<int>(expect3, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data<int>(dev_data, DATA_NUM);
  }

  GridSize = {2};
  BlockSize = {32, 2, 1};
  int expect4[DATA_NUM] = {
    48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48,
    48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64
  };
  init_data<int>(dev_data, DATA_NUM);

  SyncCountKernel<<<GridSize, BlockSize>>>(dev_data);

  cudaDeviceSynchronize();
  if(!verify_data<int>(dev_data, expect4, DATA_NUM)) {
    std::cout << "SyncCountKernel" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data<int>(expect4, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data<int>(dev_data, DATA_NUM);
  }

  GridSize = {2};
  BlockSize = {32, 2, 1};
  int expect5[DATA_NUM] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  };
  init_data<int>(dev_data, DATA_NUM);

  AnyKernel<<<GridSize, BlockSize>>>(dev_data);

  cudaDeviceSynchronize();
  if(!verify_data<int>(dev_data, expect5, DATA_NUM)) {
    std::cout << "AnyKernel" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data<int>(expect5, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data<int>(dev_data, DATA_NUM);
  }

  GridSize = {2};
  BlockSize = {32, 2, 1};
  int expect6[DATA_NUM] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  };
  init_data<int>(dev_data, DATA_NUM);

  AllKernel<<<GridSize, BlockSize>>>(dev_data);

  cudaDeviceSynchronize();
  if(!verify_data<int>(dev_data, expect6, DATA_NUM)) {
    std::cout << "AllKernel" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data<int>(expect6, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data<int>(dev_data, DATA_NUM);
  }

  GridSize = {2};
  BlockSize = {32, 2, 1};
  unsigned int expect7[DATA_NUM] = {
    4294901760, 4294901760, 4294901760, 4294901760, 4294901760, 4294901760, 4294901760, 4294901760, 4294901760, 4294901760, 4294901760, 4294901760, 4294901760, 4294901760, 4294901760, 4294901760, 4294901760, 4294901760, 4294901760, 4294901760, 4294901760, 4294901760, 4294901760, 4294901760, 4294901760, 4294901760, 4294901760, 4294901760, 4294901760, 4294901760, 4294901760, 4294901760,
    4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295,
    4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295,
    4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295
  };
  init_data<unsigned int>(dev_data_u, DATA_NUM);

  BallotKernel<<<GridSize, BlockSize>>>(dev_data_u);

  cudaDeviceSynchronize();
  if(!verify_data<unsigned int>(dev_data_u, expect7, DATA_NUM)) {
    std::cout << "BallotKernel" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data<unsigned int>(expect7, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data<unsigned int>(dev_data_u, DATA_NUM);
  }

  GridSize = {2};
  BlockSize = {32, 2, 1};
  int expect8[DATA_NUM] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  };
  init_data(dev_data, DATA_NUM);

  AnySyncKernel1<<<GridSize, BlockSize>>>(dev_data);

  cudaDeviceSynchronize();
  if(!verify_data(dev_data, expect8, DATA_NUM, 2)) {
    std::cout << "AnySyncKernel1" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data(expect8, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data(dev_data, DATA_NUM);
  }


  GridSize = {2};
  BlockSize = {32, 2, 1};
  int expect9[DATA_NUM] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  };
  init_data(dev_data, DATA_NUM);

  AllSyncKernel1<<<GridSize, BlockSize>>>(dev_data);

  cudaDeviceSynchronize();
  if(!verify_data(dev_data, expect9, DATA_NUM, 2)) {
    std::cout << "AllSyncKernel1" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data(expect9, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data(dev_data, DATA_NUM);
  }


  GridSize = {2};
  BlockSize = {32, 2, 1};
  unsigned int expect10[DATA_NUM] = {
    1431633920, 1431633920, 1431633920, 1431633920, 1431633920, 1431633920, 1431633920, 1431633920, 1431633920, 1431633920, 1431633920, 1431633920, 1431633920, 1431633920, 1431633920, 1431633920, 1431633920, 1431633920, 1431633920, 1431633920, 1431633920, 1431633920, 1431633920, 1431633920, 1431633920, 1431633920, 1431633920, 1431633920, 1431633920, 1431633920, 1431633920, 1431633920,
    1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765,
    1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765,
    1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765, 1431655765
  };
  init_data<unsigned int>(dev_data_u, DATA_NUM);

  BallotSyncKernel1<<<GridSize, BlockSize>>>(dev_data_u);

  cudaDeviceSynchronize();
  if(!verify_data<unsigned int>(dev_data_u, expect10, DATA_NUM, 2)) {
    std::cout << "BallotSyncKernel1" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data<unsigned int>(expect10, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data<unsigned int>(dev_data_u, DATA_NUM);
  }

  GridSize = {2};
  BlockSize = {32, 2, 1};
  unsigned int expect11[DATA_NUM] = {
    1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,
    0,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,
    61,62,63,32,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,
    90,91,92,93,94,95,64,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,
    114,115,116,117,118,119,120,121,122,123,124,125,126,127,96
  };
  init_data<unsigned int>(dev_data_u, DATA_NUM);

  ShuffleSyncKernel1<<<GridSize, BlockSize>>>(dev_data_u);

  cudaDeviceSynchronize();
  if(!verify_data<unsigned int>(dev_data_u, expect11, DATA_NUM)) {
    std::cout << "ShuffleSyncKernel1" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data<unsigned int>(expect11, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data<unsigned int>(dev_data_u, DATA_NUM);
  }

  GridSize = {2};
  BlockSize = {32, 2, 1};
  unsigned int expect12[DATA_NUM] = {
    0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,32,32,
    33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,64,
    64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,96,
    96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,
    120,121,122,123,124,125,126
  };
  init_data<unsigned int>(dev_data_u, DATA_NUM);

  ShuffleUpSyncKernel1<<<GridSize, BlockSize>>>(dev_data_u);

  cudaDeviceSynchronize();
  if(!verify_data<unsigned int>(dev_data_u, expect12, DATA_NUM)) {
    std::cout << "ShuffleUpSyncKernel1" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data<unsigned int>(expect12, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data<unsigned int>(dev_data_u, DATA_NUM);
  }
  

  GridSize = {2};
  BlockSize = {32, 2, 1};
  unsigned int expect13[DATA_NUM] = {
    1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,31,
    33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,
    62,63,63,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,
    91,92,93,94,95,95,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,
    115,116,117,118,119,120,121,122,123,124,125,126,127,127
  };
  init_data<unsigned int>(dev_data_u, DATA_NUM);

  ShuffleDownSyncKernel1<<<GridSize, BlockSize>>>(dev_data_u);

  cudaDeviceSynchronize();
  if(!verify_data<unsigned int>(dev_data_u, expect13, DATA_NUM)) {
    std::cout << "ShuffleDownSyncKernel1" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data<unsigned int>(expect13, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data<unsigned int>(dev_data_u, DATA_NUM);
  }

  GridSize = {2};
  BlockSize = {32, 2, 1};
  unsigned int expect14[DATA_NUM] = {
    2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13,18,19,16,17,22,23,20,21,26,27,24,25,30,31,
    28,29,34,35,32,33,38,39,36,37,42,43,40,41,46,47,44,45,50,51,48,49,54,55,52,53,58,
    59,56,57,62,63,60,61,66,67,64,65,70,71,68,69,74,75,72,73,78,79,76,77,82,83,80,81,
    86,87,84,85,90,91,88,89,94,95,92,93,98,99,96,97,102,103,100,101,106,107,104,105,
    110,111,108,109,114,115,112,113,118,119,116,117,122,123,120,121,126,127,124,125
  };
  init_data<unsigned int>(dev_data_u, DATA_NUM);

  ShuffleXorSyncKernel1<<<GridSize, BlockSize>>>(dev_data_u);

  cudaDeviceSynchronize();
  if(!verify_data<unsigned int>(dev_data_u, expect14, DATA_NUM)) {
    std::cout << "ShuffleXorSyncKernel1" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data<unsigned int>(expect14, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data<unsigned int>(dev_data_u, DATA_NUM);
  }

  GridSize = {2};
  BlockSize = {32, 2, 1};
  unsigned int expect15[DATA_NUM] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 32,
    65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 64,
    97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 96
  };
  init_data<unsigned int>(dev_data_u, DATA_NUM);

  ShuffleKernel1<<<GridSize, BlockSize>>>(dev_data_u);

  cudaDeviceSynchronize();
  if(!verify_data<unsigned int>(dev_data_u, expect15, DATA_NUM)) {
    std::cout << "ShuffleKernel1" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data<unsigned int>(expect15, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data<unsigned int>(dev_data_u, DATA_NUM);
  }

  GridSize = {2};
  BlockSize = {32, 2, 1};
  unsigned int expect16[DATA_NUM] = {
    0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    32, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
    64, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94,
    96, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126
  };
  init_data<unsigned int>(dev_data_u, DATA_NUM);

  ShuffleUpKernel1<<<GridSize, BlockSize>>>(dev_data_u);

  cudaDeviceSynchronize();
  if(!verify_data<unsigned int>(dev_data_u, expect16, DATA_NUM)) {
    std::cout << "ShuffleUpKernel1" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data<unsigned int>(expect16, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data<unsigned int>(dev_data_u, DATA_NUM);
  }

  GridSize = {2};
  BlockSize = {32, 2, 1};
  unsigned int expect17[DATA_NUM] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 31,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 63,
    65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 95,
    97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 127
  };
  init_data<unsigned int>(dev_data_u, DATA_NUM);

  ShuffleDownKernel1<<<GridSize, BlockSize>>>(dev_data_u);

  cudaDeviceSynchronize();
  if(!verify_data<unsigned int>(dev_data_u, expect17, DATA_NUM)) {
    std::cout << "ShuffleDownKernel1" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data<unsigned int>(expect17, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data<unsigned int>(dev_data_u, DATA_NUM);
  }

  GridSize = {2};
  BlockSize = {32, 2, 1};
  unsigned int expect18[DATA_NUM] = {
    2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13, 18, 19, 16, 17, 22, 23, 20, 21, 26, 27, 24, 25, 30, 31, 28, 29,
    34, 35, 32, 33, 38, 39, 36, 37, 42, 43, 40, 41, 46, 47, 44, 45, 50, 51, 48, 49, 54, 55, 52, 53, 58, 59, 56, 57, 62, 63, 60, 61,
    66, 67, 64, 65, 70, 71, 68, 69, 74, 75, 72, 73, 78, 79, 76, 77, 82, 83, 80, 81, 86, 87, 84, 85, 90, 91, 88, 89, 94, 95, 92, 93,
    98, 99, 96, 97, 102, 103, 100, 101, 106, 107, 104, 105, 110, 111, 108, 109, 114, 115, 112, 113, 118, 119, 116, 117, 122, 123, 120, 121, 126, 127, 124, 125
  };
  init_data<unsigned int>(dev_data_u, DATA_NUM);

  ShuffleXorKernel1<<<GridSize, BlockSize>>>(dev_data_u);

  cudaDeviceSynchronize();
  if(!verify_data<unsigned int>(dev_data_u, expect18, DATA_NUM)) {
    std::cout << "ShuffleXorKernel1" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data<unsigned int>(expect18, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data<unsigned int>(dev_data_u, DATA_NUM);
  }

  GridSize = {2};
  BlockSize = {32, 2, 1};
  unsigned int expect19[DATA_NUM] = {
    1, 2, 3, 4, 5, 6, 7, 7, 9, 10, 11, 12, 13, 14, 15, 15, 17, 18, 19, 20, 21, 22, 23, 23, 25, 26, 27, 28, 29, 30, 31, 31,
    33, 34, 35, 36, 37, 38, 39, 39, 41, 42, 43, 44, 45, 46, 47, 47, 49, 50, 51, 52, 53, 54, 55, 55, 57, 58, 59, 60, 61, 62, 63, 63,
    65, 66, 67, 68, 69, 70, 71, 71, 73, 74, 75, 76, 77, 78, 79, 79, 81, 82, 83, 84, 85, 86, 87, 87, 89, 90, 91, 92, 93, 94, 95, 95,
    97, 98, 99, 100, 101, 102, 103, 103, 105, 106, 107, 108, 109, 110, 111, 111, 113, 114, 115, 116, 117, 118, 119, 119, 121, 122, 123, 124, 125, 126, 127, 127
  };
  init_data<unsigned int>(dev_data_u, DATA_NUM);

  ShuffleDownKernel2<<<GridSize, BlockSize>>>(dev_data_u);

  cudaDeviceSynchronize();
  if(!verify_data<unsigned int>(dev_data_u, expect19, DATA_NUM)) {
    std::cout << "ShuffleDownKernel2" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data<unsigned int>(expect19, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data<unsigned int>(dev_data_u, DATA_NUM);
  }

  GridSize = {2};
  BlockSize = {32, 2, 1};
  unsigned int expect20[DATA_NUM] = {
    0, 0, 1, 2, 3, 4, 5, 6, 8, 8, 9, 10, 11, 12, 13, 14, 16, 16, 17, 18, 19, 20, 21, 22, 24, 24, 25, 26, 27, 28, 29, 30,
    32, 32, 33, 34, 35, 36, 37, 38, 40, 40, 41, 42, 43, 44, 45, 46, 48, 48, 49, 50, 51, 52, 53, 54, 56, 56, 57, 58, 59, 60, 61, 62,
    64, 64, 65, 66, 67, 68, 69, 70, 72, 72, 73, 74, 75, 76, 77, 78, 80, 80, 81, 82, 83, 84, 85, 86, 88, 88, 89, 90, 91, 92, 93, 94,
    96, 96, 97, 98, 99, 100, 101, 102, 104, 104, 105, 106, 107, 108, 109, 110, 112, 112, 113, 114, 115, 116, 117, 118, 120, 120, 121, 122, 123, 124, 125, 126
  };
  init_data<unsigned int>(dev_data_u, DATA_NUM);

  ShuffleUpKernel2<<<GridSize, BlockSize>>>(dev_data_u);

  cudaDeviceSynchronize();
  if(!verify_data<unsigned int>(dev_data_u, expect20, DATA_NUM)) {
    std::cout << "ShuffleUpKernel2" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data<unsigned int>(expect20, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data<unsigned int>(dev_data_u, DATA_NUM);
  }


  GridSize = {2};
  BlockSize = {32, 2, 1};
  unsigned int expect21[DATA_NUM] = {
    1, 2, 3, 4, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 15, 8, 17, 18, 19, 20, 21, 22, 23, 16, 25, 26, 27, 28, 29, 30, 31, 24,
    33, 34, 35, 36, 37, 38, 39, 32, 41, 42, 43, 44, 45, 46, 47, 40, 49, 50, 51, 52, 53, 54, 55, 48, 57, 58, 59, 60, 61, 62, 63, 56,
    65, 66, 67, 68, 69, 70, 71, 64, 73, 74, 75, 76, 77, 78, 79, 72, 81, 82, 83, 84, 85, 86, 87, 80, 89, 90, 91, 92, 93, 94, 95, 88,
    97, 98, 99, 100, 101, 102, 103, 96, 105, 106, 107, 108, 109, 110, 111, 104, 113, 114, 115, 116, 117, 118, 119, 112, 121, 122, 123, 124, 125, 126, 127, 120
  };
  init_data<unsigned int>(dev_data_u, DATA_NUM);

  ShuffleKernel2<<<GridSize, BlockSize>>>(dev_data_u);

  cudaDeviceSynchronize();
  if(!verify_data<unsigned int>(dev_data_u, expect21, DATA_NUM)) {
    std::cout << "ShuffleKernel2" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data<unsigned int>(expect21, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data<unsigned int>(dev_data_u, DATA_NUM);
  }


  GridSize = {2};
  BlockSize = {32, 2, 1};
  unsigned int expect22[DATA_NUM] = {
    1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22, 25, 24, 27, 26, 29, 28, 31, 30,
    33, 32, 35, 34, 37, 36, 39, 38, 41, 40, 43, 42, 45, 44, 47, 46, 49, 48, 51, 50, 53, 52, 55, 54, 57, 56, 59, 58, 61, 60, 63, 62,
    65, 64, 67, 66, 69, 68, 71, 70, 73, 72, 75, 74, 77, 76, 79, 78, 81, 80, 83, 82, 85, 84, 87, 86, 89, 88, 91, 90, 93, 92, 95, 94,
    97, 96, 99, 98, 101, 100, 103, 102, 105, 104, 107, 106, 109, 108, 111, 110, 113, 112, 115, 114, 117, 116, 119, 118, 121, 120, 123, 122, 125, 124, 127, 126
  };
  init_data<unsigned int>(dev_data_u, DATA_NUM);

  ShuffleXorKernel2<<<GridSize, BlockSize>>>(dev_data_u);

  cudaDeviceSynchronize();
  if(!verify_data<unsigned int>(dev_data_u, expect22, DATA_NUM)) {
    std::cout << "ShuffleXorKernel2" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data<unsigned int>(expect22, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data<unsigned int>(dev_data_u, DATA_NUM);
  }

  if(Result)
    std::cout << "passed" << std::endl;
  else {
    exit(-1);
  }
  return 0;
}


