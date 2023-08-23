// ====------ match.cu---------- *- CUDA -* ----===////
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
    host_data[i] = 0;
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

__global__ void MatchAllKernel1(unsigned int* data, int *p) {
  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  int pre = 0;
  unsigned int mask = 0xFFFFFFFF;
  unsigned int r = __match_all_sync(mask, 0, &pre);
  __syncwarp();
  p[threadid] = pre;
  data[threadid] = r;
}

__global__ void MatchAllKernel2(unsigned int* data, int *p) {
    int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
    int pre = 0;
    unsigned int mask = 0xFFFFFFFF;
    unsigned int r = __match_all_sync(mask, threadid, &pre);
    __syncwarp();
    p[threadid] = pre;
    data[threadid] = r;
}

__global__ void MatchAnyKernel1(unsigned int* data) {
    int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
    unsigned int mask = 0xFFFFFFFF;
    unsigned int r = __match_any_sync(mask, threadid);
    __syncwarp();
    data[threadid] = r;
}

__global__ void MatchAnyKernel2(unsigned int* data) {
    int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
    unsigned int mask = 0xFFFFFFFF;
    unsigned int r = __match_any_sync(mask, threadid%4);
    __syncwarp();
    data[threadid] = r;
}

__global__ void MatchAnyKernel3(unsigned int* data) {
    int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
    unsigned int mask = 0xAAAAAAAA;
    unsigned int r = __match_any_sync(mask, threadid%4);
    __syncwarp();
    data[threadid] = r;
}

int main() {
  bool Result = true;
  int* dev_data = nullptr;
  unsigned int *dev_data_u = nullptr;
  dim3 GridSize;
  dim3 BlockSize;
  cudaMalloc(&dev_data, DATA_NUM * sizeof(int));
  cudaMalloc(&dev_data_u, DATA_NUM * sizeof(unsigned int));

  //1
  GridSize = {2};
  BlockSize = {32, 2, 1};
  unsigned int expect11[DATA_NUM] = {
    4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295,
    4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295,
    4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295,
    4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295
  };

  int expect12[DATA_NUM] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  };
  init_data<int>(dev_data, DATA_NUM);
  init_data<unsigned int>(dev_data_u, DATA_NUM);
  MatchAllKernel1<<<GridSize, BlockSize>>>(dev_data_u, dev_data);

  cudaDeviceSynchronize();
  if(!verify_data<unsigned int>(dev_data_u, expect11, DATA_NUM)) {
    std::cout << "MatchAllKernel1" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data<unsigned int>(expect11, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data<unsigned int>(dev_data_u, DATA_NUM);
  }

  if(!verify_data<int>(dev_data, expect12, DATA_NUM)) {
    std::cout << "MatchAllKernel1" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data<int>(expect12, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data<int>(dev_data, DATA_NUM);
  }

  //2
  GridSize = {2};
  BlockSize = {32, 2, 1};
  unsigned int expect21[DATA_NUM] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0    
  };

  int expect22[DATA_NUM] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0    
  };
  init_data<int>(dev_data, DATA_NUM);
  init_data<unsigned int>(dev_data_u, DATA_NUM);
  MatchAllKernel2<<<GridSize, BlockSize>>>(dev_data_u, dev_data);

  cudaDeviceSynchronize();
  if(!verify_data<unsigned int>(dev_data_u, expect21, DATA_NUM)) {
    std::cout << "MatchAllKernel2" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data<unsigned int>(expect21, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data<unsigned int>(dev_data_u, DATA_NUM);
  }

  if(!verify_data<int>(dev_data, expect22, DATA_NUM)) {
    std::cout << "MatchAllKernel2" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data<int>(expect22, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data<int>(dev_data, DATA_NUM);
  }

  //3
  GridSize = {2};
  BlockSize = {32, 2, 1};
  unsigned int expect3[DATA_NUM] = {
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648,
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648,
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648,
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648    
  };

  init_data<unsigned int>(dev_data_u, DATA_NUM);
  MatchAnyKernel1<<<GridSize, BlockSize>>>(dev_data_u);

  cudaDeviceSynchronize();
  if(!verify_data<unsigned int>(dev_data_u, expect3, DATA_NUM)) {
    std::cout << "MatchAnyKernel1" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data<unsigned int>(expect3, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data<unsigned int>(dev_data_u, DATA_NUM);
  }

  //4
  GridSize = {2};
  BlockSize = {32, 2, 1};
  unsigned int expect4[DATA_NUM] = {
    286331153, 572662306, 1145324612, 2290649224, 286331153, 572662306, 1145324612, 2290649224, 286331153, 572662306, 1145324612, 2290649224, 286331153, 572662306, 1145324612, 2290649224, 286331153, 572662306, 1145324612, 2290649224, 286331153, 572662306, 1145324612, 2290649224, 286331153, 572662306, 1145324612, 2290649224, 286331153, 572662306, 1145324612, 2290649224,
    286331153, 572662306, 1145324612, 2290649224, 286331153, 572662306, 1145324612, 2290649224, 286331153, 572662306, 1145324612, 2290649224, 286331153, 572662306, 1145324612, 2290649224, 286331153, 572662306, 1145324612, 2290649224, 286331153, 572662306, 1145324612, 2290649224, 286331153, 572662306, 1145324612, 2290649224, 286331153, 572662306, 1145324612, 2290649224,
    286331153, 572662306, 1145324612, 2290649224, 286331153, 572662306, 1145324612, 2290649224, 286331153, 572662306, 1145324612, 2290649224, 286331153, 572662306, 1145324612, 2290649224, 286331153, 572662306, 1145324612, 2290649224, 286331153, 572662306, 1145324612, 2290649224, 286331153, 572662306, 1145324612, 2290649224, 286331153, 572662306, 1145324612, 2290649224,
    286331153, 572662306, 1145324612, 2290649224, 286331153, 572662306, 1145324612, 2290649224, 286331153, 572662306, 1145324612, 2290649224, 286331153, 572662306, 1145324612, 2290649224, 286331153, 572662306, 1145324612, 2290649224, 286331153, 572662306, 1145324612, 2290649224, 286331153, 572662306, 1145324612, 2290649224, 286331153, 572662306, 1145324612, 2290649224
  };

  init_data<unsigned int>(dev_data_u, DATA_NUM);
  MatchAnyKernel2<<<GridSize, BlockSize>>>(dev_data_u);

  cudaDeviceSynchronize();
  if(!verify_data<unsigned int>(dev_data_u, expect4, DATA_NUM)) {
    std::cout << "MatchAnyKernel2" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data<unsigned int>(expect4, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data<unsigned int>(dev_data_u, DATA_NUM);
  }

  //5
  GridSize = {2};
  BlockSize = {32, 2, 1};
  unsigned int expect5[DATA_NUM] = {
    0, 572662306, 0, 2290649224, 0, 572662306, 0, 2290649224, 0, 572662306, 0, 2290649224, 0, 572662306, 0, 2290649224, 0, 572662306, 0, 2290649224, 0, 572662306, 0, 2290649224, 0, 572662306, 0, 2290649224, 0, 572662306, 0, 2290649224,
    0, 572662306, 0, 2290649224, 0, 572662306, 0, 2290649224, 0, 572662306, 0, 2290649224, 0, 572662306, 0, 2290649224, 0, 572662306, 0, 2290649224, 0, 572662306, 0, 2290649224, 0, 572662306, 0, 2290649224, 0, 572662306, 0, 2290649224,
    0, 572662306, 0, 2290649224, 0, 572662306, 0, 2290649224, 0, 572662306, 0, 2290649224, 0, 572662306, 0, 2290649224, 0, 572662306, 0, 2290649224, 0, 572662306, 0, 2290649224, 0, 572662306, 0, 2290649224, 0, 572662306, 0, 2290649224,
    0, 572662306, 0, 2290649224, 0, 572662306, 0, 2290649224, 0, 572662306, 0, 2290649224, 0, 572662306, 0, 2290649224, 0, 572662306, 0, 2290649224, 0, 572662306, 0, 2290649224, 0, 572662306, 0, 2290649224, 0, 572662306, 0, 2290649224
  };

  init_data<unsigned int>(dev_data_u, DATA_NUM);
  MatchAnyKernel3<<<GridSize, BlockSize>>>(dev_data_u);

  cudaDeviceSynchronize();
  if(!verify_data<unsigned int>(dev_data_u, expect5, DATA_NUM, 2)) {
    std::cout << "MatchAnyKernel3" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data<unsigned int>(expect5, DATA_NUM);
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


