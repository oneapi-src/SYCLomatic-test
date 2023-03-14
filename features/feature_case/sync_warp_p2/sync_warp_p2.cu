// ====------ sync_warp_p2.cu---------- *- CUDA -* ----===////
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
    host_data[i] = i + 3;
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
void print_data(T* data, int num, bool is_host = false) {
  if(is_host) {
    for (int i = 0; i < num; i++) {
      std::cout << data[i] << ", ";
      if((i+1)%16 == 0)
          std::cout << std::endl;
    }
    std::cout << std::endl;
    return;
  }
  std::vector<T> host_data(num);
  cudaMemcpy(host_data.data(), data, num * sizeof(T), cudaMemcpyDeviceToHost);
  for (int i = 0; i < num; i++) {
    std::cout << host_data[i] << ", ";
    if((i+1)%16 == 0)
        std::cout << std::endl;
  }
  std::cout << std::endl;
}


//sync API
__global__ void ShuffleSyncKernel1(unsigned int* data) {
  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  int output = 0;
  unsigned int mask = 0xFFFFFFF0;
  output = __shfl_sync(mask, threadid, threadid + 1, 16);
  data[threadid] = output;
}
__global__ void ShuffleUpSyncKernel1(unsigned int* data) {
  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  int output = 0;
  unsigned int mask = 0xFFFFFFF0;
  output = __shfl_up_sync(mask, threadid, 1, 16);
  data[threadid] = output;
}
__global__ void ShuffleDownSyncKernel1(unsigned int* data) {
  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  int output = 0;
  unsigned int mask = 0xFFFFFFF0;
  output = __shfl_down_sync(mask, threadid, 1, 16);
  data[threadid] = output;
}
__global__ void ShuffleXorSyncKernel1(unsigned int* data) {
  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  int output = 0;
  unsigned int mask = 0xFFFFFFF0;
  output = __shfl_xor_sync(mask, threadid, 2, 16);
  data[threadid] = output;
}

//has branch1
__global__ void ShuffleSyncKernel2(unsigned int* data) {
  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  int output = 0;
  unsigned int mask = 0xFFFFFFF0;
  if(threadid%32 >3) {
    output = __shfl_sync(mask, threadid, threadid + 1, 16);
  }
  data[threadid] = output;
}
__global__ void ShuffleUpSyncKernel2(unsigned int* data) {
  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  int output = 0;
  unsigned int mask = 0xFFFFFFF0;
  if(threadid%32 >3) {
    output = __shfl_up_sync(mask, threadid, 1, 16);
  }
  data[threadid] = output;
}
__global__ void ShuffleDownSyncKernel2(unsigned int* data) {
  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  int output = 0;
  unsigned int mask = 0x0FFFFFFF;
  if(threadid%32 < 28) {
    output = __shfl_down_sync(mask, threadid, 1, 16);
  }
  data[threadid] = output;
}
__global__ void ShuffleXorSyncKernel2(unsigned int* data) {
  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  int output = 0;
  unsigned int mask = 0xFFFFFFF0;
  if(threadid%32 >3) {
    output = __shfl_xor_sync(mask, threadid, 2, 16);
  }
  data[threadid] = output;
}

// has branch 2
__global__ void ShuffleSyncKernel3(unsigned int* data) {
  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  int output = 0;
  unsigned int mask = 0xFFFFFFF0;
  if(threadid%32 >3) {
    unsigned remote = threadid ? threadid - 1 : 0;
    output = __shfl_sync(mask, threadid, remote , 16);
  }
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
  // NV hardware result reference
  unsigned int expect1[DATA_NUM] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 16,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 32, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 48,
    65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 64, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 80,
    97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 96, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 112,
  };
  init_data<unsigned int>(dev_data_u, DATA_NUM);

  ShuffleSyncKernel1<<<GridSize, BlockSize>>>(dev_data_u);

  cudaDeviceSynchronize();
  if(!verify_data<unsigned int>(dev_data_u, expect1, DATA_NUM)) {
    std::cout << "ShuffleSyncKernel1" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data<unsigned int>(expect1, DATA_NUM, true);
    std::cout << "current result:" << std::endl;
    print_data<unsigned int>(dev_data_u, DATA_NUM);
  }

  GridSize = {2};
  BlockSize = {32, 2, 1};
  // NV hardware result reference
  unsigned int expect2[DATA_NUM] = {
    0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    32, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
    64, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 80, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94,
    96, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 112, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126,
  };
  init_data<unsigned int>(dev_data_u, DATA_NUM);

  ShuffleUpSyncKernel1<<<GridSize, BlockSize>>>(dev_data_u);

  cudaDeviceSynchronize();
  if(!verify_data<unsigned int>(dev_data_u, expect2, DATA_NUM)) {
    std::cout << "ShuffleUpSyncKernel1" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data<unsigned int>(expect2, DATA_NUM, true);
    std::cout << "current result:" << std::endl;
    print_data<unsigned int>(dev_data_u, DATA_NUM);
  }
  

  GridSize = {2};
  BlockSize = {32, 2, 1};
  // NV hardware result reference
  unsigned int expect3[DATA_NUM] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 31,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 47, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 63,
    65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 79, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 95,
    97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 111, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 127,
  };
  init_data<unsigned int>(dev_data_u, DATA_NUM);

  ShuffleDownSyncKernel1<<<GridSize, BlockSize>>>(dev_data_u);

  cudaDeviceSynchronize();
  if(!verify_data<unsigned int>(dev_data_u, expect3, DATA_NUM)) {
    std::cout << "ShuffleDownSyncKernel1" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data<unsigned int>(expect3, DATA_NUM, true);
    std::cout << "current result:" << std::endl;
    print_data<unsigned int>(dev_data_u, DATA_NUM);
  }

  GridSize = {2};
  BlockSize = {32, 2, 1};
  // NV hardware result reference
  unsigned int expect4[DATA_NUM] = {
    2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13,18,19,16,17,22,23,20,21,26,27,24,25,30,31,
    28,29,34,35,32,33,38,39,36,37,42,43,40,41,46,47,44,45,50,51,48,49,54,55,52,53,58,
    59,56,57,62,63,60,61,66,67,64,65,70,71,68,69,74,75,72,73,78,79,76,77,82,83,80,81,
    86,87,84,85,90,91,88,89,94,95,92,93,98,99,96,97,102,103,100,101,106,107,104,105,
    110,111,108,109,114,115,112,113,118,119,116,117,122,123,120,121,126,127,124,125
  };
  init_data<unsigned int>(dev_data_u, DATA_NUM);

  ShuffleXorSyncKernel1<<<GridSize, BlockSize>>>(dev_data_u);

  cudaDeviceSynchronize();
  if(!verify_data<unsigned int>(dev_data_u, expect4, DATA_NUM)) {
    std::cout << "ShuffleXorSyncKernel1" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data<unsigned int>(expect4, DATA_NUM, true);
    std::cout << "current result:" << std::endl;
    print_data<unsigned int>(dev_data_u, DATA_NUM);
  }

// has branch 1
GridSize = {2};
BlockSize = {32, 2, 1};
  // NV hardware result reference
unsigned int expect5[DATA_NUM] = {
  0, 0, 0, 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 16,
  0, 0, 0, 0, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 0, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 48,
  0, 0, 0, 0, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 0, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 80,
  0, 0, 0, 0, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 0, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 112,
};
init_data<unsigned int>(dev_data_u, DATA_NUM);

ShuffleSyncKernel2<<<GridSize, BlockSize>>>(dev_data_u);

cudaDeviceSynchronize();
if(!verify_data<unsigned int>(dev_data_u, expect5, DATA_NUM)) {
  std::cout << "ShuffleSyncKernel2" << " verify failed" << std::endl;
  Result = false;
  std::cout << "expect:" << std::endl;
  print_data<unsigned int>(expect5, DATA_NUM, true);
  std::cout << "current result:" << std::endl;
  print_data<unsigned int>(dev_data_u, DATA_NUM);
}

GridSize = {2};
BlockSize = {32, 2, 1};
  // NV hardware result reference
  // The result[5/37/69/101] of _shfl_up function in delta 4 and logical warp size 16 is undefined.
  // But the SYCL version return 3/35/67/99, so we change these 4 number in reference to result of
  // SYCL version function.
unsigned int expect6[DATA_NUM] = {
  0, 0, 0, 0, 3/*0 -> 3*/, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
  0, 0, 0, 0, 35/*0 -> 35*/, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
  0, 0, 0, 0, 67/*0 -> 67*/, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 80, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94,
  0, 0, 0, 0, 99/*0 -> 99*/, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 112, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126,
};
init_data<unsigned int>(dev_data_u, DATA_NUM);

ShuffleUpSyncKernel2<<<GridSize, BlockSize>>>(dev_data_u);

cudaDeviceSynchronize();
if(!verify_data<unsigned int>(dev_data_u, expect6, DATA_NUM)) {
  std::cout << "ShuffleUpSyncKernel2" << " verify failed" << std::endl;
  Result = false;
  std::cout << "expect:" << std::endl;
  print_data<unsigned int>(expect6, DATA_NUM, true);
  std::cout << "current result:" << std::endl;
  print_data<unsigned int>(dev_data_u, DATA_NUM);
}


GridSize = {2};
BlockSize = {32, 2, 1};
  // NV hardware result reference
  // The result[27/59/91/123] of _shfl_down function in delta 4 and logical warp size 16 is undefined.
  // But the SYCL version return 28/60/92/124, so we change these 4 number in reference to result of
  // SYCL version function.
unsigned int expect7[DATA_NUM] = {
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28/*0 -> 28*/, 0, 0, 0, 0,
  33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 47, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60/*0 -> 60*/, 0, 0, 0, 0,
  65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 79, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92/*0 -> 92*/, 0, 0, 0, 0,
  97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 111, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124/*0 -> 124*/, 0, 0, 0, 0,
};
init_data<unsigned int>(dev_data_u, DATA_NUM);

ShuffleDownSyncKernel2<<<GridSize, BlockSize>>>(dev_data_u);

cudaDeviceSynchronize();
if(!verify_data<unsigned int>(dev_data_u, expect7, DATA_NUM)) {
  std::cout << "ShuffleDownSyncKernel2" << " verify failed" << std::endl;
  Result = false;
  std::cout << "expect:" << std::endl;
  print_data<unsigned int>(expect7, DATA_NUM, true);
  std::cout << "current result:" << std::endl;
  print_data<unsigned int>(dev_data_u, DATA_NUM);
}

GridSize = {2};
BlockSize = {32, 2, 1};
  // NV hardware result reference
unsigned int expect8[DATA_NUM] = {
  0, 0, 0, 0, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13, 18, 19, 16, 17, 22, 23, 20, 21, 26, 27, 24, 25, 30, 31, 28, 29,
  0, 0, 0, 0, 38, 39, 36, 37, 42, 43, 40, 41, 46, 47, 44, 45, 50, 51, 48, 49, 54, 55, 52, 53, 58, 59, 56, 57, 62, 63, 60, 61,
  0, 0, 0, 0, 70, 71, 68, 69, 74, 75, 72, 73, 78, 79, 76, 77, 82, 83, 80, 81, 86, 87, 84, 85, 90, 91, 88, 89, 94, 95, 92, 93,
  0, 0, 0, 0, 102, 103, 100, 101, 106, 107, 104, 105, 110, 111, 108, 109, 114, 115, 112, 113, 118, 119, 116, 117, 122, 123, 120, 121, 126, 127, 124, 125,
};
init_data<unsigned int>(dev_data_u, DATA_NUM);

ShuffleXorSyncKernel2<<<GridSize, BlockSize>>>(dev_data_u);

cudaDeviceSynchronize();
if(!verify_data<unsigned int>(dev_data_u, expect8, DATA_NUM)) {
  std::cout << "ShuffleXorSyncKernel2" << " verify failed" << std::endl;
  Result = false;
  std::cout << "expect:" << std::endl;
  print_data<unsigned int>(expect8, DATA_NUM, true);
  std::cout << "current result:" << std::endl;
  print_data<unsigned int>(dev_data_u, DATA_NUM);
}

// has branch 2

GridSize = {2};
BlockSize = {32, 2, 1};
  // NV hardware result reference
unsigned int expect9[DATA_NUM] = {
  0, 0, 0, 0, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 31, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
  0, 0, 0, 0, 0, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 63, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
  0, 0, 0, 0, 0, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 95, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94,
  0, 0, 0, 0, 0, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 127, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126,
};
init_data<unsigned int>(dev_data_u, DATA_NUM);

ShuffleSyncKernel3<<<GridSize, BlockSize>>>(dev_data_u);

cudaDeviceSynchronize();
if(!verify_data<unsigned int>(dev_data_u, expect9, DATA_NUM)) {
  std::cout << "ShuffleSyncKernel3" << " verify failed" << std::endl;
  Result = false;
  std::cout << "expect:" << std::endl;
  print_data<unsigned int>(expect9, DATA_NUM, true);
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


