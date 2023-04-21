// ====------ cub_warp.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <iostream>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define WARP_SIZE 32
#define DATA_NUM 256

void init_data(int* data, int num) {
  for(int i = 0; i < num; i++)
    data[i] = i;
}
bool verify_data(int* data, int* expect, int num, int step = 1) {
  for(int i = 0; i < num; i = i + step) {
    if(data[i] != expect[i]) {
      return false;
    }
  }
  return true;
}
void print_data(int* data, int num) {
  for (int i = 0; i < num; i++) {
    std::cout << data[i] << ", ";
    if((i+1)%32 == 0)
        std::cout << std::endl;
  }
  std::cout << std::endl;
}


__global__ void ShuffleIndexKernel1(int* data) {

  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  int input = data[threadid];
  int output = 0;
  output = cub::ShuffleIndex<32>(input, 0, 0xffffffff);
  data[threadid] = output;
}

__global__ void ExclusiveScanKernel1(int* data) {
  typedef cub::WarpScan<int> WarpScan;

  __shared__ typename WarpScan::TempStorage temp1;

  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;

  int input = data[threadid];
  int output = 0;
  WarpScan(temp1).ExclusiveScan(input, output, cub::Sum());
  data[threadid] = output;
}

__global__ void ExclusiveScanKernel2(int* data) {
  typedef cub::WarpScan<int> WarpScan;

  __shared__ typename WarpScan::TempStorage temp1[10];

  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;

  int input = data[threadid];
  int output = 0;
  WarpScan(temp1[0]).ExclusiveScan(input, output, 0, cub::Sum());
  data[threadid] = output;
}

__global__ void InclusiveScanKernel(int* data) {
  typedef cub::WarpScan<int> WarpScan;

  __shared__ typename WarpScan::TempStorage temp1;

  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;

  int input = data[threadid];
  int output = 0;
  WarpScan(temp1).InclusiveScan(input, output, cub::Sum());
  data[threadid] = output;
}

__global__ void ExclusiveSumKernel(int* data) {
  typedef cub::WarpScan<int> WarpScan;

  __shared__ typename WarpScan::TempStorage temp1;

  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;

  int input = data[threadid];
  int output = 0;
  WarpScan(temp1).ExclusiveSum(input, output);
  data[threadid] = output;
}

__global__ void InclusiveSumKernel(int* data) {
  typedef cub::WarpScan<int> WarpScan;

  __shared__ typename WarpScan::TempStorage temp1;

  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;

  int input = data[threadid];
  int output = 0;
  WarpScan(temp1).InclusiveSum(input, output);
  data[threadid] = output;
}

__global__ void BroadcastKernel(int* data) {
  typedef cub::WarpScan<int> WarpScan;

  __shared__ typename WarpScan::TempStorage temp1;

  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;

  int input = data[threadid];
  int output = 0;
  output = WarpScan(temp1).Broadcast(input, 0);
  data[threadid] = output;
}

template<typename ScanTy, typename DataTy>
__device__ DataTy Scan1(ScanTy &s, DataTy data) {
  DataTy out;
  s.InclusiveSum(data, out);
  return out;
}

__global__ void TemplateKernel1(int* data) {
  typedef cub::WarpScan<int> WarpScan;

  typename WarpScan::TempStorage temp1;
  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  WarpScan ws(temp1);
  data[threadid] = Scan1<WarpScan, int>(ws, data[threadid]);
}

__global__ void SumKernel(int* data) {
  typedef cub::WarpReduce<int> WarpReduce;

  __shared__ typename WarpReduce::TempStorage temp1;

  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;

  int input = data[threadid];
  int output = 0;
  output = WarpReduce(temp1).Sum(input);
  data[threadid] = output;
}

__global__ void ReduceKernel(int* data) {
  typedef cub::WarpReduce<int> WarpReduce;

  __shared__ typename WarpReduce::TempStorage temp1;

  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;

  int input = data[threadid];
  int output = 0;
  output = WarpReduce(temp1).Reduce(input, cub::Sum());
  data[threadid] = output;
}

__global__ void ReduceValidKernel(int* data, int valid_items) {
  typedef cub::WarpReduce<int> WarpReduce;

  __shared__ typename WarpReduce::TempStorage temp1;

  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;

  int input = data[threadid];
  int output = 0;
  output = WarpReduce(temp1).Reduce(input, cub::Sum(), valid_items);
  data[threadid] = output;
}

__global__ void ThreadLoadKernel(int* data) {
  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;

  int input = data[threadid];
  int output = 0;
  output = cub::ThreadLoad<cub::LOAD_CA>(data + threadid);
  data[threadid] = output;
}

__global__ void ThreadStoreKernel(int* data) {
  int threadid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;

  int input = data[threadid];
  int output = threadid;
  cub::ThreadStore<cub::STORE_CG>(data + threadid, output);
}

int main() {
  bool Result = true;
  int* dev_data = nullptr;
  dim3 GridSize;
  dim3 BlockSize;
  cudaMallocManaged(&dev_data, DATA_NUM * sizeof(int));
  GridSize = {2};
  BlockSize = {16, 8, 1};
  int expect1[DATA_NUM] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96,
    128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160,
    192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192,
    224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224}; 
  init_data(dev_data, DATA_NUM);

  ShuffleIndexKernel1<<<GridSize, BlockSize>>>(dev_data);

  cudaDeviceSynchronize();
  if(!verify_data(dev_data, expect1, DATA_NUM)) {
    std::cout << "ShuffleIndexKernel1" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data(expect1, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data(dev_data, DATA_NUM);
  }

  GridSize = {2};
  BlockSize = {16, 8, 1};
  int expect2[DATA_NUM] = {
    0, 0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190, 210, 231, 253, 276, 300, 325, 351, 378, 406, 435, 465,
    0, 32, 65, 99, 134, 170, 207, 245, 284, 324, 365, 407, 450, 494, 539, 585, 632, 680, 729, 779, 830, 882, 935, 989, 1044, 1100, 1157, 1215, 1274, 1334, 1395, 1457,
    0, 64, 129, 195, 262, 330, 399, 469, 540, 612, 685, 759, 834, 910, 987, 1065, 1144, 1224, 1305, 1387, 1470, 1554, 1639, 1725, 1812, 1900, 1989, 2079, 2170, 2262, 2355, 2449,
    0, 96, 193, 291, 390, 490, 591, 693, 796, 900, 1005, 1111, 1218, 1326, 1435, 1545, 1656, 1768, 1881, 1995, 2110, 2226, 2343, 2461, 2580, 2700, 2821, 2943, 3066, 3190, 3315, 3441,
    0, 128, 257, 387, 518, 650, 783, 917, 1052, 1188, 1325, 1463, 1602, 1742, 1883, 2025, 2168, 2312, 2457, 2603, 2750, 2898, 3047, 3197, 3348, 3500, 3653, 3807, 3962, 4118, 4275, 4433,
    0, 160, 321, 483, 646, 810, 975, 1141, 1308, 1476, 1645, 1815, 1986, 2158, 2331, 2505, 2680, 2856, 3033, 3211, 3390, 3570, 3751, 3933, 4116, 4300, 4485, 4671, 4858, 5046, 5235, 5425,
    0, 192, 385, 579, 774, 970, 1167, 1365, 1564, 1764, 1965, 2167, 2370, 2574, 2779, 2985, 3192, 3400, 3609, 3819, 4030, 4242, 4455, 4669, 4884, 5100, 5317, 5535, 5754, 5974, 6195, 6417,
    0, 224, 449, 675, 902, 1130, 1359, 1589, 1820, 2052, 2285, 2519, 2754, 2990, 3227, 3465, 3704, 3944, 4185, 4427, 4670, 4914, 5159, 5405, 5652, 5900, 6149, 6399, 6650, 6902, 7155, 7409
  };
  init_data(dev_data, DATA_NUM);

  ExclusiveScanKernel1<<<GridSize, BlockSize>>>(dev_data);

  cudaDeviceSynchronize();
  if(!verify_data(dev_data, expect2, DATA_NUM)) {
    std::cout << "ExclusiveScanKernel1" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data(expect2, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data(dev_data, DATA_NUM);
  }
  
  GridSize = {2};
  BlockSize = {16, 8, 1};
  int expect3[DATA_NUM] = {
    0, 0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190, 210, 231, 253, 276, 300, 325, 351, 378, 406, 435, 465,
    0, 32, 65, 99, 134, 170, 207, 245, 284, 324, 365, 407, 450, 494, 539, 585, 632, 680, 729, 779, 830, 882, 935, 989, 1044, 1100, 1157, 1215, 1274, 1334, 1395, 1457,
    0, 64, 129, 195, 262, 330, 399, 469, 540, 612, 685, 759, 834, 910, 987, 1065, 1144, 1224, 1305, 1387, 1470, 1554, 1639, 1725, 1812, 1900, 1989, 2079, 2170, 2262, 2355, 2449,
    0, 96, 193, 291, 390, 490, 591, 693, 796, 900, 1005, 1111, 1218, 1326, 1435, 1545, 1656, 1768, 1881, 1995, 2110, 2226, 2343, 2461, 2580, 2700, 2821, 2943, 3066, 3190, 3315, 3441,
    0, 128, 257, 387, 518, 650, 783, 917, 1052, 1188, 1325, 1463, 1602, 1742, 1883, 2025, 2168, 2312, 2457, 2603, 2750, 2898, 3047, 3197, 3348, 3500, 3653, 3807, 3962, 4118, 4275, 4433,
    0, 160, 321, 483, 646, 810, 975, 1141, 1308, 1476, 1645, 1815, 1986, 2158, 2331, 2505, 2680, 2856, 3033, 3211, 3390, 3570, 3751, 3933, 4116, 4300, 4485, 4671, 4858, 5046, 5235, 5425,
    0, 192, 385, 579, 774, 970, 1167, 1365, 1564, 1764, 1965, 2167, 2370, 2574, 2779, 2985, 3192, 3400, 3609, 3819, 4030, 4242, 4455, 4669, 4884, 5100, 5317, 5535, 5754, 5974, 6195, 6417,
    0, 224, 449, 675, 902, 1130, 1359, 1589, 1820, 2052, 2285, 2519, 2754, 2990, 3227, 3465, 3704, 3944, 4185, 4427, 4670, 4914, 5159, 5405, 5652, 5900, 6149, 6399, 6650, 6902, 7155, 7409
  };
  init_data(dev_data, DATA_NUM);

  ExclusiveScanKernel2<<<GridSize, BlockSize>>>(dev_data);

  cudaDeviceSynchronize();
  if(!verify_data(dev_data, expect3, DATA_NUM)) {
    std::cout << "ExclusiveScanKernel2" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data(expect3, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data(dev_data, DATA_NUM);
  }

  GridSize = {2};
  BlockSize = {16, 8, 1};
  int expect4[DATA_NUM] = {
    0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190, 210, 231, 253, 276, 300, 325, 351, 378, 406, 435, 465, 496,
    32, 65, 99, 134, 170, 207, 245, 284, 324, 365, 407, 450, 494, 539, 585, 632, 680, 729, 779, 830, 882, 935, 989, 1044, 1100, 1157, 1215, 1274, 1334, 1395, 1457, 1520,
    64, 129, 195, 262, 330, 399, 469, 540, 612, 685, 759, 834, 910, 987, 1065, 1144, 1224, 1305, 1387, 1470, 1554, 1639, 1725, 1812, 1900, 1989, 2079, 2170, 2262, 2355, 2449, 2544,
    96, 193, 291, 390, 490, 591, 693, 796, 900, 1005, 1111, 1218, 1326, 1435, 1545, 1656, 1768, 1881, 1995, 2110, 2226, 2343, 2461, 2580, 2700, 2821, 2943, 3066, 3190, 3315, 3441, 3568,
    128, 257, 387, 518, 650, 783, 917, 1052, 1188, 1325, 1463, 1602, 1742, 1883, 2025, 2168, 2312, 2457, 2603, 2750, 2898, 3047, 3197, 3348, 3500, 3653, 3807, 3962, 4118, 4275, 4433, 4592,
    160, 321, 483, 646, 810, 975, 1141, 1308, 1476, 1645, 1815, 1986, 2158, 2331, 2505, 2680, 2856, 3033, 3211, 3390, 3570, 3751, 3933, 4116, 4300, 4485, 4671, 4858, 5046, 5235, 5425, 5616,
    192, 385, 579, 774, 970, 1167, 1365, 1564, 1764, 1965, 2167, 2370, 2574, 2779, 2985, 3192, 3400, 3609, 3819, 4030, 4242, 4455, 4669, 4884, 5100, 5317, 5535, 5754, 5974, 6195, 6417, 6640,
    224, 449, 675, 902, 1130, 1359, 1589, 1820, 2052, 2285, 2519, 2754, 2990, 3227, 3465, 3704, 3944, 4185, 4427, 4670, 4914, 5159, 5405, 5652, 5900, 6149, 6399, 6650, 6902, 7155, 7409, 7664
  };
  init_data(dev_data, DATA_NUM);

  InclusiveScanKernel<<<GridSize, BlockSize>>>(dev_data);

  cudaDeviceSynchronize();
  if(!verify_data(dev_data, expect4, DATA_NUM)) {
    std::cout << "InclusiveScanKernel" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data(expect4, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data(dev_data, DATA_NUM);
  }

  GridSize = {2};
  BlockSize = {16, 8, 1};
  int expect5[DATA_NUM] = {
    0, 0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190, 210, 231, 253, 276, 300, 325, 351, 378, 406, 435, 465,
    0, 32, 65, 99, 134, 170, 207, 245, 284, 324, 365, 407, 450, 494, 539, 585, 632, 680, 729, 779, 830, 882, 935, 989, 1044, 1100, 1157, 1215, 1274, 1334, 1395, 1457,
    0, 64, 129, 195, 262, 330, 399, 469, 540, 612, 685, 759, 834, 910, 987, 1065, 1144, 1224, 1305, 1387, 1470, 1554, 1639, 1725, 1812, 1900, 1989, 2079, 2170, 2262, 2355, 2449,
    0, 96, 193, 291, 390, 490, 591, 693, 796, 900, 1005, 1111, 1218, 1326, 1435, 1545, 1656, 1768, 1881, 1995, 2110, 2226, 2343, 2461, 2580, 2700, 2821, 2943, 3066, 3190, 3315, 3441,
    0, 128, 257, 387, 518, 650, 783, 917, 1052, 1188, 1325, 1463, 1602, 1742, 1883, 2025, 2168, 2312, 2457, 2603, 2750, 2898, 3047, 3197, 3348, 3500, 3653, 3807, 3962, 4118, 4275, 4433,
    0, 160, 321, 483, 646, 810, 975, 1141, 1308, 1476, 1645, 1815, 1986, 2158, 2331, 2505, 2680, 2856, 3033, 3211, 3390, 3570, 3751, 3933, 4116, 4300, 4485, 4671, 4858, 5046, 5235, 5425,
    0, 192, 385, 579, 774, 970, 1167, 1365, 1564, 1764, 1965, 2167, 2370, 2574, 2779, 2985, 3192, 3400, 3609, 3819, 4030, 4242, 4455, 4669, 4884, 5100, 5317, 5535, 5754, 5974, 6195, 6417,
    0, 224, 449, 675, 902, 1130, 1359, 1589, 1820, 2052, 2285, 2519, 2754, 2990, 3227, 3465, 3704, 3944, 4185, 4427, 4670, 4914, 5159, 5405, 5652, 5900, 6149, 6399, 6650, 6902, 7155, 7409
  };
  init_data(dev_data, DATA_NUM);

  ExclusiveSumKernel<<<GridSize, BlockSize>>>(dev_data);

  cudaDeviceSynchronize();
  if(!verify_data(dev_data, expect5, DATA_NUM)) {
    std::cout << "ExclusiveSumKernel" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data(expect5, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data(dev_data, DATA_NUM);
  }

  GridSize = {2};
  BlockSize = {16, 8, 1};
  int expect6[DATA_NUM] = {
    0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190, 210, 231, 253, 276, 300, 325, 351, 378, 406, 435, 465, 496,
    32, 65, 99, 134, 170, 207, 245, 284, 324, 365, 407, 450, 494, 539, 585, 632, 680, 729, 779, 830, 882, 935, 989, 1044, 1100, 1157, 1215, 1274, 1334, 1395, 1457, 1520,
    64, 129, 195, 262, 330, 399, 469, 540, 612, 685, 759, 834, 910, 987, 1065, 1144, 1224, 1305, 1387, 1470, 1554, 1639, 1725, 1812, 1900, 1989, 2079, 2170, 2262, 2355, 2449, 2544,
    96, 193, 291, 390, 490, 591, 693, 796, 900, 1005, 1111, 1218, 1326, 1435, 1545, 1656, 1768, 1881, 1995, 2110, 2226, 2343, 2461, 2580, 2700, 2821, 2943, 3066, 3190, 3315, 3441, 3568,
    128, 257, 387, 518, 650, 783, 917, 1052, 1188, 1325, 1463, 1602, 1742, 1883, 2025, 2168, 2312, 2457, 2603, 2750, 2898, 3047, 3197, 3348, 3500, 3653, 3807, 3962, 4118, 4275, 4433, 4592,
    160, 321, 483, 646, 810, 975, 1141, 1308, 1476, 1645, 1815, 1986, 2158, 2331, 2505, 2680, 2856, 3033, 3211, 3390, 3570, 3751, 3933, 4116, 4300, 4485, 4671, 4858, 5046, 5235, 5425, 5616,
    192, 385, 579, 774, 970, 1167, 1365, 1564, 1764, 1965, 2167, 2370, 2574, 2779, 2985, 3192, 3400, 3609, 3819, 4030, 4242, 4455, 4669, 4884, 5100, 5317, 5535, 5754, 5974, 6195, 6417, 6640,
    224, 449, 675, 902, 1130, 1359, 1589, 1820, 2052, 2285, 2519, 2754, 2990, 3227, 3465, 3704, 3944, 4185, 4427, 4670, 4914, 5159, 5405, 5652, 5900, 6149, 6399, 6650, 6902, 7155, 7409, 7664
  };
  init_data(dev_data, DATA_NUM);

  InclusiveSumKernel<<<GridSize, BlockSize>>>(dev_data);

  cudaDeviceSynchronize();
  if(!verify_data(dev_data, expect6, DATA_NUM)) {
    std::cout << "InclusiveSumKernel" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data(expect6, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data(dev_data, DATA_NUM);
  }

  GridSize = {2};
  BlockSize = {16, 8, 1};
  int expect7[DATA_NUM] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96,
    128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160,
    192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192,
    224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224}; 
  init_data(dev_data, DATA_NUM);

  BroadcastKernel<<<GridSize, BlockSize>>>(dev_data);

  cudaDeviceSynchronize();
  if(!verify_data(dev_data, expect7, DATA_NUM)) {
    std::cout << "BroadcastKernel" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data(expect7, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data(dev_data, DATA_NUM);
  }

  GridSize = {2};
  BlockSize = {16, 8, 1};
  int expect8[DATA_NUM] = {
    0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190, 210, 231, 253, 276, 300, 325, 351, 378, 406, 435, 465, 496,
    32, 65, 99, 134, 170, 207, 245, 284, 324, 365, 407, 450, 494, 539, 585, 632, 680, 729, 779, 830, 882, 935, 989, 1044, 1100, 1157, 1215, 1274, 1334, 1395, 1457, 1520,
    64, 129, 195, 262, 330, 399, 469, 540, 612, 685, 759, 834, 910, 987, 1065, 1144, 1224, 1305, 1387, 1470, 1554, 1639, 1725, 1812, 1900, 1989, 2079, 2170, 2262, 2355, 2449, 2544,
    96, 193, 291, 390, 490, 591, 693, 796, 900, 1005, 1111, 1218, 1326, 1435, 1545, 1656, 1768, 1881, 1995, 2110, 2226, 2343, 2461, 2580, 2700, 2821, 2943, 3066, 3190, 3315, 3441, 3568,
    128, 257, 387, 518, 650, 783, 917, 1052, 1188, 1325, 1463, 1602, 1742, 1883, 2025, 2168, 2312, 2457, 2603, 2750, 2898, 3047, 3197, 3348, 3500, 3653, 3807, 3962, 4118, 4275, 4433, 4592,
    160, 321, 483, 646, 810, 975, 1141, 1308, 1476, 1645, 1815, 1986, 2158, 2331, 2505, 2680, 2856, 3033, 3211, 3390, 3570, 3751, 3933, 4116, 4300, 4485, 4671, 4858, 5046, 5235, 5425, 5616,
    192, 385, 579, 774, 970, 1167, 1365, 1564, 1764, 1965, 2167, 2370, 2574, 2779, 2985, 3192, 3400, 3609, 3819, 4030, 4242, 4455, 4669, 4884, 5100, 5317, 5535, 5754, 5974, 6195, 6417, 6640,
    224, 449, 675, 902, 1130, 1359, 1589, 1820, 2052, 2285, 2519, 2754, 2990, 3227, 3465, 3704, 3944, 4185, 4427, 4670, 4914, 5159, 5405, 5652, 5900, 6149, 6399, 6650, 6902, 7155, 7409, 7664
  };
  init_data(dev_data, DATA_NUM);

  TemplateKernel1<<<GridSize, BlockSize>>>(dev_data);

  cudaDeviceSynchronize();
  if(!verify_data(dev_data, expect8, DATA_NUM)) {
    std::cout << "TemplateKernel1" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data(expect8, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data(dev_data, DATA_NUM);
  }


  GridSize = {2};
  BlockSize = {16, 8, 1};
  int expect9[DATA_NUM] = {
    496, 496, 495, 493, 490, 486, 481, 475, 468, 460, 451, 441, 430, 418, 405, 391, 376, 360, 343, 325, 306, 286, 265, 243, 220, 196, 171, 145, 118, 90, 61, 31,
    1520, 1488, 1455, 1421, 1386, 1350, 1313, 1275, 1236, 1196, 1155, 1113, 1070, 1026, 981, 935, 888, 840, 791, 741, 690, 638, 585, 531, 476, 420, 363, 305, 246, 186, 125, 63,
    2544, 2480, 2415, 2349, 2282, 2214, 2145, 2075, 2004, 1932, 1859, 1785, 1710, 1634, 1557, 1479, 1400, 1320, 1239, 1157, 1074, 990, 905, 819, 732, 644, 555, 465, 374, 282, 189, 95,
    3568, 3472, 3375, 3277, 3178, 3078, 2977, 2875, 2772, 2668, 2563, 2457, 2350, 2242, 2133, 2023, 1912, 1800, 1687, 1573, 1458, 1342, 1225, 1107, 988, 868, 747, 625, 502, 378, 253, 127,
    4592, 4464, 4335, 4205, 4074, 3942, 3809, 3675, 3540, 3404, 3267, 3129, 2990, 2850, 2709, 2567, 2424, 2280, 2135, 1989, 1842, 1694, 1545, 1395, 1244, 1092, 939, 785, 630, 474, 317, 159,
    5616, 5456, 5295, 5133, 4970, 4806, 4641, 4475, 4308, 4140, 3971, 3801, 3630, 3458, 3285, 3111, 2936, 2760, 2583, 2405, 2226, 2046, 1865, 1683, 1500, 1316, 1131, 945, 758, 570, 381, 191,
    6640, 6448, 6255, 6061, 5866, 5670, 5473, 5275, 5076, 4876, 4675, 4473, 4270, 4066, 3861, 3655, 3448, 3240, 3031, 2821, 2610, 2398, 2185, 1971, 1756, 1540, 1323, 1105, 886, 666, 445, 223,
    7664, 7440, 7215, 6989, 6762, 6534, 6305, 6075, 5844, 5612, 5379, 5145, 4910, 4674, 4437, 4199, 3960, 3720, 3479, 3237, 2994, 2750, 2505, 2259, 2012, 1764, 1515, 1265, 1014, 762, 509, 255,
  };
  init_data(dev_data, DATA_NUM);

  SumKernel<<<GridSize, BlockSize>>>(dev_data);

  cudaDeviceSynchronize();
  if(!verify_data(dev_data, expect9, DATA_NUM, 32)) {
    std::cout << "SumKernel" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data(expect9, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data(dev_data, DATA_NUM);
  }


  GridSize = {2};
  BlockSize = {16, 8, 1};
  int expect10[DATA_NUM] = {
    496, 496, 495, 493, 490, 486, 481, 475, 468, 460, 451, 441, 430, 418, 405, 391, 376, 360, 343, 325, 306, 286, 265, 243, 220, 196, 171, 145, 118, 90, 61, 31,
    1520, 1488, 1455, 1421, 1386, 1350, 1313, 1275, 1236, 1196, 1155, 1113, 1070, 1026, 981, 935, 888, 840, 791, 741, 690, 638, 585, 531, 476, 420, 363, 305, 246, 186, 125, 63,
    2544, 2480, 2415, 2349, 2282, 2214, 2145, 2075, 2004, 1932, 1859, 1785, 1710, 1634, 1557, 1479, 1400, 1320, 1239, 1157, 1074, 990, 905, 819, 732, 644, 555, 465, 374, 282, 189, 95,
    3568, 3472, 3375, 3277, 3178, 3078, 2977, 2875, 2772, 2668, 2563, 2457, 2350, 2242, 2133, 2023, 1912, 1800, 1687, 1573, 1458, 1342, 1225, 1107, 988, 868, 747, 625, 502, 378, 253, 127,
    4592, 4464, 4335, 4205, 4074, 3942, 3809, 3675, 3540, 3404, 3267, 3129, 2990, 2850, 2709, 2567, 2424, 2280, 2135, 1989, 1842, 1694, 1545, 1395, 1244, 1092, 939, 785, 630, 474, 317, 159,
    5616, 5456, 5295, 5133, 4970, 4806, 4641, 4475, 4308, 4140, 3971, 3801, 3630, 3458, 3285, 3111, 2936, 2760, 2583, 2405, 2226, 2046, 1865, 1683, 1500, 1316, 1131, 945, 758, 570, 381, 191,
    6640, 6448, 6255, 6061, 5866, 5670, 5473, 5275, 5076, 4876, 4675, 4473, 4270, 4066, 3861, 3655, 3448, 3240, 3031, 2821, 2610, 2398, 2185, 1971, 1756, 1540, 1323, 1105, 886, 666, 445, 223,
    7664, 7440, 7215, 6989, 6762, 6534, 6305, 6075, 5844, 5612, 5379, 5145, 4910, 4674, 4437, 4199, 3960, 3720, 3479, 3237, 2994, 2750, 2505, 2259, 2012, 1764, 1515, 1265, 1014, 762, 509, 255
  };
  init_data(dev_data, DATA_NUM);

  ReduceKernel<<<GridSize, BlockSize>>>(dev_data);

  cudaDeviceSynchronize();
  if(!verify_data(dev_data, expect10, DATA_NUM, 32)) {
    std::cout << "ReduceKernel" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data(expect10, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data(dev_data, DATA_NUM);
  }

  GridSize = {2};
  BlockSize = {16, 8, 1};
  int valid_items = 4;
  int expect_valid10[1] = {
    6
  };
  init_data(dev_data, DATA_NUM);

  ReduceValidKernel<<<GridSize, BlockSize>>>(dev_data, valid_items);

  cudaDeviceSynchronize();
  if(!verify_data(dev_data, expect_valid10, 1, 1)) {
    std::cout << "ReduceValidKernel" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data(expect10, 1);
    std::cout << "current result:" << std::endl;
    print_data(dev_data, 1);
  }

  GridSize = {2};
  BlockSize = {16, 8, 1};
  int expect11[DATA_NUM] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
    96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
    128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
    160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
    192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
    224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255
  };
  init_data(dev_data, DATA_NUM);

  ThreadLoadKernel<<<GridSize, BlockSize>>>(dev_data);

  cudaDeviceSynchronize();
  if(!verify_data(dev_data, expect11, DATA_NUM)) {
    std::cout << "ThreadLoadKernel" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data(expect11, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data(dev_data, DATA_NUM);
  }

  GridSize = {2};
  BlockSize = {16, 8, 1};
  int expect12[DATA_NUM] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
    96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
    128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
    160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
    192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
    224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255
  };
  init_data(dev_data, DATA_NUM);

  ThreadStoreKernel<<<GridSize, BlockSize>>>(dev_data);

  cudaDeviceSynchronize();
  if(!verify_data(dev_data, expect12, DATA_NUM)) {
    std::cout << "ThreadStoreKernel" << " verify failed" << std::endl;
    Result = false;
    std::cout << "expect:" << std::endl;
    print_data(expect12, DATA_NUM);
    std::cout << "current result:" << std::endl;
    print_data(dev_data, DATA_NUM);
  }

  if(Result) {
    std::cout << "passed" << std::endl;
    return 0;
  }
  return 1;
}

