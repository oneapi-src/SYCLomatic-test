// ====------ cudnn-scale.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <vector>

template<cudnnDataType_t T>
struct dt_trait{
    typedef void type;
};
template<>
struct dt_trait<CUDNN_DATA_FLOAT>{
    typedef float type;
};

template<>
struct dt_trait<CUDNN_DATA_INT32>{
    typedef int type;
};
template<>
struct dt_trait<CUDNN_DATA_HALF>{
    typedef float type;
};

template<typename T>
void check(std::vector<T> &expect, std::vector<T> &actual, int num, float precision) {
  for(int i = 0; i < num; i++){
      if(std::abs(expect[i] - actual[i]) > precision) {
          std::cout << "test failed" << std::endl;
          std::cout << "expect:" << expect[i] << std::endl;
          std::cout << "actual:" << actual[i] << std::endl;
          exit(-1);
      }
  }
}

template<cudnnDataType_t T, typename HT = typename dt_trait<T>::type>
void test() {

    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor;

    cudnnCreate(&handle);



    cudnnCreateTensorDescriptor(&dataTensor);
    int n = 1, c = 2, h = 5, w = 5;
    int ele_num = n * c * h * w;

    cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, T, n, c, h, w);

    HT *data;
    std::vector<HT> host_data(ele_num);

    for(int i = 0; i < ele_num; i++) {
        host_data[i] = i;
    }

    cudaMalloc(&data, ele_num * sizeof(HT));

    cudaMemcpy(data, host_data.data(), ele_num * sizeof(HT), cudaMemcpyHostToDevice);

    float alpha = 3.f;
    auto s = cudnnScaleTensor(handle, dataTensor, data, &alpha);
    cudaDeviceSynchronize();
    cudaMemcpy(host_data.data(), data, ele_num * sizeof(HT), cudaMemcpyDeviceToHost);

    std::vector<float> expect = {
        0, 3, 6, 9, 12,
        15, 18, 21, 24, 27,
        30, 33, 36, 39, 42,
        45, 48, 51, 54, 57,
        60, 63, 66, 69, 72,
        75, 78, 81, 84, 87,
        90, 93, 96, 99, 102,
        105, 108, 111, 114, 117,
        120, 123, 126, 129, 132,
        135, 138, 141, 144, 147
      };
    check(expect, host_data, expect.size(), 1e-3);
    cudnnDestroy(handle);
    cudaFree(data);
}

int main() {
    test<CUDNN_DATA_FLOAT>();
    std::cout << "test passed" << std::endl;
    return 0;
}