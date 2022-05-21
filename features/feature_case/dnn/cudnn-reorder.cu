// ====------ cudnn-reorder.cu---------- *- CUDA -* ----===////
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
    cudnnTensorDescriptor_t dataTensor, outTensor;

    cudnnCreate(&handle);

    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudnnSetStream(handle, stream1);

    cudnnCreateTensorDescriptor(&dataTensor);
    cudnnCreateTensorDescriptor(&outTensor);
    int n = 1, c = 2, h = 5, w = 5;
    int ele_num = n * c * h * w;

    //using HT = dt_trait<T>::type;

    cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, T, n, c, h, w);
    cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NHWC, T, n, c, h, w);

    HT *data, *out;
    std::vector<HT> host_data(ele_num, 0);
    std::vector<HT> host_out(ele_num, 0);

    for(int i = 0; i < ele_num; i++) {
        host_data[i] = i;
        host_out[i] = 0;
    }
    //host_data.push_back(1.5f);
    //host_out.push_back(1.5f);

    cudaMalloc(&data, ele_num * sizeof(HT));
    cudaMalloc(&out, ele_num * sizeof(HT));

    cudaMemcpy(data, host_data.data(), ele_num * sizeof(HT), cudaMemcpyHostToDevice);
    cudaMemcpy(out, host_out.data(), ele_num * sizeof(HT), cudaMemcpyHostToDevice);

    float alpha = 3.f, beta = 1.f;
    auto s = cudnnTransformTensor(handle, &alpha, dataTensor, data, &beta, outTensor, out);
    //check(s);
    cudaMemcpy(host_out.data(), out, ele_num * sizeof(HT), cudaMemcpyDeviceToHost);
    std::vector<float> expect = {
      0, 75, 3, 78, 6,
      81, 9, 84, 12, 87,
      15, 90, 18, 93, 21,
      96, 24, 99, 27, 102,
      30, 105, 33, 108, 36,
      111, 39, 114, 42, 117,
      45, 120, 48, 123, 51,
      126, 54, 129, 57, 132,
      60, 135, 63, 138, 66,
      141, 69, 144, 72, 147
    };
    check(expect, host_out, expect.size(), 1e-3);
    cudnnDestroy(handle);
    cudaFree(data);
    cudaFree(out);
}

int main() {
    test<CUDNN_DATA_FLOAT>();
    std::cout << "test passed" << std::endl;
    return 0;
}