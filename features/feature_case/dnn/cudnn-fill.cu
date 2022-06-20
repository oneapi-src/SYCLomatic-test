// ====------ cudnn-fill.cu ---------- *- CUDA -* ----===////
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


template<cudnnDataType_t T, typename HT = typename dt_trait<T>::type>
void test() {

    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor;

    cudnnCreate(&handle);

    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudnnSetStream(handle, stream1);

    cudnnCreateTensorDescriptor(&dataTensor);
    int n = 1, c = 2, h = 5, w = 5;
    int ele_num = n * c * h * w;
    HT * data;
    HT value = 1.5;
    std::vector<HT> host_data(ele_num, 0);

    cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, T, n, c, h, w);

    cudaMalloc(&data, ele_num * sizeof(HT));

    cudnnSetTensor(handle, dataTensor, data, &value);

    cudaMemcpy(host_data.data(), data, ele_num * sizeof(HT), cudaMemcpyDeviceToHost);
    float precision = 1e-3;
    for(int i = 0; i < ele_num; i++) {
        if(std::abs(host_data[i] -value) > precision) {
            std::cout << "test fail" << std::endl;
            exit(-1);
        } 
    }
}

int main() {
    test<CUDNN_DATA_FLOAT>();
    std::cout << "test passed" << std::endl;
    return 0;
}