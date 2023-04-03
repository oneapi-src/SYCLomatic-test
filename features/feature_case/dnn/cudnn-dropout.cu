// ====------ cudnn-dropout.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include<cuda_runtime.h>
#include<cudnn.h>
#include<iostream>

void test1(){

    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor;

    cudnnCreate(&handle);

    cudnnCreateTensorDescriptor(&dataTensor);
    cudnnCreateTensorDescriptor(&outTensor);

    int n = 4, c = 3, h = 32, w = 32;
    int ele_num = n * c * h * w;
    float *data, *out, *d_data, *d_out;

    cudaMallocManaged(&data, ele_num * sizeof(float));
    cudaMallocManaged(&out, ele_num * sizeof(float));
    cudaMallocManaged(&d_data, ele_num * sizeof(float));
    cudaMallocManaged(&d_out, ele_num * sizeof(float));

    for(int i = 0; i < ele_num; i++){
      data[i] = 2.f;
      d_out[i] = 3.f;
    }

    cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
    cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);

    float dropout = 0.8f;

    size_t reserve_size;
    void *reserve;
    size_t state_size;
    void *state;
    cudnnDropoutGetStatesSize(handle, &state_size);
    cudnnDropoutGetReserveSpaceSize(dataTensor, &reserve_size);
    cudaMalloc(&reserve, reserve_size);
    cudaMalloc(&state, state_size);

    cudnnDropoutDescriptor_t desc;
    cudnnCreateDropoutDescriptor(&desc);
    cudnnSetDropoutDescriptor(desc, handle, dropout, state, state_size, 1231);
    cudnnDropoutForward(handle, desc, dataTensor, data, outTensor, out, reserve, reserve_size);

    cudaDeviceSynchronize();
    float sum = 0.f, ave = 0.f, expect = 0.f, precision = 1.e-1;
    for(int i = 0; i < ele_num; i++){
      sum += out[i];
    }

    expect = 2.f * (1.f / (1.f - dropout)) * (1.f - dropout);
    ave = sum / ele_num;
    if(std::abs(ave - expect) > precision) {
        std::cout << "expect: " << expect << std::endl;
        std::cout << "get: " << ave << std::endl;
        std::cout << "test failed" << std::endl;
        exit(-1);
    }
    cudnnDropoutBackward(handle, desc, dataTensor, d_out, outTensor, d_data, reserve, reserve_size);
    cudaDeviceSynchronize();
    sum = 0.f;
    for(int i = 0; i < ele_num; i++){
      sum += d_data[i];
    }
    expect = 3.f * (1.f / (1.f - dropout)) * (1.f - dropout);
    ave = sum / ele_num;
    if(std::abs(ave - expect) > precision) {
        std::cout << "test failed" << std::endl;
        exit(-1);
    }
    cudaFree(data);
    cudaFree(out);
    cudaFree(d_data);
    cudaFree(d_out);
    cudnnDestroy(handle);
}
void test2(){

    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor;

    cudnnCreate(&handle);

    cudnnCreateTensorDescriptor(&dataTensor);
    cudnnCreateTensorDescriptor(&outTensor);

    int n = 4, c = 3, h = 32, w = 32;
    int ele_num = n * c * h * w;
    float *data, *out;

    cudaMallocManaged(&data, ele_num * sizeof(float));
    cudaMallocManaged(&out, ele_num * sizeof(float));

    for(int i = 0; i < ele_num; i++){
      data[i] = 2.f;
    }

    cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
    cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);

    float dropout = 0.8f;

    size_t reserve_size;
    void *reserve;
    size_t state_size;
    void *state;
    cudnnDropoutGetStatesSize(handle, &state_size);
    cudnnDropoutGetReserveSpaceSize(dataTensor, &reserve_size);
    cudaMalloc(&reserve, reserve_size);
    cudaMalloc(&state, state_size);

    cudnnDropoutDescriptor_t desc;
    cudnnCreateDropoutDescriptor(&desc);
    cudnnSetDropoutDescriptor(desc, handle, dropout, state, state_size, 1231);
    cudnnDropoutForward(handle, desc, dataTensor, data, outTensor, out, reserve, reserve_size);

    cudaDeviceSynchronize();
    float sum = 0.f, ave = 0.f, expect = 0.f, precision = 1.e-1;
    for(int i = 0; i < ele_num; i++){
      sum += out[i];
    }

    expect = 2.f * (1.f / (1.f - dropout)) * (1.f - dropout);
    ave = sum / ele_num;
    if(std::abs(ave - expect) > precision) {
        std::cout << "expect: " << expect << std::endl;
        std::cout << "get: " << ave << std::endl;
        std::cout << "test failed" << std::endl;
        exit(-1);
    }

    cudnnDropoutDescriptor_t desc2;
    cudnnCreateDropoutDescriptor(&desc2);
    cudnnRestoreDropoutDescriptor(desc2, handle, dropout, state, state_size, 1231);
    cudnnDropoutForward(handle, desc2, dataTensor, data, outTensor, out, reserve, reserve_size);

    cudaDeviceSynchronize();
    sum = 0.f;
    for(int i = 0; i < ele_num; i++){
      sum += out[i];
    }
    ave = sum / ele_num;
    if(std::abs(ave - expect) > precision) {
        std::cout << "expect: " << expect << std::endl;
        std::cout << "get: " << ave << std::endl;
        std::cout << "test failed" << std::endl;
        exit(-1);
    }

    cudaFree(data);
    cudaFree(out);
    cudnnDestroy(handle);
}
int main() {
    test1();
    test2();
    std::cout << "test passed" << std::endl;
    return 0;
}
