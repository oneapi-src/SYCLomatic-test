// ====------ cudnn-pooling.cu ---------- *- CUDA -* ----===////
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
void test1() {

    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor;

    cudnnCreate(&handle);



    cudnnPoolingDescriptor_t desc;
    cudnnCreatePoolingDescriptor(&desc);
    cudnnSetPooling2dDescriptor(desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 4, 4, 3, 3, 2, 2);

    cudnnCreateTensorDescriptor(&dataTensor);
    cudnnCreateTensorDescriptor(&outTensor);
    int n = 1, c = 2, h = 5, w = 5;
    int ele_num = n * c * h * w;

    cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, T, n, c, h, w);
    int on, oc, oh, ow;
    cudnnGetPooling2dForwardOutputDim(desc, dataTensor, &on, &oc, &oh, &ow);

    cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, T, on, oc, oh, ow);

    HT *data, *out;
    std::vector<HT> host_data(ele_num);
    int ele_num2 = on * oc * oh * ow;
    std::vector<HT> host_out(ele_num2);

    for(int i = 0; i < ele_num; i++) {
        host_data[i] = i;
    }

    for(int i = 0; i < ele_num2; i++) {
        host_out[i] = i;
    }

    cudaMalloc(&data, ele_num * sizeof(HT));
    cudaMalloc(&out, ele_num2 * sizeof(HT));

    cudaMemcpy(data, host_data.data(), ele_num * sizeof(HT), cudaMemcpyHostToDevice);
    cudaMemcpy(out, host_out.data(), ele_num2 * sizeof(HT), cudaMemcpyHostToDevice);

    float alpha = 1.f, beta = 0.f;
    auto s = cudnnPoolingForward(handle, desc, &alpha, dataTensor, data, &beta, outTensor, out);

    cudaMemcpy(host_out.data(), out, ele_num2 * sizeof(HT), cudaMemcpyDeviceToHost);

    std::vector<float> expect = {
        0, 2, 4, 4,
        10, 12, 14, 14,
        20, 22, 24, 24,
        20, 22, 24, 24,
        25, 27, 29, 29,
        35, 37, 39, 39,
        45, 47, 49, 49,
        45, 47, 49, 49
      };
      check(expect, host_out, expect.size(), 1e-3);
    cudnnDestroy(handle);
    cudaFree(data);
    cudaFree(out);
}

template<cudnnDataType_t T, typename HT = typename dt_trait<T>::type>
void test2() {

    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor, diffdataTensor, diffoutTensor;

    cudnnCreate(&handle);



    cudnnPoolingDescriptor_t desc;
    cudnnCreatePoolingDescriptor(&desc);
    cudnnSetPooling2dDescriptor(desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 4, 4, 3, 3, 2, 2);

    cudnnCreateTensorDescriptor(&dataTensor);
    cudnnCreateTensorDescriptor(&outTensor);
    cudnnCreateTensorDescriptor(&diffdataTensor);
    cudnnCreateTensorDescriptor(&diffoutTensor);
    int n = 1, c = 2, h = 5, w = 5;
    int ele_num = n * c * h * w;

    cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, T, n, c, h, w);

    cudnnSetTensor4dDescriptor(diffdataTensor, CUDNN_TENSOR_NCHW, T, n, c, h, w);

    int on, oc, oh, ow;
    cudnnGetPooling2dForwardOutputDim(desc, dataTensor, &on, &oc, &oh, &ow);

    cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, T, on, oc, oh, ow);
    cudnnSetTensor4dDescriptor(diffoutTensor, CUDNN_TENSOR_NCHW, T, on, oc, oh, ow);
    int ele_num2 = on * oc * oh * ow;

    HT *data, *out, *diffdata, *diffout;
    std::vector<HT> host_data(ele_num);
    std::vector<HT> host_out(ele_num2);
    std::vector<HT> host_diffdata(ele_num);
    std::vector<HT> host_diffout(ele_num2);
    for(int i = 0; i < ele_num; i++) {
        host_data[i] = i * 0.1f;
        host_diffdata[i] = i;
    }
    for(int i = 0; i < ele_num2; i++) {
        host_out[i] = i;
        host_diffout[i] = 1.f;
    }

    cudaMalloc(&data, ele_num * sizeof(HT));
    cudaMalloc(&out, ele_num2 * sizeof(HT));
    cudaMalloc(&diffdata, ele_num * sizeof(HT));
    cudaMalloc(&diffout, ele_num2 * sizeof(HT));

    cudaMemcpy(data, host_data.data(), ele_num * sizeof(HT), cudaMemcpyHostToDevice);
    cudaMemcpy(out, host_out.data(), ele_num2 * sizeof(HT), cudaMemcpyHostToDevice);
    cudaMemcpy(diffdata, host_diffdata.data(), ele_num * sizeof(HT), cudaMemcpyHostToDevice);
    cudaMemcpy(diffout, host_diffout.data(), ele_num2 * sizeof(HT), cudaMemcpyHostToDevice);

    float alpha = 1.5f, beta = 1.f;
    cudnnPoolingForward(handle, desc, &alpha, dataTensor, data, &beta, outTensor, out);
    cudaMemcpy(host_out.data(), out, ele_num2 * sizeof(HT), cudaMemcpyDeviceToHost);
    alpha = 1.5f, beta = 1.f;
    auto s = cudnnPoolingBackward(handle, desc, &alpha, outTensor, out, diffoutTensor, diffout, dataTensor, data, &beta, diffdataTensor, diffdata);

    cudaMemcpy(host_diffdata.data(), diffdata, ele_num * sizeof(HT), cudaMemcpyDeviceToHost);

    std::vector<float> expect = {
        1.5, 1, 3.5, 3, 7,
        5, 6, 7, 8, 9,
        11.5, 11, 13.5, 13, 17,
        15, 16, 17, 18, 19,
        23, 21, 25, 23, 30,
        26.5, 26, 28.5, 28, 32,
        30, 31, 32, 33, 34,
        36.5, 36, 38.5, 38, 42,
        40, 41, 42, 43, 44,
        48, 46, 50, 48, 55
      };
      check(expect, host_diffdata, expect.size(), 1e-3);
    cudnnDestroy(handle);
    cudaFree(data);
    cudaFree(out);
    cudaFree(diffdata);
    cudaFree(diffout);
}

int main() {
    test1<CUDNN_DATA_FLOAT>();
    test2<CUDNN_DATA_FLOAT>();
    std::cout << "test passed" << std::endl;
    return 0;
}