// ====------ cudnn-activation.cu---------- *- CUDA -* ----===////
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

    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudnnSetStream(handle, stream1);

    cudnnCreateTensorDescriptor(&dataTensor);
    cudnnCreateTensorDescriptor(&outTensor);
    int n = 1, c = 2, h = 5, w = 5;
    int ele_num = n * c * h * w;

    cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, T, n, c, h, w);
    cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, T, n, c, h, w);

    HT *data, *out;
    std::vector<HT> host_data(ele_num);
    std::vector<HT> host_out(ele_num);

    for(int i = 0; i < ele_num; i++) {
        host_data[i] = i;
        host_out[i] = i;
    }

    cudaMalloc(&data, ele_num * sizeof(HT));
    cudaMalloc(&out, ele_num * sizeof(HT));

    cudaMemcpy(data, host_data.data(), ele_num * sizeof(HT), cudaMemcpyHostToDevice);
    cudaMemcpy(out, host_out.data(), ele_num * sizeof(HT), cudaMemcpyHostToDevice);

    cudnnActivationDescriptor_t desc;
    cudnnCreateActivationDescriptor(&desc);
    cudnnSetActivationDescriptor(desc, CUDNN_ACTIVATION_SIGMOID, CUDNN_NOT_PROPAGATE_NAN, 0.f);

    float alpha = 2.f, beta = 1.5f;
    auto s = cudnnActivationForward(handle, desc, &alpha, dataTensor, data, &beta, outTensor, out);

    cudaMemcpy(host_out.data(), out, ele_num * sizeof(HT), cudaMemcpyDeviceToHost);

    std::vector<float> expect = {
        1, 2.96212, 4.76159, 6.40515, 7.96403,
        9.48661, 10.9951, 12.4982, 13.9993, 15.4998,
        16.9999, 18.5, 20, 21.5, 23,
        24.5, 26, 27.5, 29, 30.5,
        32, 33.5, 35, 36.5, 38,
        39.5, 41, 42.5, 44, 45.5,
        47, 48.5, 50, 51.5, 53,
        54.5, 56, 57.5, 59, 60.5,
        62, 63.5, 65, 66.5, 68,
        69.5, 71, 72.5, 74, 75.5    
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

    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudnnSetStream(handle, stream1);

    cudnnCreateTensorDescriptor(&dataTensor);
    cudnnCreateTensorDescriptor(&outTensor);
    cudnnCreateTensorDescriptor(&diffdataTensor);
    cudnnCreateTensorDescriptor(&diffoutTensor);
    int n = 1, c = 2, h = 5, w = 5;
    int ele_num = n * c * h * w;

    cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, T, n, c, h, w);
    cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, T, n, c, h, w);
    cudnnSetTensor4dDescriptor(diffdataTensor, CUDNN_TENSOR_NCHW, T, n, c, h, w);
    cudnnSetTensor4dDescriptor(diffoutTensor, CUDNN_TENSOR_NCHW, T, n, c, h, w);
    HT *data, *out, *diffdata, *diffout;
    std::vector<HT> host_data(ele_num);
    std::vector<HT> host_out(ele_num);
    std::vector<HT> host_diffdata(ele_num);
    std::vector<HT> host_diffout(ele_num);
    for(int i = 0; i < ele_num; i++) {
        host_data[i] = i * 0.1f;
        host_out[i] = i;
        host_diffdata[i] = i;
        host_diffout[i] = 1.f;
    }

    cudaMalloc(&data, ele_num * sizeof(HT));
    cudaMalloc(&out, ele_num * sizeof(HT));
    cudaMalloc(&diffdata, ele_num * sizeof(HT));
    cudaMalloc(&diffout, ele_num * sizeof(HT));

    cudaMemcpy(data, host_data.data(), ele_num * sizeof(HT), cudaMemcpyHostToDevice);
    cudaMemcpy(out, host_out.data(), ele_num * sizeof(HT), cudaMemcpyHostToDevice);
    cudaMemcpy(diffdata, host_diffdata.data(), ele_num * sizeof(HT), cudaMemcpyHostToDevice);
    cudaMemcpy(diffout, host_diffout.data(), ele_num * sizeof(HT), cudaMemcpyHostToDevice);

    cudnnActivationDescriptor_t desc;
    cudnnCreateActivationDescriptor(&desc);
    cudnnSetActivationDescriptor(desc, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0.f);

    float alpha = 1.5f, beta = 0.f;
    cudnnActivationForward(handle, desc, &alpha, dataTensor, data, &beta, outTensor, out);

    alpha = 2.f, beta = 0.f;

    auto s = cudnnActivationBackward(handle, desc, &alpha, outTensor, out, diffoutTensor, diffout, dataTensor, data, &beta, diffdataTensor, diffdata);

    cudaMemcpy(host_diffdata.data(), diffdata, ele_num * sizeof(HT), cudaMemcpyDeviceToHost);

    std::vector<float> expect = {
        0.375, 0.334723, 0.289074, 0.238399, 0.183142,
        0.123828, 0.0610447, -0.00457374, -0.072368, -0.141673,
        -0.211834, -0.282226, -0.352262, -0.42141, -0.489194,
        -0.555202, -0.61909, -0.680577, -0.739441, -0.795526,
        -0.848724, -0.898978, -0.946273, -0.990628, -1.03209,
        -1.07075, -1.10668, -1.14001, -1.17084, -1.19932,
        -1.22557, -1.24972, -1.27191, -1.29227, -1.31092,
        -1.32799, -1.3436, -1.35786, -1.37087, -1.38273,
        -1.39354, -1.40338, -1.41234, -1.42049, -1.42789,
        -1.43462, -1.44073, -1.44629, -1.45132, -1.4559             
    };

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