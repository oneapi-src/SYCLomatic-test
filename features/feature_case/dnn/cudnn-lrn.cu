// ====------ cudnn-lrn.cu ---------- *- CUDA -* ----===////
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

    unsigned int local_size = 3;
    float lrn_alpha = 1.5f;
    float lrn_beta = 1.5f;
    float lrn_k = 1.f;

    cudnnLRNDescriptor_t desc;
    cudnnCreateLRNDescriptor(&desc);
    cudnnSetLRNDescriptor(desc, local_size, lrn_alpha, lrn_beta, lrn_k);

    float alpha = 2.f, beta = 1.5f;
    auto s = cudnnLRNCrossChannelForward(handle, desc, CUDNN_LRN_CROSS_CHANNEL_DIM1, &alpha, dataTensor, data, &beta, outTensor, out);

    cudaMemcpy(host_out.data(), out, ele_num * sizeof(HT), cudaMemcpyDeviceToHost);

    std::vector<float> expect = {
        0, 1.50032, 3.00057, 4.50076, 6.0009,
        7.501, 9.00107, 10.5011, 12.0012, 13.5012,
        15.0012, 16.5012, 18.0012, 19.5011, 21.0011,
        22.5011, 24.0011, 25.501, 27.001, 28.501,
        30.0009, 31.5009, 33.0009, 34.5009, 36.0008,
        37.509, 39.0083, 40.5077, 42.0071, 43.5065,
        45.006, 46.5056, 48.0051, 49.5048, 51.0044,
        52.5041, 54.0038, 55.5035, 57.0033, 58.5031,
        60.0029, 61.5027, 63.0026, 64.5024, 66.0023,
        67.5021, 69.002, 70.5019, 72.0018, 73.5017
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
        host_data[i] = i;
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

    unsigned int local_size = 3;
    float lrn_alpha = 1.5f;
    float lrn_beta = 1.5f;
    float lrn_k = 1.f;

    cudnnLRNDescriptor_t desc;
    cudnnCreateLRNDescriptor(&desc);
    cudnnSetLRNDescriptor(desc, local_size, lrn_alpha, lrn_beta, lrn_k);

    float alpha = 1.5f, beta = 0.f;
    cudnnLRNCrossChannelForward(handle, desc, CUDNN_LRN_CROSS_CHANNEL_DIM1, &alpha, dataTensor, data, &beta, outTensor, out);

    alpha = 2000.f, beta = 0.f;
    cudaDeviceSynchronize();
    auto s = cudnnLRNCrossChannelBackward(handle, desc, CUDNN_LRN_CROSS_CHANNEL_DIM1, &alpha, outTensor, out, diffoutTensor, diffout, dataTensor, data, &beta, diffdataTensor, diffdata);
    cudaDeviceSynchronize();

    cudaMemcpy(host_diffdata.data(), diffdata, ele_num * sizeof(HT), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    std::vector<float> expect = {
        0.360308, 0.28158, 0.21668, 0.163798, 0.121108,
        0.0869165, 0.059718, 0.0382204, 0.021336, 0.00816559,
        -0.00202841, -0.00984461, -0.0157668, -0.0201844, -0.0234096,
        -0.0256924, -0.0272326, -0.02819, -0.0286917, -0.0288397,
        -0.0287147, -0.0283811, -0.0278906, -0.0272837, -0.0265927,
        -0.717169, -0.67193, -0.623392, -0.574243, -0.526284,
        -0.480635, -0.437933, -0.398476, -0.362341, -0.329454,
        -0.299655, -0.272737, -0.248469, -0.226616, -0.206946,
        -0.189243, -0.173305, -0.158945, -0.145997, -0.13431,
        -0.123748, -0.114191, -0.105531, -0.0976739, -0.0905342
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