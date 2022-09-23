// ====------ cudnn-convp5.cu ---------- *- CUDA -* ----===////
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
    cudnnTensorDescriptor_t dataTensor, outTensor, biasTensor;
    cudnnFilterDescriptor_t filterTensor;
    auto status = cudnnCreate(&handle);

    cudnnCreateTensorDescriptor(&dataTensor);
    cudnnCreateTensorDescriptor(&outTensor);
    cudnnCreateFilterDescriptor(&filterTensor);
    cudnnCreateTensorDescriptor(&biasTensor);
    int in = 2, ic = 4, ih = 5, iw = 5;
    int on = 2, oc = 4, oh = 4, ow = 4;
    int fk = 4, fc = 4, fh = 2, fw = 2;
    int ele_num = in * ic * ih * iw;
    int oele_num = on * oc * oh * ow;
    int fele_num = fk *fc * fh * fw;
    std::vector<int> bias_dim = {1, oc, 1, 1};
    std::vector<int> bias_stride = {oc, 1, 1, 1};
    int bele_num = oc * 1;
    cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in, ic, ih, iw);
    cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, on, oc, oh, ow);
    cudnnSetTensorNdDescriptor(biasTensor, CUDNN_DATA_FLOAT, 4, bias_dim.data(), bias_stride.data());
    int filterdim[4] = {fk, fc, fh, fw};
    cudnnSetFilterNdDescriptor(filterTensor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NHWC, 4, filterdim);

    float *data, *out, *filter, *z, *bias;
    std::vector<float> host_data(in * ic * ih * iw, 1.0f);
    std::vector<float> host_out(on * oc * oh * ow, 0.0f);
    std::vector<float> host_z(oele_num, 0.0f);
    std::vector<float> host_bias(bele_num, 0.0f);
    std::vector<float> host_filter(fk * fc * fh * fw, 0.0f);

    for(int i = 0; i < in * ic * ih * iw; i++) {
        host_data[i] = i;
    }
    for(int i = 0; i < oele_num; i++) {
        host_out[i] = i;
        host_z[i] = i;
    }
    for(int i = 0; i < bele_num; i++) {
        host_bias[i] = i;
    }
    for(int i = 0; i < fele_num; i++) {
        host_filter[i] = i;
    }

    cudaMalloc(&data, sizeof(float) * in * ic * ih * iw);
    cudaMalloc(&out, sizeof(float) * on * oc * oh * ow);
    cudaMalloc(&z, sizeof(float) * on * oc * oh * ow);
    cudaMalloc(&bias, sizeof(float) * bele_num);
    cudaMalloc(&filter, sizeof(float) * fk * fc * fh * fw);

    cudaMemcpy(data, host_data.data(), sizeof(float) * in * ic * ih * iw, cudaMemcpyHostToDevice);
    cudaMemcpy(out, host_out.data(), sizeof(float) * on * oc * oh * ow, cudaMemcpyHostToDevice);
    cudaMemcpy(z, host_z.data(), sizeof(float) * on * oc * oh * ow, cudaMemcpyHostToDevice);
    cudaMemcpy(bias, host_bias.data(), sizeof(float) * bele_num, cudaMemcpyHostToDevice);

    float alpha = 2.5f, beta = 1.5f;

    cudnnConvolutionBackwardBias(
        handle,
        &alpha,
        outTensor,
        out,
        &beta,
        biasTensor,
        bias
    );

    cudaDeviceSynchronize();
    cudaMemcpy(host_bias.data(), bias, sizeof(float) * bele_num, cudaMemcpyDeviceToHost);

    std::vector<float> expect = {
        3160,

        4441.5,

        5723,

        7004.5 
        };
    check(expect, host_bias, expect.size(), 1.f);

    cudnnDestroy(handle);
    cudaFree(data);
    cudaFree(out);
}

int main() {
    test1<CUDNN_DATA_FLOAT>();
    std::cout << "test passed" << std::endl;
    return 0;
}
