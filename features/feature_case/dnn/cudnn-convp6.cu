// ====------ cudnn-convp6.cu ---------- *- CUDA -* ----===////
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
    cudnnFilterDescriptor_t filterTensor;
    cudnnCreate(&handle);

    cudnnCreateTensorDescriptor(&dataTensor);
    cudnnCreateTensorDescriptor(&outTensor);
    cudnnCreateFilterDescriptor(&filterTensor);
    int in = 2, ic = 4, ih = 5, iw = 5;
    int on = 2, oc = 4, oh = 4, ow = 4;
    int fk = 4, fc = 4, fh = 2, fw = 2;
    int ele_num = in * ic * ih * iw;
    int oele_num = on * oc * oh * ow;
    int fele_num = fk *fc * fh * fw;
    cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in, ic, ih, iw);
    cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, on, oc, oh, ow);

    int filterdim[4] = {fk, fc, fh, fw};
    cudnnSetFilterNdDescriptor(filterTensor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, filterdim);

    float *data, *out, *filter;
    std::vector<float> host_data(in * ic * ih * iw, 1.0f);
    std::vector<float> host_out(on * oc * oh * ow, 0.0f);
    std::vector<float> host_filter(fk * fc * fh * fw, 0.0f);

    for(int i = 0; i < in * ic * ih * iw; i++) {
        host_data[i] = i;
    }
    for(int i = 0; i < oele_num; i++) {
        host_out[i] = i;
    }
    for(int i = 0; i < fele_num; i++) {
        host_filter[i] = i;
    }
    cudaMalloc(&data, sizeof(float) * in * ic * ih * iw);
    cudaMalloc(&out, sizeof(float) * on * oc * oh * ow);
    cudaMalloc(&filter, sizeof(float) * fk * fc * fh * fw);

    cudaMemcpy(data, host_data.data(), sizeof(float) * in * ic * ih * iw, cudaMemcpyHostToDevice);
    cudaMemcpy(out, host_out.data(), sizeof(float) * on * oc * oh * ow, cudaMemcpyHostToDevice);
    cudaMemcpy(filter, host_filter.data(), sizeof(float) * fk * fc * fh * fw, cudaMemcpyHostToDevice);

    cudnnConvolutionDescriptor_t covdes;
    cudnnCreateConvolutionDescriptor(&covdes);
    cudnnSetConvolution2dDescriptor(covdes, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    cudnnConvolutionFwdAlgoPerf_t perf_data;
    int returned_count;
    cudnnFindConvolutionForwardAlgorithm(handle, dataTensor, filterTensor, covdes, outTensor, 1, &returned_count, &perf_data);

    size_t size;
    void *workspacesize;
    cudnnSetConvolutionMathType(covdes, CUDNN_FMA_MATH);
    cudnnGetConvolutionForwardWorkspaceSize(
        handle, 
        dataTensor, 
        filterTensor, 
        covdes, 
        outTensor, 
        perf_data.algo, 
        &size);
    cudaMalloc(&workspacesize, size);

    float alpha = 2.5f, beta = 1.5f;
    cudnnConvolutionForward(
        handle, 
        &alpha, 
        dataTensor, 
        data, 
        filterTensor, 
        filter, 
        covdes, 
        perf_data.algo,
        workspacesize, 
        size, 
        &beta,
        outTensor, 
        out);
    cudaDeviceSynchronize();
    cudaMemcpy(host_out.data(), out, sizeof(float) * on * oc * oh * ow, cudaMemcpyDeviceToHost);

    std::vector<float> expect = {
        17260, 17561.5, 17863, 18164.5,
        18766, 19067.5, 19369, 19670.5,
        20272, 20573.5, 20875, 21176.5,
        21778, 22079.5, 22381, 22682.5,
        
        43204, 44145.5, 45087, 46028.5,
        47910, 48851.5, 49793, 50734.5,
        52616, 53557.5, 54499, 55440.5,
        57322, 58263.5, 59205, 60146.5,
        
        69148, 70729.5, 72311, 73892.5,
        77054, 78635.5, 80217, 81798.5,
        84960, 86541.5, 88123, 89704.5,
        92866, 94447.5, 96029, 97610.5,
        
        95092, 97313.5, 99535, 101756,
        106198, 108420, 110641, 112862,
        117304, 119526, 121747, 123968,
        128410, 130632, 132853, 135074,

        47356, 47657.5, 47959, 48260.5,
        48862, 49163.5, 49465, 49766.5,
        50368, 50669.5, 50971, 51272.5,
        51874, 52175.5, 52477, 52778.5,
        
        137300, 138242, 139183, 140124,
        142006, 142948, 143889, 144830,
        146712, 147654, 148595, 149536,
        151418, 152360, 153301, 154242,
        
        227244, 228826, 230407, 231988,
        235150, 236732, 238313, 239894,
        243056, 244638, 246219, 247800,
        250962, 252544, 254125, 255706,
        
        317188, 319410, 321631, 323852,
        328294, 330516, 332737, 334958,
        339400, 341622, 343843, 346064,
        350506, 352728, 354949, 357170,               
        };
    check(expect, host_out, expect.size(), 1.f);

    cudnnDestroy(handle);
    cudaFree(data);
    cudaFree(out);
}

int main() {
    test1<CUDNN_DATA_FLOAT>();
    std::cout << "test passed" << std::endl;
    return 0;
}
