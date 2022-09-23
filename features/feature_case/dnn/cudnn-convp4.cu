// ====------ cudnn-convp4.cu ---------- *- CUDA -* ----===////
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
    cudnnTensorDescriptor_t diffdataTensor, diffoutTensor;
    cudnnFilterDescriptor_t filterTensor, difffilterTensor;
    cudnnCreate(&handle);

    cudnnCreateTensorDescriptor(&dataTensor);
    cudnnCreateTensorDescriptor(&outTensor);
    cudnnCreateFilterDescriptor(&filterTensor);
    cudnnCreateTensorDescriptor(&diffdataTensor);
    cudnnCreateTensorDescriptor(&diffoutTensor);
    cudnnCreateFilterDescriptor(&difffilterTensor);
    int in = 2, ic = 4, ih = 5, iw = 5;
    int on = 2, oc = 4, oh = 4, ow = 4;
    int fk = 4, fc = 4, fh = 2, fw = 2;
    int ele_num = in * ic * ih * iw;
    int oele_num = on * oc * oh * ow;
    int fele_num = fk *fc * fh * fw;
    cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in, ic, ih, iw);
    cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, on, oc, oh, ow);
    cudnnSetTensor4dDescriptor(diffdataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in, ic, ih, iw);
    cudnnSetTensor4dDescriptor(diffoutTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, on, oc, oh, ow);

    int filterdim[4] = {fk, fc, fh, fw};
    cudnnSetFilterNdDescriptor(filterTensor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, filterdim);
    cudnnSetFilterNdDescriptor(difffilterTensor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, filterdim);

    float *data, *out, *filter, *diffdata, *diffout, *difffilter;
    std::vector<float> host_data(in * ic * ih * iw, 1.0f);
    std::vector<float> host_out(on * oc * oh * ow, 0.0f);
    std::vector<float> host_filter(fk * fc * fh * fw, 0.0f);
    std::vector<float> host_diffdata(in * ic * ih * iw, 1.0f);
    std::vector<float> host_diffout(on * oc * oh * ow, 0.0f);
    std::vector<float> host_difffilter(fk * fc * fh * fw, 0.0f);

    for(int i = 0; i < in * ic * ih * iw; i++) {
        host_data[i] = i;
        host_diffdata[i] = i;
    }
    for(int i = 0; i < oele_num; i++) {
        host_out[i] = i;
        host_diffout[i] = i;
    }
    for(int i = 0; i < fele_num; i++) {
        host_filter[i] = i;
        host_difffilter[i] = i;
    }

    cudaMalloc(&data, sizeof(float) * in * ic * ih * iw);
    cudaMalloc(&out, sizeof(float) * on * oc * oh * ow);
    cudaMalloc(&filter, sizeof(float) * fk * fc * fh * fw);
    cudaMalloc(&diffdata, sizeof(float) * in * ic * ih * iw);
    cudaMalloc(&diffout, sizeof(float) * on * oc * oh * ow);
    cudaMalloc(&difffilter, sizeof(float) * fk * fc * fh * fw);

    cudaMemcpy(data, host_data.data(), sizeof(float) * in * ic * ih * iw, cudaMemcpyHostToDevice);
    cudaMemcpy(out, host_out.data(), sizeof(float) * on * oc * oh * ow, cudaMemcpyHostToDevice);
    cudaMemcpy(filter, host_filter.data(), sizeof(float) * fk * fc * fh * fw, cudaMemcpyHostToDevice);
    cudaMemcpy(diffdata, host_diffdata.data(), sizeof(float) * in * ic * ih * iw, cudaMemcpyHostToDevice);
    cudaMemcpy(diffout, host_diffout.data(), sizeof(float) * on * oc * oh * ow, cudaMemcpyHostToDevice);
    cudaMemcpy(difffilter, host_difffilter.data(), sizeof(float) * fk * fc * fh * fw, cudaMemcpyHostToDevice);

    cudnnConvolutionDescriptor_t covdes;
    cudnnCreateConvolutionDescriptor(&covdes);
    cudnnSetConvolution2dDescriptor(covdes, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    size_t size;
    void *workspacesize;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(
        handle, 
        dataTensor,
        diffoutTensor, 
        covdes, 
        difffilterTensor, 
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0, 
        &size);
    cudaMalloc(&workspacesize, size);

    float alpha = 1.5f, beta = 1.5f;
    cudnnConvolutionBackwardFilter(
        handle, 
        &alpha,
        dataTensor,
        data,
        diffoutTensor,
        diffout, 
        covdes, 
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0, 
        workspacesize, 
        size, 
        &beta, 
        difffilterTensor, 
        difffilter);
    cudaDeviceSynchronize();
    cudaMemcpy(host_difffilter.data(), difffilter, sizeof(float) * fele_num, cudaMemcpyDeviceToHost);

    std::vector<float> expect = {
        189924, 191822,
        199407, 201304,
        
        237330, 239228,
        246813, 248710,
        
        284736, 286634,
        294219, 296116,
        
        332142, 334040,
        341625, 343522,

        235260, 237926,
        248583, 251248,
        
        301866, 304532,
        315189, 317854,
        
        368472, 371138,
        381795, 384460,
        
        435078, 437744,
        448401, 451066,

        280596, 284030,
        297759, 301192,
        
        366402, 369836,
        383565, 386998,
        
        452208, 455642,
        469371, 472804,
        
        538014, 541448,
        555177, 558610,

        325932, 330134,
        346935, 351136,
        
        430938, 435140,
        451941, 456142,
        
        535944, 540146,
        556947, 561148,
        
        640950, 645152,
        661953, 666154,        
        };
    check(expect, host_difffilter, expect.size(), 1.f);

    cudnnDestroy(handle);
    cudaFree(data);
    cudaFree(out);
}

template<cudnnDataType_t T, typename HT = typename dt_trait<T>::type>
void test2() {
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor;
    cudnnTensorDescriptor_t diffdataTensor, diffoutTensor;
    cudnnFilterDescriptor_t filterTensor, difffilterTensor;
    cudnnCreate(&handle);

    cudnnCreateTensorDescriptor(&dataTensor);
    cudnnCreateTensorDescriptor(&outTensor);
    cudnnCreateFilterDescriptor(&filterTensor);
    cudnnCreateTensorDescriptor(&diffdataTensor);
    cudnnCreateTensorDescriptor(&diffoutTensor);
    cudnnCreateFilterDescriptor(&difffilterTensor);
    int in = 2, ic = 4, ih = 5, iw = 5;
    int on = 2, oc = 4, oh = 4, ow = 4;
    int fk = 4, fc = 2, fh = 2, fw = 2;
    int ele_num = in * ic * ih * iw;
    int oele_num = on * oc * oh * ow;
    int fele_num = fk *fc * fh * fw;
    cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in, ic, ih, iw);
    cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, on, oc, oh, ow);
    cudnnSetTensor4dDescriptor(diffdataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in, ic, ih, iw);
    cudnnSetTensor4dDescriptor(diffoutTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, on, oc, oh, ow);

    int filterdim[4] = {fk, fc, fh, fw};
    cudnnSetFilterNdDescriptor(filterTensor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, filterdim);
    cudnnSetFilterNdDescriptor(difffilterTensor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, filterdim);

    float *data, *out, *filter, *diffdata, *diffout, *difffilter;
    std::vector<float> host_data(in * ic * ih * iw, 1.0f);
    std::vector<float> host_out(on * oc * oh * ow, 0.0f);
    std::vector<float> host_filter(fk * fc * fh * fw, 0.0f);
    std::vector<float> host_diffdata(in * ic * ih * iw, 1.0f);
    std::vector<float> host_diffout(on * oc * oh * ow, 0.0f);
    std::vector<float> host_difffilter(fk * fc * fh * fw, 0.0f);

    for(int i = 0; i < in * ic * ih * iw; i++) {
        host_data[i] = i;
        host_diffdata[i] = i;
    }
    for(int i = 0; i < oele_num; i++) {
        host_out[i] = i;
        host_diffout[i] = i;
    }
    for(int i = 0; i < fele_num; i++) {
        host_filter[i] = i;
        host_difffilter[i] = i;
    }

    cudaMalloc(&data, sizeof(float) * in * ic * ih * iw);
    cudaMalloc(&out, sizeof(float) * on * oc * oh * ow);
    cudaMalloc(&filter, sizeof(float) * fk * fc * fh * fw);
    cudaMalloc(&diffdata, sizeof(float) * in * ic * ih * iw);
    cudaMalloc(&diffout, sizeof(float) * on * oc * oh * ow);
    cudaMalloc(&difffilter, sizeof(float) * fk * fc * fh * fw);

    cudaMemcpy(data, host_data.data(), sizeof(float) * in * ic * ih * iw, cudaMemcpyHostToDevice);
    cudaMemcpy(out, host_out.data(), sizeof(float) * on * oc * oh * ow, cudaMemcpyHostToDevice);
    cudaMemcpy(filter, host_filter.data(), sizeof(float) * fk * fc * fh * fw, cudaMemcpyHostToDevice);
    cudaMemcpy(diffdata, host_diffdata.data(), sizeof(float) * in * ic * ih * iw, cudaMemcpyHostToDevice);
    cudaMemcpy(diffout, host_diffout.data(), sizeof(float) * on * oc * oh * ow, cudaMemcpyHostToDevice);
    cudaMemcpy(difffilter, host_difffilter.data(), sizeof(float) * fk * fc * fh * fw, cudaMemcpyHostToDevice);

    cudnnConvolutionDescriptor_t covdes;
    cudnnCreateConvolutionDescriptor(&covdes);
    cudnnSetConvolution2dDescriptor(covdes, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    cudnnSetConvolutionGroupCount(covdes, 2);

    size_t size;
    void *workspacesize;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(
        handle, 
        dataTensor,
        diffoutTensor, 
        covdes, 
        difffilterTensor, 
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0, 
        &size);
    cudaMalloc(&workspacesize, size);

    float alpha = 1.5f, beta = 1.5f;
    cudnnConvolutionBackwardFilter(
        handle, 
        &alpha,
        dataTensor,
        data,
        diffoutTensor,
        diffout, 
        covdes, 
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0, 
        workspacesize, 
        size, 
        &beta, 
        difffilterTensor, 
        difffilter);
    cudaDeviceSynchronize();
    cudaMemcpy(host_difffilter.data(), difffilter, sizeof(float) * fele_num, cudaMemcpyDeviceToHost);

    std::vector<float> expect = {
        189924, 191822,
        199407, 201304,
        
        237330, 239228,
        246813, 248710,
        
        235248, 237914,
        248571, 251236,
        
        301854, 304520,
        315177, 317842,
        
        452172, 455606,
        469335, 472768,
        
        537978, 541412,
        555141, 558574,
        
        535896, 540098,
        556899, 561100,
        
        640902, 645104,
        661905, 666106, 
        };
    check(expect, host_difffilter, expect.size(), 1.f);

    cudnnDestroy(handle);
    cudaFree(data);
    cudaFree(out);
}

int main() {
    test1<CUDNN_DATA_FLOAT>();
    test2<CUDNN_DATA_FLOAT>();
    std::cout << "test passed" << std::endl;
    return 0;
}
