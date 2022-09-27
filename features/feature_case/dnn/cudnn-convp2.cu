// ====------ cudnn-convp2.cu ---------- *- CUDA -* ----===////
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
    cudaMemcpy(filter, host_filter.data(), sizeof(float) * fk * fc * fh * fw, cudaMemcpyHostToDevice);

    cudnnConvolutionDescriptor_t covdes;
    cudnnCreateConvolutionDescriptor(&covdes);
    cudnnSetConvolution2dDescriptor(covdes, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    size_t size;
    void *workspacesize;
    cudnnGetConvolutionForwardWorkspaceSize(handle, dataTensor, filterTensor, covdes, outTensor, CUDNN_CONVOLUTION_FWD_ALGO_GEMM, &size);
    cudaMalloc(&workspacesize, size);

    cudnnActivationDescriptor_t ActivationDesc;
    cudnnCreateActivationDescriptor(&ActivationDesc);
    cudnnSetActivationDescriptor(ActivationDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0f);
    
    float alpha = 2.5f, beta = 1.5f;
    cudnnConvolutionBiasActivationForward(
        handle, 
        &alpha, 
        dataTensor, 
        data, 
        filterTensor, 
        filter, 
        covdes, 
        CUDNN_CONVOLUTION_FWD_ALGO_GEMM, 
        workspacesize, 
        size,
        &beta,
        outTensor,
        z,
        biasTensor,
        bias,
        ActivationDesc,
        outTensor, 
        out);

    cudaDeviceSynchronize();
    cudaMemcpy(host_out.data(), out, sizeof(float) * on * oc * oh * ow, cudaMemcpyDeviceToHost);

    std::vector<float> expect = {
        13840, 14141.5, 14443, 14744.5,
        15346, 15647.5, 15949, 16250.5,
        16852, 17153.5, 17455, 17756.5,
        18358, 18659.5, 18961, 19262.5,
        
        39785, 40726.5, 41668, 42609.5,
        44491, 45432.5, 46374, 47315.5,
        49197, 50138.5, 51080, 52021.5,
        53903, 54844.5, 55786, 56727.5,
        
        65730, 67311.5, 68893, 70474.5,
        73636, 75217.5, 76799, 78380.5,
        81542, 83123.5, 84705, 86286.5,
        89448, 91029.5, 92611, 94192.5,
        
        91675, 93896.5, 96118, 98339.5,
        102781, 105002, 107224, 109446,
        113887, 116108, 118330, 120552,
        124993, 127214, 129436, 131658,

        43936, 44237.5, 44539, 44840.5,
        45442, 45743.5, 46045, 46346.5,
        46948, 47249.5, 47551, 47852.5,
        48454, 48755.5, 49057, 49358.5,
        
        133881, 134822, 135764, 136706,
        138587, 139528, 140470, 141412,
        143293, 144234, 145176, 146118,
        147999, 148940, 149882, 150824,
        
        223826, 225408, 226989, 228570,
        231732, 233314, 234895, 236476,
        239638, 241220, 242801, 244382,
        247544, 249126, 250707, 252288,
        
        313771, 315992, 318214, 320436,
        324877, 327098, 329320, 331542,
        335983, 338204, 340426, 342648,
        347089, 349310, 351532, 353754,        
        };
    check(expect, host_out, expect.size(), 1.f);

    cudnnDestroy(handle);
    cudaFree(data);
    cudaFree(out);
}

template<cudnnDataType_t T, typename HT = typename dt_trait<T>::type>
void test2() {
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
    int fk = 4, fc = 2, fh = 2, fw = 2;
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
    cudaMemcpy(filter, host_filter.data(), sizeof(float) * fk * fc * fh * fw, cudaMemcpyHostToDevice);

    cudnnConvolutionDescriptor_t covdes;
    cudnnCreateConvolutionDescriptor(&covdes);
    cudnnSetConvolution2dDescriptor(covdes, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    cudnnSetConvolutionGroupCount(covdes, 2);

    size_t size;
    void *workspacesize;
    cudnnGetConvolutionForwardWorkspaceSize(handle, dataTensor, filterTensor, covdes, outTensor, CUDNN_CONVOLUTION_FWD_ALGO_GEMM, &size);
    cudaMalloc(&workspacesize, size);

    cudnnActivationDescriptor_t ActivationDesc;
    cudnnCreateActivationDescriptor(&ActivationDesc);
    cudnnSetActivationDescriptor(ActivationDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0f);
    
    float alpha = 2.5f, beta = 1.5f;
    cudnnConvolutionBiasActivationForward(
        handle, 
        &alpha, 
        dataTensor, 
        data, 
        filterTensor, 
        filter, 
        covdes, 
        CUDNN_CONVOLUTION_FWD_ALGO_GEMM, 
        workspacesize, 
        size,
        &beta,
        outTensor,
        z,
        biasTensor,
        bias,
        ActivationDesc,
        outTensor, 
        out);

    cudaDeviceSynchronize();
    cudaMemcpy(host_out.data(), out, sizeof(float) * on * oc * oh * ow, cudaMemcpyDeviceToHost);

    std::vector<float> expect = {
        1320, 1391.5, 1463, 1534.5,
        1676, 1747.5, 1819, 1890.5,
        2032, 2103.5, 2175, 2246.5,
        2388, 2459.5, 2531, 2602.5,
        
        3825, 4056.5, 4288, 4519.5,
        4981, 5212.5, 5444, 5675.5,
        6137, 6368.5, 6600, 6831.5,
        7293, 7524.5, 7756, 7987.5,
        
        25830, 26221.5, 26613, 27004.5,
        27786, 28177.5, 28569, 28960.5,
        29742, 30133.5, 30525, 30916.5,
        31698, 32089.5, 32481, 32872.5,
        
        36335, 36886.5, 37438, 37989.5,
        39091, 39642.5, 40194, 40745.5,
        41847, 42398.5, 42950, 43501.5,
        44603, 45154.5, 45706, 46257.5,

        8416, 8487.5, 8559, 8630.5,
        8772, 8843.5, 8915, 8986.5,
        9128, 9199.5, 9271, 9342.5,
        9484, 9555.5, 9627, 9698.5,
        
        26921, 27152.5, 27384, 27615.5,
        28077, 28308.5, 28540, 28771.5,
        29233, 29464.5, 29696, 29927.5,
        30389, 30620.5, 30852, 31083.5,
        
        64926, 65317.5, 65709, 66100.5,
        66882, 67273.5, 67665, 68056.5,
        68838, 69229.5, 69621, 70012.5,
        70794, 71185.5, 71577, 71968.5,
        
        91431, 91982.5, 92534, 93085.5,
        94187, 94738.5, 95290, 95841.5,
        96943, 97494.5, 98046, 98597.5,
        99699, 100250, 100802, 101354,            
        };
    check(expect, host_out, expect.size(), 1.f);

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
