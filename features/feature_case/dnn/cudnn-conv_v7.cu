// ====------ cudnn-conv_v7.cu ---------- *- CUDA -* ----===////
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

    cudnnConvolutionFwdAlgo_t fwd_a;
    cudnnGetConvolutionForwardAlgorithm(handle, dataTensor, filterTensor, covdes, outTensor, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 10, &fwd_a);

    size_t size;
    void *workspacesize;
    cudnnGetConvolutionForwardWorkspaceSize(
        handle, 
        dataTensor, 
        filterTensor, 
        covdes, 
        outTensor, 
        fwd_a, 
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
        fwd_a,
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

    cudnnConvolutionBwdDataAlgo_t bwd_data_a;
    cudnnGetConvolutionBackwardDataAlgorithm(handle, filterTensor, dataTensor, covdes, outTensor, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 10, &bwd_data_a);

    size_t size;
    void *workspacesize;
    cudnnGetConvolutionBackwardDataWorkspaceSize(
        handle, 
        filterTensor,
        diffoutTensor, 
        covdes, 
        diffdataTensor, 
        bwd_data_a,
        &size);
    cudaMalloc(&workspacesize, size);

    float alpha = 2.5f, beta = 1.5f;
    cudnnConvolutionBackwardData(
        handle, 
        &alpha,
        filterTensor, 
        filter,
        diffoutTensor,
        diffout, 
        covdes, 
        bwd_data_a, 
        workspacesize, 
        size, 
        &beta, 
        diffdataTensor, 
        diffdata);
    cudaDeviceSynchronize();
    cudaMemcpy(host_diffdata.data(), diffdata, sizeof(float) * ele_num, cudaMemcpyDeviceToHost);

    std::vector<float> expect = {
        8960, 18401.5, 18893, 19384.5, 9956,
        19367.5, 39749, 40770.5, 41792, 21453.5,
        21375, 43836.5, 44858, 45879.5, 23541,
        23382.5, 47924, 48945.5, 49967, 25628.5,
        12590, 25771.5, 26303, 26834.5, 13766,
        
        9957.5, 20399, 20970.5, 21542, 11073.5,
        21485, 44026.5, 45208, 46389.5, 23811,
        23812.5, 48754, 49935.5, 51117, 26218.5,
        26140, 53481.5, 54663, 55844.5, 28626,
        14067.5, 28729, 29340.5, 29952, 15363.5,
        
        10955, 22396.5, 23048, 23699.5, 12191,
        23602.5, 48304, 49645.5, 50987, 26168.5,
        26250, 53671.5, 55013, 56354.5, 28896,
        28897.5, 59039, 60380.5, 61722, 31623.5,
        15545, 31686.5, 32378, 33069.5, 16961,
        
        11952.5, 24394, 25125.5, 25857, 13308.5,
        25720, 52581.5, 54083, 55584.5, 28526,
        28687.5, 58589, 60090.5, 61592, 31573.5,
        31655, 64596.5, 66098, 67599.5, 34621,
        17022.5, 34644, 35415.5, 36187, 18558.5,

        24470, 49911.5, 50403, 50894.5, 26106,
        51517.5, 105179, 106200, 107222, 54883.5,
        53525, 109266, 110288, 111310, 56971,
        55532.5, 113354, 114376, 115397, 59058.5,
        29380, 59841.5, 60373, 60904.5, 31196,
        
        28027.5, 57029, 57600.5, 58172, 29783.5,
        58755, 119696, 120878, 122060, 62361,
        61082.5, 124424, 125606, 126787, 64768.5,
        63410, 129152, 130333, 131514, 67176,
        33417.5, 67919, 68530.5, 69142, 35353.5,
        
        31585, 64146.5, 64798, 65449.5, 33461,
        65992.5, 134214, 135556, 136897, 69838.5,
        68640, 139582, 140923, 142264, 72566,
        71287.5, 144949, 146290, 147632, 75293.5,
        37455, 75996.5, 76688, 77379.5, 39511,
        
        35142.5, 71264, 71995.5, 72727, 37138.5,
        73230, 148732, 150233, 151734, 77316,
        76197.5, 154739, 156240, 157742, 80363.5,
        79165, 160746, 162248, 163750, 83411,
        41492.5, 84074, 84845.5, 85617, 43668.5,        
        };
    check(expect, host_diffdata, expect.size(), 1.f);

    cudnnDestroy(handle);
    cudaFree(data);
    cudaFree(out);
}

template<cudnnDataType_t T, typename HT = typename dt_trait<T>::type>
void test3() {
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

    cudnnConvolutionBwdFilterAlgo_t bwd_filter_a;
    cudnnGetConvolutionBackwardFilterAlgorithm(handle, dataTensor, diffoutTensor, covdes, difffilterTensor, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 10, &bwd_filter_a);

    size_t size;
    void *workspacesize;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(
        handle, 
        dataTensor,
        diffoutTensor, 
        covdes, 
        difffilterTensor, 
        bwd_filter_a, 
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
        bwd_filter_a, 
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

int main() {
    test1<CUDNN_DATA_FLOAT>();
    test2<CUDNN_DATA_FLOAT>();
    test3<CUDNN_DATA_FLOAT>();
    std::cout << "test passed" << std::endl;
    return 0;
}
