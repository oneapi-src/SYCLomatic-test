// ====------ cudnn-normp1.cu ---------- *- CUDA -* ----===////
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
    cudnnTensorDescriptor_t dataTensor, outTensor, scalebiasTensor;
    cudnnCreate(&handle);

    cudnnCreateTensorDescriptor(&dataTensor);
    cudnnCreateTensorDescriptor(&outTensor);
    cudnnCreateTensorDescriptor(&scalebiasTensor);

    int in = 2, ic = 4, ih = 5, iw = 5;
    int on = 2, oc = 4, oh = 5, ow = 5;
    int sbn = 1, sbc = 4, sbh = 5, sbw = 5;
    int ele_num = in* ic * ih * iw;
    int oele_num = on* oc * oh * ow;
    int sele_num = sbn*sbc * sbh * sbw;
    cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in, ic, ih, iw);
    cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, on, oc, oh, ow);
    cudnnSetTensor4dDescriptor(scalebiasTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, sbn, sbc, sbh, sbw);

    int save = 1;
    float *data, *out, *scale, *bias, *rmean, *rvar, *smean, *svar, *z;
    std::vector<float> host_data(ele_num, 1.0f);
    std::vector<float> host_z(oele_num, 1.0f);
    std::vector<float> host_out(oele_num, 0.0f);
    std::vector<float> host_scale(sele_num, 1.0f);
    std::vector<float> host_bias(sele_num, 0.0f);
    std::vector<float> host_rmean(sele_num, 0.0f);
    std::vector<float> host_rvar(sele_num, 0.0f);
    std::vector<float> host_smean(save * sele_num, 0.0f);
    std::vector<float> host_svar(save * sele_num, 0.0f);

    for(int i = 0; i < ele_num; i++) {
        host_data[i] =  i + 4.f;
        host_out[i] = 1.f;
        host_z[i] = 10;
    }
    for(int i = 0; i < sele_num; i++) {
        host_scale[i] = i;
        host_bias[i] = i;
        host_rmean[i] = i;
        host_rvar[i] = i;
        host_smean[i] = i;
        host_svar[i] = i;
    }

    cudaMalloc(&data, sizeof(float) * ele_num);
    cudaMalloc(&z, sizeof(float) * oele_num);
    cudaMalloc(&out, sizeof(float) * oele_num);
    cudaMalloc(&scale, sizeof(float) * sele_num);
    cudaMalloc(&bias, sizeof(float) * sele_num);
    cudaMalloc(&rmean, sizeof(float) * sele_num);
    cudaMalloc(&rvar, sizeof(float) * sele_num);
    cudaMalloc(&smean, sizeof(float) * save*sele_num);
    cudaMalloc(&svar, sizeof(float)  * save*sele_num);

    cudaMemcpy(data, host_data.data(), sizeof(float) * ele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(z, host_z.data(), sizeof(float) * oele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(out, host_out.data(), sizeof(float) * oele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(scale, host_scale.data(), sizeof(float) * sele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(bias, host_bias.data(), sizeof(float) * sele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(rmean, host_rmean.data(), sizeof(float) * sele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(rvar, host_rvar.data(), sizeof(float) * sele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(smean, host_smean.data(),  sizeof(float) * save * sele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(svar, host_svar.data(), sizeof(float) * save * sele_num, cudaMemcpyHostToDevice);

    float alpha = 2.5f, beta = 1.5f, eps = 1.f;
    double factor = 0.5f;
    cudnnActivationDescriptor_t ActivationDesc;
    cudnnCreateActivationDescriptor(&ActivationDesc);
    cudnnSetActivationDescriptor(ActivationDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0f);

    auto status = cudnnNormalizationForwardInference(
        handle, 
        CUDNN_NORM_PER_ACTIVATION,
        CUDNN_NORM_OPS_NORM,
        CUDNN_NORM_ALGO_STANDARD,
        &alpha,
        &beta,
        dataTensor,
        data,
        scalebiasTensor,
        scale,
        bias,
        scalebiasTensor,
        smean,
        svar,
        dataTensor,
        z,
        ActivationDesc,
        outTensor,
        out,
        eps,
        1);

    cudaDeviceSynchronize();
    cudaMemcpy(host_out.data(), out, sizeof(float) * oele_num, cudaMemcpyDeviceToHost);

    std::vector<float> expect = {
        1.5, 11.0711, 18.047, 24, 29.3885,
        34.4124, 39.1779, 43.7487, 48.1667, 52.4605,
        56.6511, 60.7543, 64.782, 68.744, 72.6478,
        76.5, 80.3057, 84.0694, 87.7948, 91.4853,
        95.1436, 98.7721, 102.373, 105.949, 109.5,
        
        113.029, 116.537, 120.025, 123.495, 126.947,
        130.382, 133.801, 137.205, 140.595, 143.97,
        147.333, 150.684, 154.022, 157.349, 160.664,
        163.969, 167.264, 170.549, 173.825, 177.091,
        180.349, 183.598, 186.839, 190.071, 193.296,
        
        196.514, 199.724, 202.927, 206.124, 209.314,
        212.497, 215.674, 218.845, 222.01, 225.169,
        228.322, 231.47, 234.613, 237.75, 240.882,
        244.009, 247.132, 250.249, 253.362, 256.471,
        259.575, 262.674, 265.77, 268.861, 271.948,
        
        275.031, 278.11, 281.185, 284.257, 287.325,
        290.389, 293.45, 296.507, 299.56, 302.611,
        305.658, 308.702, 311.742, 314.78, 317.814,
        320.846, 323.874, 326.9, 329.922, 332.942,
        335.959, 338.973, 341.985, 344.994, 348,

        1.5, 187.848, 306.722, 399, 476.602,
        544.723, 606.125, 662.467, 714.833, 763.973,
        810.43, 854.611, 896.832, 937.343, 976.344,
        1014, 1050.45, 1085.8, 1120.17, 1153.62,
        1186.23, 1218.08, 1249.2, 1279.66, 1309.5,
        
        1338.75, 1367.46, 1395.66, 1423.36, 1450.61,
        1477.42, 1503.82, 1529.83, 1555.46, 1580.73,
        1605.67, 1630.27, 1654.57, 1678.57, 1702.27,
        1725.71, 1748.87, 1771.78, 1794.45, 1816.87,
        1839.07, 1861.05, 1882.81, 1904.36, 1925.71,
        
        1946.86, 1967.83, 1988.61, 2009.22, 2029.65,
        2049.92, 2070.02, 2089.96, 2109.75, 2129.39,
        2148.88, 2168.22, 2187.43, 2206.5, 2225.44,
        2244.25, 2262.93, 2281.49, 2299.92, 2318.24,
        2336.44, 2354.53, 2372.51, 2390.38, 2408.14,
        
        2425.8, 2443.36, 2460.82, 2478.18, 2495.44,
        2512.61, 2529.69, 2546.67, 2563.57, 2580.38,
        2597.1, 2613.74, 2630.3, 2646.78, 2663.17,
        2679.49, 2695.73, 2711.89, 2727.98, 2743.99,
        2759.93, 2775.8, 2791.6, 2807.34, 2823,        
        };
    check(expect, host_out, expect.size(), 1e-1);

    cudnnDestroy(handle);
    cudaFree(data);
    cudaFree(out);
}

template<cudnnDataType_t T, typename HT = typename dt_trait<T>::type>
void test2() {
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor, scalebiasTensor;
    cudnnCreate(&handle);

    cudnnCreateTensorDescriptor(&dataTensor);
    cudnnCreateTensorDescriptor(&outTensor);
    cudnnCreateTensorDescriptor(&scalebiasTensor);

    int in = 2, ic = 4, ih = 5, iw = 5;
    int on = 2, oc = 4, oh = 5, ow = 5;
    int sbn = 1, sbc = 4, sbh = 1, sbw = 1;
    int ele_num = in* ic * ih * iw;
    int oele_num = on* oc * oh * ow;
    int sele_num = sbn*sbc * sbh * sbw;
    cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in, ic, ih, iw);
    cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, on, oc, oh, ow);
    cudnnSetTensor4dDescriptor(scalebiasTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, sbn, sbc, sbh, sbw);

    int save = 1;
    float *data, *out, *scale, *bias, *rmean, *rvar, *smean, *svar, *z;
    std::vector<float> host_data(ele_num, 1.0f);
    std::vector<float> host_z(oele_num, 1.0f);
    std::vector<float> host_out(oele_num, 0.0f);
    std::vector<float> host_scale(sele_num, 1.0f);
    std::vector<float> host_bias(sele_num, 0.0f);
    std::vector<float> host_rmean(sele_num, 0.0f);
    std::vector<float> host_rvar(sele_num, 0.0f);
    std::vector<float> host_smean(save * sele_num, 0.0f);
    std::vector<float> host_svar(save * sele_num, 0.0f);

    for(int i = 0; i < ele_num; i++) {
        host_data[i] =  i + 4.f;
        host_out[i] = 1.f;
        host_z[i] = 10;
    }
    for(int i = 0; i < sele_num; i++) {
        host_scale[i] = i;
        host_bias[i] = i;
        host_rmean[i] = i;
        host_rvar[i] = i;
        host_smean[i] = i;
        host_svar[i] = i;
    }

    cudaMalloc(&data, sizeof(float) * ele_num);
    cudaMalloc(&z, sizeof(float) * oele_num);
    cudaMalloc(&out, sizeof(float) * oele_num);
    cudaMalloc(&scale, sizeof(float) * sele_num);
    cudaMalloc(&bias, sizeof(float) * sele_num);
    cudaMalloc(&rmean, sizeof(float) * sele_num);
    cudaMalloc(&rvar, sizeof(float) * sele_num);
    cudaMalloc(&smean, sizeof(float) * save*sele_num);
    cudaMalloc(&svar, sizeof(float)  * save*sele_num);

    cudaMemcpy(data, host_data.data(), sizeof(float) * ele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(z, host_z.data(), sizeof(float) * oele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(out, host_out.data(), sizeof(float) * oele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(scale, host_scale.data(), sizeof(float) * sele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(bias, host_bias.data(), sizeof(float) * sele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(rmean, host_rmean.data(), sizeof(float) * sele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(rvar, host_rvar.data(), sizeof(float) * sele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(smean, host_smean.data(),  sizeof(float) * save * sele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(svar, host_svar.data(), sizeof(float) * save * sele_num, cudaMemcpyHostToDevice);

    float alpha = 2.5f, beta = 1.5f, eps = 1.f;
    double factor = 0.5f;
    cudnnActivationDescriptor_t ActivationDesc;
    cudnnCreateActivationDescriptor(&ActivationDesc);
    cudnnSetActivationDescriptor(ActivationDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0f);

    auto status = cudnnNormalizationForwardInference(
        handle,
        CUDNN_NORM_PER_CHANNEL,
        CUDNN_NORM_OPS_NORM,
        CUDNN_NORM_ALGO_STANDARD,
        &alpha,
        &beta,
        dataTensor,
        data,
        scalebiasTensor,
        scale,
        bias,
        scalebiasTensor,
        smean,
        svar,
        dataTensor,
        z,
        ActivationDesc,
        outTensor,
        out,
        eps,
        1);

    cudaDeviceSynchronize();
    cudaMemcpy(host_out.data(), out, sizeof(float) * oele_num, cudaMemcpyDeviceToHost);

    std::vector<float> expect = {
        1.5, 1.5, 1.5, 1.5, 1.5,
        1.5, 1.5, 1.5, 1.5, 1.5,
        1.5, 1.5, 1.5, 1.5, 1.5,
        1.5, 1.5, 1.5, 1.5, 1.5,
        1.5, 1.5, 1.5, 1.5, 1.5,
        
        53.4975, 55.2652, 57.033, 58.8008, 60.5685,
        62.3363, 64.1041, 65.8718, 67.6396, 69.4074,
        71.1751, 72.9429, 74.7107, 76.4784, 78.2462,
        80.014, 81.7817, 83.5495, 85.3173, 87.085,
        88.8528, 90.6206, 92.3883, 94.1561, 95.9239,
        
        156.611, 159.498, 162.385, 165.271, 168.158,
        171.045, 173.932, 176.818, 179.705, 182.592,
        185.479, 188.365, 191.252, 194.139, 197.026,
        199.912, 202.799, 205.686, 208.573, 211.459,
        214.346, 217.233, 220.12, 223.006, 225.893,
        
        294, 297.75, 301.5, 305.25, 309,
        312.75, 316.5, 320.25, 324, 327.75,
        331.5, 335.25, 339, 342.75, 346.5,
        350.25, 354, 357.75, 361.5, 365.25,
        369, 372.75, 376.5, 380.25, 384,

        1.5, 1.5, 1.5, 1.5, 1.5,
        1.5, 1.5, 1.5, 1.5, 1.5,
        1.5, 1.5, 1.5, 1.5, 1.5,
        1.5, 1.5, 1.5, 1.5, 1.5,
        1.5, 1.5, 1.5, 1.5, 1.5,
        
        230.274, 232.042, 233.81, 235.577, 237.345,
        239.113, 240.881, 242.649, 244.416, 246.184,
        247.952, 249.72, 251.487, 253.255, 255.023,
        256.791, 258.558, 260.326, 262.094, 263.862,
        265.629, 267.397, 269.165, 270.933, 272.701,
        
        445.286, 448.173, 451.06, 453.946, 456.833,
        459.72, 462.607, 465.493, 468.38, 471.267,
        474.154, 477.04, 479.927, 482.814, 485.701,
        488.587, 491.474, 494.361, 497.248, 500.134,
        503.021, 505.908, 508.795, 511.681, 514.568,

        669, 672.75, 676.5, 680.25, 684,
        687.75, 691.5, 695.25, 699, 702.75,
        706.5, 710.25, 714, 717.75, 721.5,
        725.25, 729, 732.75, 736.5, 740.25,
        744, 747.75, 751.5, 755.25, 759,              
        };
    check(expect, host_out, expect.size(), 1e-1);

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
