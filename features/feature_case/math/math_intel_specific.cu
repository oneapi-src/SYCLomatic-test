// ====------ math_intel_specific.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
// RUN: dpct --rule-file=%S/../../tools/dpct/DpctOptRules/intel_specific_math.yaml --format-range=none -out-root %T/math_specific_UDR_test %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only

#include "cuda_fp16.h"
#include <iostream>

__global__ void kernelFunc(double *deviceArray) {
  double &d0 = *deviceArray;
  d0 = erfinv(d0);
  d0 = normcdf(d0);

}

__global__ void kernelFunc(float *deviceArray) {
  float &f0 = *deviceArray;
  f0 = erfinvf(f0);
  f0 = normcdff(f0);

}

bool testDouble() {
  double *hostArrayDouble = (double *)malloc(sizeof(double));
  *hostArrayDouble = 0.956841;
  double *deviceArrayDouble;
  cudaMalloc((double **)&deviceArrayDouble, sizeof(double));
  cudaMemcpy(deviceArrayDouble, hostArrayDouble, sizeof(double), cudaMemcpyHostToDevice);
  kernelFunc<<<1, 1>>>(deviceArrayDouble);
  cudaMemcpy(hostArrayDouble, deviceArrayDouble, sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(deviceArrayDouble);
  if((*hostArrayDouble-0.923625516883796)>1e-6){
    std::cout << "test on double failed" << std::endl;
    return false;
  }
  free(hostArrayDouble);
  return true;
}

bool testFloat() {
  float *hostArrayFloat = (float *)malloc(sizeof(float));
  *hostArrayFloat = 0.1568541541f;
  float *deviceArrayFloat;
  cudaMalloc((float **)&deviceArrayFloat,sizeof(float));
  cudaMemcpy(deviceArrayFloat, hostArrayFloat, sizeof(float), cudaMemcpyHostToDevice);
  kernelFunc<<<1, 1>>>(deviceArrayFloat);
  cudaMemcpy(hostArrayFloat, deviceArrayFloat, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(deviceArrayFloat);
  if((*hostArrayFloat- 0.555636882781982)>1e-6){
    std::cout << "test on float failed" << std::endl;
    return false;
  }
  free(hostArrayFloat);
  return true;
}




int main() {
  if(testDouble()&&testFloat()) return 0;
  return 1;
}