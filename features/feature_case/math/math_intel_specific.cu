// ====------ math_intel_specific.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
// RUN: dpct --rule-file=%S/../../tools/dpct/DpctOptRules/intel_specific_math.yaml --format-range=none -out-root %T/math_specific_UDR_test %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/math_specific_UDR_test/math_specific_UDR_test.dp.cpp --match-full-lines %s

// CHECK: #include <CL/sycl.hpp>
// CHECK: #include <dpct/dpct.hpp>

// CHECK: #include <sycl/ext/intel/math.hpp>

#include "cuda_fp16.h"
#include <iostream>
// CHECK: void kernelFunc(double *deviceArray) {
// CHECK:   double &d0 = *deviceArray;
// CHECK:   d0 = sycl::ext::intel::math::erfinv(d0);
// CHECK:   d0 = sycl::ext::intel::math::cdfnorm(d0);
// CHECK: }
__global__ void kernelFunc(double *deviceArray) {
  double &d0 = *deviceArray;
  d0 = erfinv(d0);
  d0 = normcdf(d0);

}

// CHECK: void kernelFunc(float *deviceArray) {
// CHECK:   float &f0 = *deviceArray;
// CHECK:   f0 = sycl::ext::intel::math::erfinv(f0);
// CHECK:   f0 = sycl::ext::intel::math::cdfnorm(f0);
// CHECK: }
__global__ void kernelFunc(float *deviceArray) {
  float &f0 = *deviceArray;
  f0 = erfinvf(f0);
  f0 = normcdff(f0);

}


// CHECK: void testDouble() {
// CHECK:   dpct::device_ext &dev_ct1 = dpct::get_current_device();
// CHECK:   sycl::queue &q_ct1 = dev_ct1.default_queue();
// CHECK:   const unsigned int NUM = 1;
// CHECK:   const unsigned int bytes = NUM * sizeof(double);
// CHECK:   double *hostArrayDouble = (double *)malloc(bytes);
// CHECK:   memset(hostArrayDouble, 0, bytes);
// CHECK:   double *deviceArrayDouble;
// CHECK:   deviceArrayDouble = (double *)sycl::malloc_device(bytes, q_ct1);
// CHECK:   q_ct1.memcpy(deviceArrayDouble, hostArrayDouble, bytes).wait();
// CHECK:   q_ct1.parallel_for(
// CHECK:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
// CHECK:       [=](sycl::nd_item<3> item_ct1) {
// CHECK:         kernelFunc(deviceArrayDouble);
// CHECK:       });
// CHECK:   q_ct1.memcpy(hostArrayDouble, deviceArrayDouble, bytes).wait();
// CHECK:   sycl::free(deviceArrayDouble, q_ct1);
// CHECK: }
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

// CHECK: void testFloat() {
// CHECK:   dpct::device_ext &dev_ct1 = dpct::get_current_device();
// CHECK:   sycl::queue &q_ct1 = dev_ct1.default_queue();
// CHECK:   const unsigned int NUM = 1;
// CHECK:   const unsigned int bytes = NUM * sizeof(float);
// CHECK:   float *hostArrayFloat = (float *)malloc(bytes);
// CHECK:   memset(hostArrayFloat, 0, bytes);
// CHECK:   float *deviceArrayFloat;
// CHECK:   deviceArrayFloat = (float *)sycl::malloc_device(bytes, q_ct1);
// CHECK:   q_ct1.memcpy(deviceArrayFloat, hostArrayFloat, bytes).wait();
// CHECK:   q_ct1.parallel_for(
// CHECK:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
// CHECK:       [=](sycl::nd_item<3> item_ct1) {
// CHECK:         kernelFunc(deviceArrayFloat);
// CHECK:       });
// CHECK:   q_ct1.memcpy(hostArrayFloat, deviceArrayFloat, bytes).wait();
// CHECK:   sycl::free(deviceArrayFloat, q_ct1);
// CHECK: }
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