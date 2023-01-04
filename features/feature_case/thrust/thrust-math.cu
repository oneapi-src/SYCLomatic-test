// ====------ thrust-vector.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <thrust/complex.h>
#include <iostream>

__global__ void complex_math_kernel(float * res)
{
  auto c = thrust::polar(11.48569, 1.33698);

  c = thrust::log10(c);
  c = thrust::sqrt(c);
  c = thrust::pow(1.0, c);
  c = thrust::pow(c, 1.0);
  c = thrust::pow(c, c);
  c = thrust::sin(c);
  c = thrust::cos(c);
  c = thrust::tan(c);
  c = thrust::asin(c);
  c = thrust::acos(c);
  c = thrust::atan(c);
  c = thrust::sinh(c);
  c = thrust::cosh(c);
  c = thrust::tanh(c);
  c = thrust::asinh(c);
  c = thrust::acosh(c);
  c = thrust::atanh(c);
  c = thrust::log(c);
  c = thrust::exp(c);
  c = thrust::proj(thrust::norm(c));
  c = thrust::conj(c);
  *res = thrust::abs(c);  
}

void complex_math(float * res)
{
  auto c = thrust::polar(11.48569, 1.33698);

  c = thrust::log10(c);
  c = thrust::sqrt(c);
  c = thrust::pow(1.0, c);
  c = thrust::pow(c, 1.0);
  c = thrust::pow(c, c);
  c = thrust::sin(c);
  c = thrust::cos(c);
  c = thrust::tan(c);
  c = thrust::asin(c);
  c = thrust::acos(c);
  c = thrust::atan(c);
  c = thrust::sinh(c);
  c = thrust::cosh(c);
  c = thrust::tanh(c);
  c = thrust::asinh(c);
  c = thrust::acosh(c);
  c = thrust::atanh(c);
  c = thrust::log(c);
  c = thrust::exp(c);
  c = thrust::proj(thrust::norm(c));
  c = thrust::conj(c);
  *res = thrust::abs(c);  
}

bool test_math(){
  float *hostRes = (float *)malloc(sizeof(float));
  float *Res = (float *)malloc(sizeof(float));
  float *deviceRes;
  cudaMalloc((float **)&deviceRes, sizeof(float));
  complex_math_kernel<<<1, 1>>>(deviceRes);
  cudaMemcpy(hostRes, deviceRes, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(deviceRes);
  complex_math(Res);
  if(std::abs(*hostRes-*Res)>1e-6){
    std::cout<<*hostRes<<'\t'<<*Res<<'\n';
    std::cout<<"test failed \n";
    free(hostRes);
    free(Res);
    return false;
  }
  else{
    std::cout<<"test pass \n";
    free(hostRes);
    free(Res);
    return true;
  }
}

int main()
{
  if(test_math()) return 0;
  return 1;
}
