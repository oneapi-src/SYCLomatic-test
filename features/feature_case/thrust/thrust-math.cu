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
#include <functional>

__global__ void complex_log10_kernel(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::log10(c);
  *res = thrust::abs(c);  
}

__global__ void complex_sqrt_kernel(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::sqrt(c);
  *res = thrust::abs(c);  
}


__global__ void complex_pow1_kernel(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::pow(1.0f, c);
  *res = thrust::abs(c);  
}

__global__ void complex_pow2_kernel(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::pow(c, 1.0f);
  *res = thrust::abs(c);  
}

__global__ void complex_pow3_kernel(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::pow(c, c);
  *res = thrust::abs(c);  
}

__global__ void complex_sin_kernel(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::sin(c);
  *res = thrust::abs(c);  
}

__global__ void complex_cos_kernel(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::cos(c);
  *res = thrust::abs(c);  
}

void complex_log10(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::log10(c);
  *res = thrust::abs(c);  
}

void complex_sqrt(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::sqrt(c);
  *res = thrust::abs(c);  
}


void complex_pow1(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::pow(1.0f, c);
  *res = thrust::abs(c);  
}

void complex_pow2(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::pow(c, 1.0f);
  *res = thrust::abs(c);  
}

void complex_pow3(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::pow(c, c);
  *res = thrust::abs(c);  
}

void complex_sin(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::sin(c);
  *res = thrust::abs(c);  
}

void complex_cos(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::cos(c);
  *res = thrust::abs(c);  
}

bool test_math(){
  float *hostRes = (float *)malloc(sizeof(float));
  float *Res = (float *)malloc(sizeof(float));
  float *deviceRes;
  bool flag = true;
  cudaMalloc((float **)&deviceRes, sizeof(float));

  complex_log10_kernel<<<1, 1>>>(deviceRes);
  cudaMemcpy(hostRes, deviceRes, sizeof(float), cudaMemcpyDeviceToHost);
  complex_log10(Res);
  if(std::abs(*hostRes-*Res)>1e-6){
    std::cout<<"complex_log10 "<<" test failed \n";
    std::cout<<*hostRes<<'\t'<<*Res<<'\n';
  }
  else{
    std::cout<<"complex_log10 "<<"test pass \n";
  }
  flag = flag & (std::abs(*hostRes-*Res)>1e-6);

  complex_sqrt_kernel<<<1, 1>>>(deviceRes);
  cudaMemcpy(hostRes, deviceRes, sizeof(float), cudaMemcpyDeviceToHost);
  complex_sqrt(Res);
  if(std::abs(*hostRes-*Res)>1e-6){
    std::cout<<"complex_sqrt "<<" test failed \n";
    std::cout<<*hostRes<<'\t'<<*Res<<'\n';
  }
  else{
    std::cout<<"complex_sqrt "<<"test pass \n";
  }
  flag = flag & (std::abs(*hostRes-*Res)>1e-6);

  complex_pow1_kernel<<<1, 1>>>(deviceRes);
  cudaMemcpy(hostRes, deviceRes, sizeof(float), cudaMemcpyDeviceToHost);
  complex_pow1(Res);
  if(std::abs(*hostRes-*Res)>1e-6){
    std::cout<<"complex_pow1 "<<" test failed \n";
    std::cout<<*hostRes<<'\t'<<*Res<<'\n';
  }
  else{
    std::cout<<"complex_pow1 "<<"test pass \n";
  }
  flag = flag & (std::abs(*hostRes-*Res)>1e-6);

  complex_pow2_kernel<<<1, 1>>>(deviceRes);
  cudaMemcpy(hostRes, deviceRes, sizeof(float), cudaMemcpyDeviceToHost);
  complex_pow2(Res);
  if(std::abs(*hostRes-*Res)>1e-6){
    std::cout<<"complex_pow2 "<<" test failed \n";
    std::cout<<*hostRes<<'\t'<<*Res<<'\n';
  }
  else{
    std::cout<<"complex_pow2 "<<"test pass \n";
  }
  flag = flag & (std::abs(*hostRes-*Res)>1e-6);

  complex_pow3_kernel<<<1, 1>>>(deviceRes);
  cudaMemcpy(hostRes, deviceRes, sizeof(float), cudaMemcpyDeviceToHost);
  complex_pow3(Res);
  if(std::abs(*hostRes-*Res)>1e-6){
    std::cout<<"complex_pow3 "<<" test failed \n";
    std::cout<<*hostRes<<'\t'<<*Res<<'\n';
  }
  else{
    std::cout<<"complex_pow3 "<<"test pass \n";
  }
  flag = flag & (std::abs(*hostRes-*Res)>1e-6);

  complex_sin_kernel<<<1, 1>>>(deviceRes);
  cudaMemcpy(hostRes, deviceRes, sizeof(float), cudaMemcpyDeviceToHost);
  complex_sin(Res);
  if(std::abs(*hostRes-*Res)>1e-6){
    std::cout<<"complex_sin "<<" test failed \n";
    std::cout<<*hostRes<<'\t'<<*Res<<'\n';
  }
  else{
    std::cout<<"complex_sin "<<"test pass \n";
  }
  flag = flag & (std::abs(*hostRes-*Res)>1e-6);

  complex_cos_kernel<<<1, 1>>>(deviceRes);
  cudaMemcpy(hostRes, deviceRes, sizeof(float), cudaMemcpyDeviceToHost);
  complex_cos(Res);
  if(std::abs(*hostRes-*Res)>1e-6){
    std::cout<<"complex_cos "<<" test failed \n";
    std::cout<<*hostRes<<'\t'<<*Res<<'\n';
  }
  else{
    std::cout<<"complex_cos "<<"test pass \n";
  }
  flag = flag & (std::abs(*hostRes-*Res)>1e-6);

  free(hostRes);
  free(Res);
  cudaFree(deviceRes);
  return flag;
}

int main()
{
  if(test_math()) return 0;
  return 1;
}
