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

__global__ void complex_tan_kernel(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::tan(c);
  *res = thrust::abs(c);  
}

void complex_tan(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::tan(c);
  *res = thrust::abs(c);  
}

__global__ void complex_asin_kernel(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::asin(c);
  *res = thrust::abs(c);  
}

void complex_asin(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::asin(c);
  *res = thrust::abs(c);  
}

__global__ void complex_acos_kernel(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::acos(c);
  *res = thrust::abs(c);  
}

void complex_acos(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::acos(c);
  *res = thrust::abs(c);  
}

__global__ void complex_atan_kernel(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::atan(c);
  *res = thrust::abs(c);  
}

void complex_atan(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::atan(c);
  *res = thrust::abs(c);  
}

__global__ void complex_sinh_kernel(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::sinh(c);
  *res = thrust::abs(c);  
}

void complex_sinh(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::sinh(c);
  *res = thrust::abs(c);  
}

__global__ void complex_cosh_kernel(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::cosh(c);
  *res = thrust::abs(c);  
}

void complex_cosh(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::cosh(c);
  *res = thrust::abs(c);  
}

__global__ void complex_tanh_kernel(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::tanh(c);
  *res = thrust::abs(c);  
}

void complex_tanh(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::tanh(c);
  *res = thrust::abs(c);  
}

bool test_math(){
  float *hostRes = (float *)malloc(sizeof(float));
  float *Res = (float *)malloc(sizeof(float));
  float *deviceRes;
  bool flag = true;
  cudaMalloc((float **)&deviceRes, sizeof(float));

  complex_tan_kernel<<<1, 1>>>(deviceRes);
  cudaMemcpy(hostRes, deviceRes, sizeof(float), cudaMemcpyDeviceToHost);
  complex_tan(Res);
  if(std::abs(*hostRes-*Res)>1e-6){
    std::cout<<"complex_tan "<<" test failed \n";
    std::cout<<*hostRes<<'\t'<<*Res<<'\n';
  }
  else{
    std::cout<<"complex_tan "<<"test pass \n";
  }
  flag = flag & (std::abs(*hostRes-*Res)>1e-6);

  complex_asin_kernel<<<1, 1>>>(deviceRes);
  cudaMemcpy(hostRes, deviceRes, sizeof(float), cudaMemcpyDeviceToHost);
  complex_asin(Res);
  if(std::abs(*hostRes-*Res)>1e-6){
    std::cout<<"complex_asin "<<" test failed \n";
    std::cout<<*hostRes<<'\t'<<*Res<<'\n';
  }
  else{
    std::cout<<"complex_asin "<<"test pass \n";
  }
  flag = flag & (std::abs(*hostRes-*Res)>1e-6);

  complex_acos_kernel<<<1, 1>>>(deviceRes);
  cudaMemcpy(hostRes, deviceRes, sizeof(float), cudaMemcpyDeviceToHost);
  complex_acos(Res);
  if(std::abs(*hostRes-*Res)>1e-6){
    std::cout<<"complex_acos "<<" test failed \n";
    std::cout<<*hostRes<<'\t'<<*Res<<'\n';
  }
  else{
    std::cout<<"complex_acos "<<"test pass \n";
  }
  flag = flag & (std::abs(*hostRes-*Res)>1e-6);

  complex_atan_kernel<<<1, 1>>>(deviceRes);
  cudaMemcpy(hostRes, deviceRes, sizeof(float), cudaMemcpyDeviceToHost);
  complex_atan(Res);
  if(std::abs(*hostRes-*Res)>1e-6){
    std::cout<<"complex_atan "<<" test failed \n";
    std::cout<<*hostRes<<'\t'<<*Res<<'\n';
  }
  else{
    std::cout<<"complex_atan "<<"test pass \n";
  }
  flag = flag & (std::abs(*hostRes-*Res)>1e-6);

  complex_sinh_kernel<<<1, 1>>>(deviceRes);
  cudaMemcpy(hostRes, deviceRes, sizeof(float), cudaMemcpyDeviceToHost);
  complex_sinh(Res);
  if(std::abs(*hostRes-*Res)>1e-6){
    std::cout<<"complex_sinh "<<" test failed \n";
    std::cout<<*hostRes<<'\t'<<*Res<<'\n';
  }
  else{
    std::cout<<"complex_sinh "<<"test pass \n";
  }
  flag = flag & (std::abs(*hostRes-*Res)>1e-6);

  complex_cosh_kernel<<<1, 1>>>(deviceRes);
  cudaMemcpy(hostRes, deviceRes, sizeof(float), cudaMemcpyDeviceToHost);
  complex_cosh(Res);
  if(std::abs(*hostRes-*Res)>1e-6){
    std::cout<<"complex_cosh "<<" test failed \n";
    std::cout<<*hostRes<<'\t'<<*Res<<'\n';
  }
  else{
    std::cout<<"complex_cosh "<<"test pass \n";
  }
  flag = flag & (std::abs(*hostRes-*Res)>1e-6);

  complex_tanh_kernel<<<1, 1>>>(deviceRes);
  cudaMemcpy(hostRes, deviceRes, sizeof(float), cudaMemcpyDeviceToHost);
  complex_tanh(Res);
  if(std::abs(*hostRes-*Res)>1e-6){
    std::cout<<"complex_tanh "<<" test failed \n";
    std::cout<<*hostRes<<'\t'<<*Res<<'\n';
  }
  else{
    std::cout<<"complex_tanh "<<"test pass \n";
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
