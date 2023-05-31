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

__global__ void complex_asinh_kernel(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::asinh(c);
  *res = thrust::abs(c);  
}

void complex_asinh(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::asinh(c);
  *res = thrust::abs(c);  
}

__global__ void complex_acosh_kernel(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::acosh(c);
  *res = thrust::abs(c);  
}

void complex_acosh(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::acosh(c);
  *res = thrust::abs(c);  
}

__global__ void complex_atanh_kernel(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::atanh(c);
  *res = thrust::abs(c);  
}

void complex_atanh(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::atanh(c);
  *res = thrust::abs(c);  
}

__global__ void complex_log_kernel(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::log(c);
  *res = thrust::abs(c);  
}

void complex_log(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::log(c);
  *res = thrust::abs(c);  
}

__global__ void complex_exp_kernel(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::exp(c);
  *res = thrust::abs(c);  
}

void complex_exp(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::exp(c);
  *res = thrust::abs(c);  
}

__global__ void complex_proj_kernel(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::proj(c);
  *res = thrust::abs(c);  
}

void complex_proj(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::proj(c);
  *res = thrust::abs(c);  
}

__global__ void complex_conj_kernel(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::conj(c);
  *res = thrust::abs(c);  
}

void complex_conj(float * res)
{
  auto c = thrust::polar(11.48569f, 1.33698f);
  c = thrust::conj(c);
  *res = thrust::abs(c);  
}

bool test_math(){
  float *hostRes = (float *)malloc(sizeof(float));
  float *Res = (float *)malloc(sizeof(float));
  float *deviceRes;
  bool flag = true;
  cudaMalloc((float **)&deviceRes, sizeof(float));

  complex_asinh_kernel<<<1, 1>>>(deviceRes);
  cudaMemcpy(hostRes, deviceRes, sizeof(float), cudaMemcpyDeviceToHost);
  complex_asinh(Res);
  if(std::abs(*hostRes-*Res)>1e-6){
    std::cout<<"complex_asinh "<<" test failed \n";
    std::cout<<*hostRes<<'\t'<<*Res<<'\n';
  }
  else{
    std::cout<<"complex_asinh "<<"test pass \n";
  }
  flag = flag & (std::abs(*hostRes-*Res)>1e-6);

  complex_acosh_kernel<<<1, 1>>>(deviceRes);
  cudaMemcpy(hostRes, deviceRes, sizeof(float), cudaMemcpyDeviceToHost);
  complex_acosh(Res);
  if(std::abs(*hostRes-*Res)>1e-6){
    std::cout<<"complex_acosh "<<" test failed \n";
    std::cout<<*hostRes<<'\t'<<*Res<<'\n';
  }
  else{
    std::cout<<"complex_acosh "<<"test pass \n";
  }
  flag = flag & (std::abs(*hostRes-*Res)>1e-6);
  
  complex_atanh_kernel<<<1, 1>>>(deviceRes);
  cudaMemcpy(hostRes, deviceRes, sizeof(float), cudaMemcpyDeviceToHost);
  complex_atanh(Res);
  if(std::abs(*hostRes-*Res)>1e-6){
    std::cout<<"complex_atanh "<<" test failed \n";
    std::cout<<*hostRes<<'\t'<<*Res<<'\n';
  }
  else{
    std::cout<<"complex_atanh "<<"test pass \n";
  }
  flag = flag & (std::abs(*hostRes-*Res)>1e-6);

  complex_log_kernel<<<1, 1>>>(deviceRes);
  cudaMemcpy(hostRes, deviceRes, sizeof(float), cudaMemcpyDeviceToHost);
  complex_log(Res);
  if(std::abs(*hostRes-*Res)>1e-6){
    std::cout<<"complex_log "<<" test failed \n";
    std::cout<<*hostRes<<'\t'<<*Res<<'\n';
  }
  else{
    std::cout<<"complex_log "<<"test pass \n";
  }
  flag = flag & (std::abs(*hostRes-*Res)>1e-6);

  complex_exp_kernel<<<1, 1>>>(deviceRes);
  cudaMemcpy(hostRes, deviceRes, sizeof(float), cudaMemcpyDeviceToHost);
  complex_exp(Res);
  if(std::abs(*hostRes-*Res)>1e-6){
    std::cout<<"complex_exp "<<" test failed \n";
    std::cout<<*hostRes<<'\t'<<*Res<<'\n';
  }
  else{
    std::cout<<"complex_exp "<<"test pass \n";
  }
  flag = flag & (std::abs(*hostRes-*Res)>1e-6);

  complex_proj_kernel<<<1, 1>>>(deviceRes);
  cudaMemcpy(hostRes, deviceRes, sizeof(float), cudaMemcpyDeviceToHost);
  complex_proj(Res);
  if(std::abs(*hostRes-*Res)>1e-6){
    std::cout<<"complex_proj "<<" test failed \n";
    std::cout<<*hostRes<<'\t'<<*Res<<'\n';
  }
  else{
    std::cout<<"complex_proj "<<"test pass \n";
  }
  flag = flag & (std::abs(*hostRes-*Res)>1e-6);

  complex_conj_kernel<<<1, 1>>>(deviceRes);
  cudaMemcpy(hostRes, deviceRes, sizeof(float), cudaMemcpyDeviceToHost);
  complex_conj(Res);
  if(std::abs(*hostRes-*Res)>1e-6){
    std::cout<<"complex_conj "<<" test failed \n";
    std::cout<<*hostRes<<'\t'<<*Res<<'\n';
  }
  else{
    std::cout<<"complex_conj "<<"test pass \n";
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
