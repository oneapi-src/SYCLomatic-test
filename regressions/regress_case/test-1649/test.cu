// ====------ test.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include "test.h"
#include <cuda_runtime.h>

__host__ __device__ static int Env_cuda_thread_in_threadblock(int axis)
{
  int a = 1;
#ifdef __CUDA_ARCH__
  return axis==0 ? threadIdx.x :
         axis==1 ? threadIdx.y :
                   threadIdx.z;
#else
  cudaDeviceSynchronize();
  return 0;
#endif

#if !defined(__CUDA_ARCH__)
  cudaDeviceSynchronize();
  return 0;
#else
  return axis==0 ? threadIdx.x :
         axis==1 ? threadIdx.y :
                   threadIdx.z;
#endif

#if defined(__CUDA_ARCH__)
  return axis==0 ? threadIdx.x :
         axis==1 ? threadIdx.y :
                   threadIdx.z;
#else
  cudaDeviceSynchronize();
  return 0;
#endif

  return a;
}

__host__ __device__ static int Env_cuda_thread_in_threadblock1(int axis)
{
  int a = 1;
#if defined(__CUDA_ARCH__)
  return axis==0 ? threadIdx.x :
         axis==1 ? threadIdx.y :
                   threadIdx.z;
#else
  cudaDeviceSynchronize();
  return 0;
#endif
  return a;
}

__host__ __device__ static int Env_cuda_thread_in_threadblock2(int axis)
{
  int a = 1;
#if __CUDA_ARCH__
  return axis==0 ? threadIdx.x :
         axis==1 ? threadIdx.y :
                   threadIdx.z;
#else
  cudaDeviceSynchronize();
  return 0;
#endif
  return a;
}

__host__ __device__ static int Env_cuda_thread_in_threadblock3(int axis)
{
  int a = 1;
#ifndef __CUDA_ARCH__
  cudaDeviceSynchronize();
  return 0;
#else
  return axis==0 ? threadIdx.x :
         axis==1 ? threadIdx.y :
                   threadIdx.z;
#endif
  return a;
}

__host__ __device__ static int Env_cuda_thread_in_threadblock4(int axis)
{
  int a = 1;
#if !defined(__CUDA_ARCH__)
  cudaDeviceSynchronize();
  return 0;
#else
  return axis==0 ? threadIdx.x :
         axis==1 ? threadIdx.y :
                   threadIdx.z;
#endif
  return a;
}

template<typename T>
__host__ __device__ int test(T a, T b);

template<typename T>
__host__ __device__ int test(T a, T b){
#ifdef __CUDA_ARCH__
  return threadIdx.x > 10 ? a : b;
#else
  cudaDeviceSynchronize();
  return a;
#endif
}

__host__ __device__ int test1(){
  #if __CUDA_ARCH__ > 800
    return threadIdx.x > 8;
  #elif __CUDA_ARCH__ > 700
    return threadIdx.x > 7;
  #elif __CUDA_ARCH__ > 600
    return threadIdx.x > 6;
  #elif __CUDA_ARCH__ > 500
    return threadIdx.x > 5;
  #elif __CUDA_ARCH__ > 400
    return threadIdx.x > 4;
  #elif __CUDA_ARCH__ > 300
    return threadIdx.x > 3;
  #elif __CUDA_ARCH__ > 200
    return threadIdx.x > 2;
  #elif !defined(__CUDA_ARCH__)
    cudaDeviceSynchronize();
    return 0;
  #endif
}

__global__ void kernel(){
  float a, b;
  Env_cuda_thread_in_threadblock(0);
  Env_cuda_thread_in_threadblock1(0);
  Env_cuda_thread_in_threadblock2(0);
  Env_cuda_thread_in_threadblock3(0);
  Env_cuda_thread_in_threadblock4(0);
  test(0, 0);
  test<float>(a, b);
  test1();
}

int main(){
float a, b;
test(a, b);
test<int>(1, 1);
Env_cuda_thread_in_threadblock(0);
Env_cuda_thread_in_threadblock1(0);
Env_cuda_thread_in_threadblock2(0);
Env_cuda_thread_in_threadblock3(0);
Env_cuda_thread_in_threadblock4(0);
test1();
kernel<<<1,1>>>();
cudaDeviceSynchronize();

return 0;
}
