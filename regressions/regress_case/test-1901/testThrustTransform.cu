// ====------ testThrustTransform.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include<cuda_runtime.h>
#include<iostream>



struct my_math
{
  __host__ __device__
  int operator()(const int &r) const{
   return r+1;
  }
};



int main(){


cudaStream_t stream;
cudaStreamCreate(&stream);

int* host;
cudaHostAlloc((void**)&host, 10 * sizeof(int), cudaHostAllocDefault);
for(int i=0;i<10;i++)
  host[i]=i;

int *dev_a, *dev_b;
cudaMalloc(&dev_a,10*sizeof(int));
cudaMalloc(&dev_b,10*sizeof(int));

cudaMemcpyAsync(dev_a,host,10*sizeof(int),cudaMemcpyHostToDevice,stream);

my_math c;
thrust::transform(thrust::cuda::par.on(stream),dev_a,dev_a + 10,dev_b,c);

cudaMemcpyAsync(host,dev_b,10*sizeof(int),cudaMemcpyDeviceToHost,stream);

cudaStreamSynchronize(stream);
for(int i=0;i<10;i++)
 std::cout<<host[i]<<std::endl;

return 0;


}