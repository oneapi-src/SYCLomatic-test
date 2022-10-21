// ====------ libcu_atomic.cu----------- *- CUDA -*  ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <cuda/atomic>

__global__ void example_kernel(int* data) {
  *data = 42;
  cuda::atomic_thread_fence(cuda::std::memory_order_release,
                            cuda::thread_scope_device);
}

int main(){
  cuda::atomic<int> a;
  cuda::atomic<int> b(0);
  cuda::atomic<int, cuda::thread_scope_block> c(0);
  int ans = c.load();
  a.store(0);
  int ans2 = a.load();

  int tmp =1,tmp1=2;
  ans = a.exchange(1);
  a.compare_exchange_weak(tmp,2);
  a.compare_exchange_strong(tmp1,3);

  ans = a.fetch_add(1);
  ans = a.fetch_sub(-1);
  int data_host = 0;
  int * data_ptr_device; 
  cudaMalloc(&data_ptr_device, sizeof(unsigned int));
  cudaMemcpy(data_ptr_device, &data_host, sizeof(unsigned int) , cudaMemcpyHostToDevice);
  example_kernel<<<1,1>>>(data_ptr_device) ;
  cudaMemcpy(&data_host, data_ptr_device, sizeof(unsigned int) , cudaMemcpyDeviceToHost);
  cudaFree(data_ptr_device);
  if(data_host==42) 
    return 0;
  return 1;
}