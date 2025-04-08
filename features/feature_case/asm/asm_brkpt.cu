// ====------ asm_brkpt.cu -------------------------------- *- CUDA -* ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <cuda_runtime.h>
#include <iostream>

__global__ void testKernel() {
  asm("brkpt;"); // PTX breakpoint instruction
}

int main() {
  std::cout << "Launching kernel with PTX breakpoint..." << std::endl;
  testKernel<<<1, 1>>>();
  cudaError_t err = cudaDeviceSynchronize();

  if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    return 1;
  } else {
    std::cout << "Kernel execution completed successfully." << std::endl;
    return 0;
  }
}
