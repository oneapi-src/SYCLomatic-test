// ====------ cuda_event_record_with_flags.cu------ *- CUDA -*-------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <cuda_runtime.h>
#include <iostream>

__global__ void simpleKernel() { printf("Hello simpleKernel\n"); }

int main() {
  cudaEvent_t start, stop;
  cudaStream_t stream;

  // Create a stream
  cudaStreamCreate(&stream);

  // Create events
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Record start event with flags
  cudaEventRecordWithFlags(start, stream, cudaEventRecordDefault);

  // Launch a simple kernel in the stream
  simpleKernel<<<1, 1, 0, stream>>>();
  cudaDeviceSynchronize();

  // Record stop event with flags
  cudaEventRecordWithFlags(stop, stream, cudaEventRecordDefault);

  // Wait for the event to complete
  cudaEventSynchronize(stop);

  // Calculate elapsed time
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Elapsed time: " << milliseconds << " ms" << std::endl;

  // Clean up
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);

  return 0;
}
