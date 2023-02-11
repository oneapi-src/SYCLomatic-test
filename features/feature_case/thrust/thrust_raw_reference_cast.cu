// ====------ thrust_raw_reference_cast.cu---------- *- CUDA -*
// -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===------------------------------------------------------------------------------===//

#include <cuda.h>
#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1

int main(void) {
  thrust::host_vector<int> h_vec(1);
  thrust::device_vector<int> d_vec = h_vec;
  d_vec[0] = 13;

  thrust::device_reference<int> ref_to_thirteen = d_vec[0];
  int &ref = thrust::raw_reference_cast(ref_to_thirteen);

  int val = 0;
  cudaError_t err = cudaMemcpy(&val, &ref, sizeof(int), cudaMemcpyDeviceToHost);

  if (err != cudaSuccess) {
    exit(EXIT_FAILURE);
  }

  if (val != 13) {
    std::cout << "get_raw_reference test failed.\n";
    return EXIT_FAILURE;
  }

  std::cout << "get_raw_reference test passed.\n";
  return EXIT_SUCCESS;
}
