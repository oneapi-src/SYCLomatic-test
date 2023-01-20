// ====------ thrust_raw_reference_cast.cu---------- *- CUDA -* -------------------===//
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

int main(void) {

  int num_failures = 0;

  thrust::host_vector<int> h_vec(1);
  thrust::device_vector<int> d_vec = h_vec;
  d_vec[0] = 13;

  thrust::device_reference<int> ref_to_thirteen = d_vec[0];
  int &ref = thrust::raw_reference_cast(ref_to_thirteen);

  if (ref != 13) {
    std::cout << "get_raw_reference test 1 failed.\n";
    num_failures++;
  }

  ref = 14;
  if (ref_to_thirteen != 14) {
    std::cout << "get_raw_reference test 2 failed.\n";
    num_failures++;
  }

  int &ref2 = thrust::raw_reference_cast(ref);
  ref2 = 15;
  if (ref != 15) {
    std::cout << "get_raw_reference test 3 failed.\n";
    num_failures++;
  }

  if (num_failures == 0)
    std::cout << "get_raw_reference test passed.\n";
  return 0;
}
