// ====------ memory_async_dpct_free.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#define VECTOR_SIZE 256

void VectorAddKernel(float *A, float *B, float *C, sycl::nd_item<3> item_ct1) {
  A[item_ct1.get_local_id(2)] = item_ct1.get_local_id(2) + 1.0f;
  B[item_ct1.get_local_id(2)] = item_ct1.get_local_id(2) + 1.0f;
  C[item_ct1.get_local_id(2)] =
      A[item_ct1.get_local_id(2)] + B[item_ct1.get_local_id(2)];
}

int main() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  float *d_A, *d_B, *d_C;

  d_A = sycl::malloc_device<float>(VECTOR_SIZE, q_ct1);
  d_B = sycl::malloc_device<float>(VECTOR_SIZE, q_ct1);
  d_C = sycl::malloc_device<float>(VECTOR_SIZE, q_ct1);

  sycl::event e =
      q_ct1.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, VECTOR_SIZE),
                                           sycl::range<3>(1, 1, VECTOR_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) {
                           VectorAddKernel(d_A, d_B, d_C, item_ct1);
                         });

  std::vector<void *> ptrs{d_A, d_B, d_C};
  dpct::async_dpct_free(ptrs, {e}, q_ct1);

  q_ct1.wait();

  return 0;
}