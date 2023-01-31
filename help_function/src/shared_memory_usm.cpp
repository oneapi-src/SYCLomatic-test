// ====------ shared_memory_usm.cpp---------- -*- C++ -* ----===////
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
#include <string.h>

#define M 4
#define N 8

dpct::shared_memory<float, 1> array(N);
dpct::shared_memory<float, 1> result(M*N);

void my_kernel(float* array, float* result,
               sycl::nd_item<3> item_ct1,
               float *resultInGroup)
{


  array[item_ct1.get_local_id(2)] = item_ct1.get_local_id(2);
  resultInGroup[item_ct1.get_local_id(2)] = item_ct1.get_group(2);

  item_ct1.barrier();

  if (item_ct1.get_local_id(2) == 0) {
    memcpy(&result[item_ct1.get_group(2)*N], resultInGroup, sizeof(float)*N);
  }
}


int main () {
  {
    auto array_ct0 = array.get_ptr();
    auto result_ct1 = result.get_ptr();
    dpct::get_default_queue().submit(
      [&](sycl::handler &cgh) {
        sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> resultInGroup_acc_ct1(sycl::range<1>(8), cgh);

        cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, M) * sycl::range<3>(1, 1, N), sycl::range<3>(1, 1, N)), 
          [=](sycl::nd_item<3> item_ct1) {
            my_kernel(array_ct0, result_ct1, item_ct1, resultInGroup_acc_ct1.get_pointer());
          });
      });
  }

  dpct::get_current_device().queues_wait_and_throw();
  for(int j = 0; j < M; j++) {
    for (int i = 0; i < N; i++) {
      printf("%f ", result[j*N + i]);
    }
    printf("\n");
  }

  return 0;
}

