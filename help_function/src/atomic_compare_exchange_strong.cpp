// ====------ atomic_compare_exchange_strong.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>
#include <cstdio>
#include <ctime>
#include <dpct/dpct.hpp>
#include <math.h>
#include <stdint.h>

template <typename T>
void atomicKernel(T *atom_arr, T *result_arr)
{
  *(result_arr + 0) = dpct::atomic_compare_exchange_strong(atom_arr, 4, 1);
  *(result_arr + 1) = dpct::atomic_compare_exchange_strong(atom_arr + 1, 24.0f, 2.0f);
  *(result_arr + 2) = dpct::atomic_compare_exchange_strong(atom_arr + 2, 32, 3);
  *(result_arr + 3) = dpct::atomic_compare_exchange_strong(atom_arr + 3, 44.0f, 4.0f);
  *(result_arr + 4) = dpct::atomic_compare_exchange_strong(atom_arr + 4, 52, 5);
  *(result_arr + 5) = dpct::atomic_compare_exchange_strong(atom_arr + 5, 63.0f, 6.0f);
}

int main()
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

  float *floatRes, *deviceResf;
  int *intRes, *deviceResi;
  deviceResf = sycl::malloc_shared<float>(6, q_ct1);
  deviceResi = sycl::malloc_shared<int>(6, q_ct1);
  floatRes = sycl::malloc_shared<float>(6, q_ct1);
  intRes = sycl::malloc_shared<int>(6, q_ct1);
  for (int i = 0; i < 6; ++i)
  {
    *(deviceResf + i) = i;
    *(deviceResi + i) = i;
    *(floatRes + i) = 0;
    *(intRes + i) = 0;
  }
  q_ct1.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
      [=](sycl::nd_item<3> item_ct1)
      {
        atomicKernel(deviceResf, floatRes);
        atomicKernel(deviceResi, intRes);
      });
  q_ct1.wait();
  for (int i = 0; i < 6; ++i)
  {
    if (*(floatRes + i) != i || *(intRes + i) != i)
    {
      sycl::free(deviceResf, q_ct1);
      sycl::free(deviceResi, q_ct1);
      sycl::free(floatRes, q_ct1);
      sycl::free(intRes, q_ct1);
      return 1;
    }
  }
  sycl::free(deviceResf, q_ct1);
  sycl::free(deviceResi, q_ct1);
  sycl::free(floatRes, q_ct1);
  sycl::free(intRes, q_ct1);
  return 0;
}