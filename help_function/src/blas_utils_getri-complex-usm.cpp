// ====------ blas_utils_getri-complex-usm.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <oneapi/mkl.hpp>
#include <dpct/blas_utils.hpp>
#include <stdio.h>

template<class T>
bool verify_data(T* data, T* expect, int num) {
  for(int i = 0; i < num; ++i) {
    if((std::abs(data[i].x() - expect[i].x()) > 0.01) ||
       (std::abs(data[i].y() - expect[i].y()) > 0.01)) {
      return false;
    }
  }
  return true;
}

template<class T>
int test() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::cout << "Device Name: " << dev_ct1.get_info<sycl::info::device::name>() << std::endl;
  int n = 2;
  T *A = (T *)malloc(n * n * sizeof(T));
  A[0].x() = 2;   A[0].y() = 0;
  A[1].x() = 0.5; A[1].y() = 0;
  A[2].x() = 4;   A[2].y() = 0;
  A[3].x() = 1;   A[3].y() = 0;

  int *Pivots_h = (int *)malloc(2 * n * sizeof(int));
  Pivots_h[0] = 2;
  Pivots_h[1] = 2;
  Pivots_h[2] = 2;
  Pivots_h[3] = 2;

  sycl::queue *handle;
  handle = &q_ct1;

  T **Aarray;
  T **Carray;
  T *a0, *a1;
  T *c0, *c1;
  int *Pivots;
  int *dInfo;
  size_t sizeA = n * n * sizeof(T);

  Aarray = sycl::malloc_device<T *>(2, q_ct1);
  Carray = sycl::malloc_device<T *>(2, q_ct1);
  a0 = (T *)sycl::malloc_device(sizeA, q_ct1);
  c0 = (T *)sycl::malloc_device(sizeA, q_ct1);
  a1 = (T *)sycl::malloc_device(sizeA, q_ct1);
  c1 = (T *)sycl::malloc_device(sizeA, q_ct1);
  Pivots = sycl::malloc_device<int>(2 * n, q_ct1);
  dInfo = sycl::malloc_device<int>(2, q_ct1);

  q_ct1.memcpy(Pivots, Pivots_h, 2 * n * sizeof(int)).wait();
  q_ct1.memcpy(a0, A, sizeA).wait();
  q_ct1.memcpy(a1, A, sizeA).wait();
  q_ct1.memcpy(Aarray, &a0, sizeof(T *)).wait();
  q_ct1.memcpy(Carray, &c0, sizeof(T *)).wait();
  q_ct1.memcpy(Aarray + 1, &a1, sizeof(T *)).wait();
  q_ct1.memcpy(Carray + 1, &c1, sizeof(T *)).wait();

  dpct::getri_batch_wrapper(*handle, n, (const T **)Aarray, n, Pivots,
                            Carray, n, dInfo, 2);
  dev_ct1.queues_wait_and_throw();

  T *inv = (T *)malloc(2 * sizeA);

  q_ct1.memcpy(inv, c0, sizeA).wait();
  q_ct1.memcpy(inv + n * n, c1, sizeA).wait();

  sycl::free(Aarray, q_ct1);
  sycl::free(Carray, q_ct1);
  sycl::free(a0, q_ct1);
  sycl::free(c0, q_ct1);
  sycl::free(Pivots, q_ct1);
  sycl::free(dInfo, q_ct1);
  sycl::free(a1, q_ct1);
  sycl::free(c1, q_ct1);

  handle = nullptr;

  printf("inv0[0]:(%f,%f), inv0[1]:(%f,%f), inv0[2]:(%f,%f), inv0[3]:(%f,%f)\n",
         inv[0].x(), inv[0].y(), inv[1].x(), inv[1].y(), inv[2].x(), inv[2].y(), inv[3].x(), inv[3].y());
  printf("inv1[0]:(%f,%f), inv1[1]:(%f,%f), inv1[2]:(%f,%f), inv1[3]:(%f,%f)\n",
         inv[4].x(), inv[4].y(), inv[5].x(), inv[5].y(), inv[6].x(), inv[6].y(), inv[7].x(), inv[7].y());

  // check result:
  T expect[8] = {
    T(-2, 0), T(1, 0), T(1.5, 0), T(-0.5, 0),
    T(-2, 0), T(1, 0), T(1.5, 0), T(-0.5, 0)
  };

  bool success = false;
  if(verify_data(inv, expect, 8))
    success = true;

  free(A);
  free(inv);
  printf("done.\n");

  return (success ? 0 : 1);
}
int main() {
  bool pass = true;
  if(test<sycl::float2>()) {
    pass = false;
    printf("float fail\n");
  }
  if(test<sycl::double2>()) {
    pass = false;
    printf("double fail\n");
  }
  return (pass ? 0 : 1);
}