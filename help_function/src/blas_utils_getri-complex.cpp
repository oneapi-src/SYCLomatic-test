// ====------ blas_utils_getri-complex.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#define DPCT_USM_LEVEL_NONE
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
  handle = &dpct::get_default_queue();

  T **Aarray;
  T **Carray;
  T *a0, *a1;
  T *c0, *c1;
  int *Pivots;
  int *dInfo;
  size_t sizeA = n * n * sizeof(T);

  Aarray = (T **)dpct::dpct_malloc(2 * sizeof(T *));
  Carray = (T **)dpct::dpct_malloc(2 * sizeof(T *));
  a0 = (T *)dpct::dpct_malloc(sizeA);
  c0 = (T *)dpct::dpct_malloc(sizeA);
  a1 = (T *)dpct::dpct_malloc(sizeA);
  c1 = (T *)dpct::dpct_malloc(sizeA);
  Pivots = (int *)dpct::dpct_malloc(2 * n * sizeof(int));
  dInfo = (int *)dpct::dpct_malloc(2 * sizeof(int));

  dpct::dpct_memcpy(Pivots, Pivots_h, 2 * n * sizeof(int),
                    dpct::host_to_device);
  dpct::dpct_memcpy(a0, A, sizeA, dpct::host_to_device);
  dpct::dpct_memcpy(a1, A, sizeA, dpct::host_to_device);
  dpct::dpct_memcpy(Aarray, &a0, sizeof(T *), dpct::host_to_device);
  dpct::dpct_memcpy(Carray, &c0, sizeof(T *), dpct::host_to_device);
  dpct::dpct_memcpy(Aarray + 1, &a1, sizeof(T *), dpct::host_to_device);
  dpct::dpct_memcpy(Carray + 1, &c1, sizeof(T *), dpct::host_to_device);

  dpct::getri_batch_wrapper(*handle, n, (const T **)Aarray, n, Pivots,
                            Carray, n, dInfo, 2);
  dpct::get_current_device().queues_wait_and_throw();

  T *inv = (T *)malloc(2 * sizeA);

  dpct::dpct_memcpy(inv, c0, sizeA, dpct::device_to_host);
  dpct::dpct_memcpy(inv + n * n, c1, sizeA, dpct::device_to_host);

  dpct::dpct_free(Aarray);
  dpct::dpct_free(Carray);
  dpct::dpct_free(a0);
  dpct::dpct_free(c0);
  dpct::dpct_free(Pivots);
  dpct::dpct_free(dInfo);
  dpct::dpct_free(a1);
  dpct::dpct_free(c1);

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