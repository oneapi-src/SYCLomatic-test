// ====------ blas_utils_getrfnp.cpp --------------------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//
#define DPCT_USM_LEVEL_NONE
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <oneapi/mkl.hpp>
#include <dpct/blas_utils.hpp>
#include <stdio.h>

// origin matrix A
// |1 3|
// |2 4|

template<class T>
bool verify_data(T* data, T* expect, int num) {
  for(int i = 0; i < num; ++i) {
    if(std::abs(data[i] - expect[i]) > 0.01) {
      return false;
    }
  }
  return true;
}

template<>
bool verify_data(int* data, int* expect, int num) {
  for(int i = 0; i < num; ++i) {
    if(data[i] != expect[i]) {
      return false;
    }
  }
  return true;
}

template<class T>
int test() {
  int n = 2;
  T *A = (T *)malloc(n * n * sizeof(T));
  A[0] = 1;
  A[1] = 2;
  A[2] = 3;
  A[3] = 4;

  sycl::queue *handle;
  handle = &dpct::get_default_queue();
  std::cout << "Device Name: " << handle->get_device().get_info<sycl::info::device::name>() << std::endl;

  T **Aarray;
  T *a0, *a1;
  int *Pivots = nullptr;
  int *dInfo;
  size_t sizeA = n * n * sizeof(T);

  Aarray = (T **)dpct::dpct_malloc(2 * sizeof(T *));
  a1 = (T *)dpct::dpct_malloc(sizeA);
  a0 = (T *)dpct::dpct_malloc(sizeA);
  dInfo = (int *)dpct::dpct_malloc(2 * sizeof(int));

  dpct::dpct_memcpy(a0, A, sizeA, dpct::host_to_device);
  dpct::dpct_memcpy(a1, A, sizeA, dpct::host_to_device);
  dpct::dpct_memcpy(Aarray, &a0, sizeof(T *), dpct::host_to_device);
  dpct::dpct_memcpy(Aarray + 1, &a1, sizeof(T *), dpct::host_to_device);

  dpct::getrf_batch_wrapper(*handle, n, Aarray, n, Pivots, dInfo, 2);
  dpct::get_current_device().queues_wait_and_throw();

  T *new_a_h= (T *)malloc(2 * sizeA);
  dpct::dpct_memcpy(new_a_h, a0, sizeA, dpct::device_to_host);
  dpct::dpct_memcpy(new_a_h + n * n, a1, sizeA, dpct::device_to_host);

  int* info_h = (int*)malloc(2*sizeof(int));
  dpct::dpct_memcpy(info_h, dInfo, 2 * sizeof(int), dpct::device_to_host);

  printf("info0:%d, info1:%d\n", info_h[0], info_h[1]);

  dpct::dpct_free(Aarray);
  dpct::dpct_free(a0);
  dpct::dpct_free(dInfo);
  dpct::dpct_free(a1);

  handle = nullptr;

  printf("newa0[0]:%f, newa0[1]:%f, newa0[2]:%f, newa0[3]:%f\n", new_a_h[0], new_a_h[1], new_a_h[2],
         new_a_h[3]);
  printf("newa1[0]:%f, newa1[1]:%f, newa1[2]:%f, newa1[3]:%f\n", new_a_h[4], new_a_h[5], new_a_h[6],
         new_a_h[7]);

  // check result:
  T expect_a[8] = {
    1, 2, 3, -2,
    1, 2, 3, -2
  };

  bool success = false;
  if(verify_data(new_a_h, expect_a, 8))
    success = true;

  free(new_a_h);
  printf("done.\n");

  return (success ? 0 : 1);
}
int main() {
  bool pass = true;
  if(test<float>()) {
    pass = false;
    printf("float fail\n");
  }
  if(test<double>()) {
    pass = false;
    printf("double fail\n");
  }
  return (pass ? 0 : 1);
}

