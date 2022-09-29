// ====------ blas_utils_getrf-usm.cpp---------- -*- C++ -* ----===////
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
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::cout << "Device Name: " << dev_ct1.get_info<sycl::info::device::name>() << std::endl;
  int n = 2;
  T *A = (T *)malloc(n * n * sizeof(T));
  A[0] = 1;
  A[1] = 2;
  A[2] = 3;
  A[3] = 4;

  sycl::queue *handle;
  handle = &q_ct1;

  T **Aarray;
  T *a0, *a1;
  int *Pivots;
  int *dInfo;
  size_t sizeA = n * n * sizeof(T);

  Aarray = sycl::malloc_device<T *>(2, q_ct1);
  a1 = (T *)sycl::malloc_device(sizeA, q_ct1);
  a0 = (T *)sycl::malloc_device(sizeA, q_ct1);
  Pivots = sycl::malloc_device<int>(2 * n, q_ct1);
  dInfo = sycl::malloc_device<int>(2, q_ct1);

  q_ct1.memcpy(a0, A, sizeA).wait();
  q_ct1.memcpy(a1, A, sizeA).wait();
  q_ct1.memcpy(Aarray, &a0, sizeof(T *)).wait();
  q_ct1.memcpy(Aarray + 1, &a1, sizeof(T *)).wait();

  /*
  DPCT1047:0: The meaning of Pivots in the dpct::getrf_batch_wrapper is
  different from the cublasSgetrfBatched. You may need to check the migrated
  code.
  */
  dpct::getrf_batch_wrapper(*handle, n, Aarray, n, Pivots, dInfo, 2);
  dev_ct1.queues_wait_and_throw();

  int *pivot_h= (int *)malloc(2 * 2 * sizeof(int));
  T *new_a_h= (T *)malloc(2 * sizeA);
  q_ct1.memcpy(new_a_h, a0, sizeA).wait();
  q_ct1.memcpy(new_a_h + n * n, a1, sizeA).wait();

  q_ct1.memcpy(pivot_h, Pivots, 2 * 2 * sizeof(int)).wait();

  int* info_h = (int*)malloc(2*sizeof(int));
  q_ct1.memcpy(info_h, dInfo, 2 * sizeof(int)).wait();

  printf("info0:%d, info1:%d\n", info_h[0], info_h[1]);

  sycl::free(Aarray, q_ct1);
  sycl::free(a0, q_ct1);
  sycl::free(Pivots, q_ct1);
  sycl::free(dInfo, q_ct1);
  sycl::free(a1, q_ct1);

  handle = nullptr;

  printf("newa0[0]:%f, newa0[1]:%f, newa0[2]:%f, newa0[3]:%f\n", new_a_h[0], new_a_h[1], new_a_h[2],
         new_a_h[3]);
  printf("newa1[0]:%f, newa1[1]:%f, newa1[2]:%f, newa1[3]:%f\n", new_a_h[4], new_a_h[5], new_a_h[6],
         new_a_h[7]);

  printf("pivot_h[0]:%d, pivot_h[1]:%d, pivot_h[2]:%d, pivot_h[3]:%d\n", pivot_h[0], pivot_h[1], pivot_h[2],
         pivot_h[3]);

  // check result:
  T expect_a[8] = {
    2, 0.5, 4, 1,
    2, 0.5, 4, 1
  };
  int expect_pivot[4] = {
    2, 2,
    2, 2
  };

  bool success = false;
  if(verify_data(new_a_h, expect_a, 8) && verify_data(pivot_h, expect_pivot, 4))
    success = true;

  free(new_a_h);
  free(pivot_h);
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