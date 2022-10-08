// ====------ blas_utils_getrs-usm.cpp---------- -*- C++ -* ----===////
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

// A     *   X         =    B
// |1 3|   | 5  7  9 |   | 23  31  39 |
// |2 4| * | 6  8  10| = | 34  46  58 |

template<class T>
bool verify_data(T* data, T* expect, int num) {
  for(int i = 0; i < num; ++i) {
    if(std::abs(data[i] - expect[i]) > 0.01) {
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
  int nrhs = 3;
  T *A = (T *)malloc(n * n * sizeof(T));
  T *B = (T *)malloc(n * nrhs * sizeof(T));
  A[0] = 2;
  A[1] = 0.5;
  A[2] = 4;
  A[3] = 1;

  B[0] = 23;
  B[1] = 34;
  B[2] = 31;
  B[3] = 46;
  B[4] = 39;
  B[5] = 58;

  int *Pivots_h = (int *)malloc(2 * n * sizeof(int));
  Pivots_h[0] = 2;
  Pivots_h[1] = 2;
  Pivots_h[2] = 2;
  Pivots_h[3] = 2;

  sycl::queue *handle;
  handle = &q_ct1;

  T **Aarray;
  T *a0, *a1;
  int *Pivots;
  int *dInfo;
  size_t szA = n * n * sizeof(T);
  size_t szB = n * nrhs * sizeof(T);

  Aarray = sycl::malloc_device<T *>(2, q_ct1);
  a0 = (T *)sycl::malloc_device(szA, q_ct1);
  a1 = (T *)sycl::malloc_device(szA, q_ct1);
  Pivots = sycl::malloc_device<int>(2 * n, q_ct1);
  dInfo = sycl::malloc_device<int>(2, q_ct1);

  q_ct1.memcpy(Pivots, Pivots_h, 2 * n * sizeof(int)).wait();
  q_ct1.memcpy(a0, A, szA).wait();
  q_ct1.memcpy(a1, A, szA).wait();
  q_ct1.memcpy(Aarray, &a0, sizeof(T *)).wait();
  q_ct1.memcpy(Aarray + 1, &a1, sizeof(T *)).wait();

  int Info;

  T **Barray;
  T *b0, *b1;
  Barray = sycl::malloc_device<T *>(2, q_ct1);
  b0 = (T *)sycl::malloc_device(szB, q_ct1);
  b1 = (T *)sycl::malloc_device(szB, q_ct1);
  q_ct1.memcpy(b0, B, szB).wait();
  q_ct1.memcpy(b1, B, szB).wait();
  q_ct1.memcpy(Barray, &b0, sizeof(T *)).wait();
  q_ct1.memcpy(Barray + 1, &b1, sizeof(T *)).wait();

  dpct::getrs_batch_wrapper(*handle, oneapi::mkl::transpose::nontrans, n, nrhs,
                            (const T**)Aarray, n, Pivots, Barray, n, &Info, 2);
  dev_ct1.queues_wait_and_throw();
  printf("info:%d\n", Info);
  T *res = (T *)malloc(2 * szB);

  q_ct1.memcpy(res, b0, szB).wait();
  q_ct1.memcpy(res + nrhs * n, b1, szB).wait();

  sycl::free(Aarray, q_ct1);
  sycl::free(a0, q_ct1);
  sycl::free(a1, q_ct1);
  sycl::free(Pivots, q_ct1);
  sycl::free(dInfo, q_ct1);

  sycl::free(Barray, q_ct1);
  sycl::free(b0, q_ct1);
  sycl::free(b1, q_ct1);

  handle = nullptr;

  printf("res0[0]:%f, res0[1]:%f, res0[2]:%f, res0[3]:%f, res0[4]:%f, "
         "res0[5]:%f\n",
         res[0], res[1], res[2], res[3], res[4], res[5]);
  printf("res1[0]:%f, res1[1]:%f, res1[2]:%f, res1[3]:%f, res1[4]:%f, "
         "res1[5]:%f\n",
         res[6], res[7], res[8], res[9], res[10], res[11]);

  // check result:
  T expect[12] = {
    5, 6, 7, 8, 9, 10,
    5, 6, 7, 8, 9, 10
  };

  bool success = false;
  if(verify_data(res, expect, 12))
    success = true;

  free(A);
  free(B);
  free(res);
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