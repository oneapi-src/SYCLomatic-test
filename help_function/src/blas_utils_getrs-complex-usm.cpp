// ====------ blas_utils_getrs-complex-usm.cpp---------- -*- C++ -* ----===////
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
  int n = 2;
  int nrhs = 3;
  T *A = (T *)malloc(n * n * sizeof(T));
  T *B = (T *)malloc(n * nrhs * sizeof(T));
  A[0].x() = 2;   A[0].y() = 0;
  A[1].x() = 0.5; A[1].y() = 0;
  A[2].x() = 4;   A[2].y() = 0;
  A[3].x() = 1;   A[3].y() = 0;

  B[0].x() = 23; B[0].y() = 0;
  B[1].x() = 34; B[1].y() = 0;
  B[2].x() = 31; B[2].y() = 0;
  B[3].x() = 46; B[3].y() = 0;
  B[4].x() = 39; B[4].y() = 0;
  B[5].x() = 58; B[5].y() = 0;

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

  printf("res0[0]:(%f,%f), res0[1]:(%f,%f), res0[2]:(%f,%f), res0[3]:(%f,%f), res0[4]:(%f,%f), "
         "res0[5]:(%f,%f)\n",
         res[0].x(), res[0].y(), res[1].x(), res[1].y(), res[2].x(), res[2].y(), res[3].x(), res[3].y(), res[4].x(), res[4].y(), res[5].x(), res[5].y());
  printf("res1[0]:(%f,%f), res1[1]:(%f,%f), res1[2]:(%f,%f), res1[3]:(%f,%f), res1[4]:(%f,%f), "
         "res1[5]:(%f,%f)\n",
         res[6].x(), res[6].y(), res[7].x(), res[7].y(), res[8].x(), res[8].y(), res[9].x(), res[9].y(), res[10].x(), res[10].y(), res[11].x(), res[11].y());

  // check result:
  T expect[12] = {
    T(5, 0), T(6, 0), T(7, 0), T(8, 0), T(9, 0), T(10, 0),
    T(5, 0), T(6, 0), T(7, 0), T(8, 0), T(9, 0), T(10, 0)
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