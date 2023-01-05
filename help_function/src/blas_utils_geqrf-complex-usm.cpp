// ====------ blas_utils_geqrf-complex-usm.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <oneapi/mkl.hpp>
#include <dpct/blas_utils.hpp>
#include <stdio.h>

// origin matrix
// |1 3|
// |2 4|

// Q                           R
// |1/sqrt(5)   2/sqrt(5)|    |sqrt(5)  11/sqrt(5)|
// |2/sqrt(5)  -1/sqrt(5)|    |0         2/sqrt(5)|
//
//=                          =
// |0.4472136   0.8944272|    |2.236068  4.9193496|
// |0.8944272  -0.4472136|    |0         0.8944272|
//
//   a[0]:-2.236068,   a[1]:0.618034, a[2]:-4.919349, a[3]:-0.894427
// tau[0]: 1.447214, tau[1]:0.000000

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
  A[0].x() = 1; A[0].y() = 0;
  A[1].x() = 2; A[1].y() = 0;
  A[2].x() = 3; A[2].y() = 0;
  A[3].x() = 4; A[3].y() = 0;

  sycl::queue *handle;
  handle = &q_ct1;

  T **Aarray;
  T *a0, *a1;
  int Info;
  T **TauArray;
  T *Tau0, *Tau1;
  size_t sizeTau = 2 * sizeof(T);
  size_t sizeA = n * n * sizeof(T);

  Aarray = sycl::malloc_device<T *>(2, q_ct1);
  a0 = (T *)sycl::malloc_device(sizeA, q_ct1);
  a1 = (T *)sycl::malloc_device(sizeA, q_ct1);
  TauArray = sycl::malloc_device<T *>(2, q_ct1);
  Tau0 = (T *)sycl::malloc_device(sizeTau, q_ct1);
  Tau1 = (T *)sycl::malloc_device(sizeTau, q_ct1);

  q_ct1.memcpy(a0, A, sizeA).wait();
  q_ct1.memcpy(a1, A, sizeA).wait();
  q_ct1.memcpy(Aarray, &a0, sizeof(T *)).wait();
  q_ct1.memcpy(Aarray + 1, &a1, sizeof(T *)).wait();
  q_ct1.memcpy(TauArray, &Tau0, sizeof(T *)).wait();
  q_ct1.memcpy(TauArray + 1, &Tau1, sizeof(T *)).wait();

  int a =
      (dpct::geqrf_batch_wrapper(*handle, n, n, Aarray, n, TauArray, &Info, 2),
       0);
  dev_ct1.queues_wait_and_throw();

  T *res = (T *)malloc(2 * sizeA + 2 * sizeTau);

  q_ct1.memcpy(res, a0, sizeA).wait();
  q_ct1.memcpy(res + n * n, a1, sizeA).wait();
  q_ct1.memcpy(res + 2 * n * n, Tau0, sizeTau).wait();
  q_ct1.memcpy(res + 2 * n * n + 2, Tau1, sizeTau).wait();

  sycl::free(Aarray, q_ct1);
  sycl::free(a0, q_ct1);
  sycl::free(a1, q_ct1);
  sycl::free(TauArray, q_ct1);
  sycl::free(Tau0, q_ct1);
  sycl::free(Tau1, q_ct1);

  handle = nullptr;

  printf("a0[0]:(%f,%f), a0[1]:(%f,%f), a0[2]:(%f,%f), a0[3]:(%f,%f)\n",
         res[0].x(), res[0].y(), res[1].x(), res[1].y(), res[2].x(), res[2].y(), res[3].x(), res[3].y());
  printf("a1[0]:(%f,%f), a1[1]:(%f,%f), a1[2]:(%f,%f), a1[3]:(%f,%f)\n",
         res[4].x(), res[4].y(), res[5].x(), res[5].y(), res[6].x(), res[6].y(), res[7].x(), res[7].y());
  printf("tau0[0]:(%f,%f), tau0[1]:(%f,%f)\n", res[8].x(), res[8].y(), res[9].x(), res[9].y());
  printf("tau1[0]:(%f,%f), tau1[1]:(%f,%f)\n", res[10].x(), res[10].y(), res[11].x(), res[11].y());

  // check result:
  T expect_a[8] = {
    T(-2.236068, 0), T(0.618034, 0), T(-4.919349, 0), T(-0.894427, 0),
    T(-2.236068, 0), T(0.618034, 0), T(-4.919349, 0), T(-0.894427, 0)
  };
  T expect_tau[4] = {
    T(1.447214, 0), T(0, 0),
    T(1.447214, 0),T(0, 0)
  };

  bool success = false;
  if(verify_data(res, expect_a, 8) && verify_data(res + 8, expect_tau, 4))
    success = true;

  free(A);
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