// ====------ blas_utils_geqrf.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#define DPCT_USM_LEVEL_NONE
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
    if(std::abs(data[i] - expect[i]) > 0.01) {
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
  int Info;
  T **TauArray;
  T *Tau0, *Tau1;
  size_t sizeTau = 2 * sizeof(T);
  size_t sizeA = n * n * sizeof(T);

  Aarray = (T **)dpct::dpct_malloc(2 * sizeof(T *));
  a0 = (T*)dpct::dpct_malloc(sizeA);
  a1 = (T*)dpct::dpct_malloc(sizeA);
  TauArray = (T **)dpct::dpct_malloc(2 * sizeof(T *));
  Tau0 = (T *)dpct::dpct_malloc(sizeTau);
  Tau1 = (T *)dpct::dpct_malloc(sizeTau);

  dpct::dpct_memcpy(a0, A, sizeA, dpct::host_to_device);
  dpct::dpct_memcpy(a1, A, sizeA, dpct::host_to_device);
  dpct::dpct_memcpy(Aarray, &a0, sizeof(T *), dpct::host_to_device);
  dpct::dpct_memcpy(Aarray + 1, &a1, sizeof(T *), dpct::host_to_device);
  dpct::dpct_memcpy(TauArray, &Tau0, sizeof(T *), dpct::host_to_device);
  dpct::dpct_memcpy(TauArray + 1, &Tau1, sizeof(T *), dpct::host_to_device);

  int a =
      (dpct::geqrf_batch_wrapper(*handle, n, n, Aarray, n, TauArray, &Info, 2),
       0);
  dpct::get_current_device().queues_wait_and_throw();

  T *res = (T *)malloc(2 * sizeA + 2 * sizeTau);

  dpct::dpct_memcpy(res, a0, sizeA, dpct::device_to_host);
  dpct::dpct_memcpy(res + n * n, a1, sizeA, dpct::device_to_host);
  dpct::dpct_memcpy(res + 2 * n * n, Tau0, sizeTau, dpct::device_to_host);
  dpct::dpct_memcpy(res + 2 * n * n + 2, Tau1, sizeTau, dpct::device_to_host);

  dpct::dpct_free(Aarray);
  dpct::dpct_free(a0);
  dpct::dpct_free(a1);
  dpct::dpct_free(TauArray);
  dpct::dpct_free(Tau0);
  dpct::dpct_free(Tau1);

  handle = nullptr;

  printf("a0[0]:%f, a0[1]:%f, a0[2]:%f, a0[3]:%f\n", res[0], res[1], res[2],
         res[3]);
  printf("a1[0]:%f, a1[1]:%f, a1[2]:%f, a1[3]:%f\n", res[4], res[5], res[6],
         res[7]);
  printf("tau0[0]:%f, tau0[1]:%f\n", res[8], res[9]);
  printf("tau1[0]:%f, tau1[1]:%f\n", res[10], res[11]);

  // check result:
  T expect_a[8] = {
    -2.236068, 0.618034, -4.919349, -0.894427,
    -2.236068, 0.618034, -4.919349, -0.894427
  };
  T expect_tau[4] = {
    1.447214, 0,
    1.447214, 0
  };

  bool success = false;
  if(verify_data(res, expect_a, 8) && verify_data(res + 8, expect_tau, 4))
    success = true;

  free(A);
  free(res);

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