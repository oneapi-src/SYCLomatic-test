// ====------ blas_utils_geqrf-usm.cpp---------- -*- C++ -* ----===////
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