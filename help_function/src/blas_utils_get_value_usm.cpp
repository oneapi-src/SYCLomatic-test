// ====------ blas_utils_get_value_usm.cpp---------- -*- C++ -* ----===////
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

bool test_get_value() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

  float f = 1;
  double d = 2;
  sycl::float2 f2(3, 4);
  sycl::double2 d2(5, 6);

  float* f_p = sycl::malloc_device<float>(1, q_ct1);
  double* d_p = sycl::malloc_device<double>(1, q_ct1);
  sycl::float2* f2_p = sycl::malloc_device<sycl::float2>(1, q_ct1);
  sycl::double2* d2_p = sycl::malloc_device<sycl::double2>(1, q_ct1);

  q_ct1.memcpy(f_p, &f, sizeof(float));
  q_ct1.memcpy(d_p, &d, sizeof(double));
  q_ct1.memcpy(f2_p, &f2, sizeof(sycl::float2));
  q_ct1.memcpy(d2_p, &d2, sizeof(sycl::double2)).wait();

  float f_res = 0;
  double d_res = 0;
  std::complex<float> f2_res(0, 0);
  std::complex<double> d2_res(0, 0);

  f_res = dpct::get_value(f_p, q_ct1);
  d_res = dpct::get_value(d_p, q_ct1);
  f2_res = dpct::get_value(f2_p, q_ct1);
  d2_res = dpct::get_value(d2_p, q_ct1);

  sycl::free(f_p, q_ct1);
  sycl::free(d_p, q_ct1);
  sycl::free(f2_p, q_ct1);
  sycl::free(d2_p, q_ct1);

  if (std::abs(f_res-1) > 0.01)
    return false;
  if (std::abs(d_res-2) > 0.01)
    return false;
  if (std::abs(f2_res.real()-3) > 0.01 || std::abs(f2_res.imag()-4) > 0.01)
    return false;
  if (std::abs(d2_res.real()-5) > 0.01 || std::abs(d2_res.imag()-6) > 0.01)
    return false;

  return true;
}


int main() {
  bool pass = true;
  if(!test_get_value()) {
    pass = false;
    printf("get_value fail\n");
  }
  return (pass ? 0 : 1);
}