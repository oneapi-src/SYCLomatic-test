// ====------ blas_utils_get_value.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

//test_feature:BlasUtils_get_value
#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <oneapi/mkl.hpp>
#include <dpct/blas_utils.hpp>
#include <stdio.h>

bool test_get_value() {

  float f = 1;
  double d = 2;
  sycl::float2 f2(3, 4);
  sycl::double2 d2(5, 6);

  float* f_p = (float*)dpct::dpct_malloc(sizeof(float));
  double* d_p = (double*)dpct::dpct_malloc(sizeof(double));
  sycl::float2* f2_p = (sycl::float2*)dpct::dpct_malloc(sizeof(sycl::float2));
  sycl::double2* d2_p = (sycl::double2*)dpct::dpct_malloc(sizeof(sycl::double2));

  dpct::dpct_memcpy(f_p, &f, sizeof(float), dpct::host_to_device);
  dpct::dpct_memcpy(d_p, &d, sizeof(double), dpct::host_to_device);
  dpct::dpct_memcpy(f2_p, &f2, sizeof(sycl::float2), dpct::host_to_device);
  dpct::dpct_memcpy(d2_p, &d2, sizeof(sycl::double2), dpct::host_to_device);

  float f_res = 0;
  double d_res = 0;
  std::complex<float> f2_res(0, 0);
  std::complex<double> d2_res(0, 0);

  f_res = dpct::get_value(f_p, dpct::get_default_queue());
  d_res = dpct::get_value(d_p, dpct::get_default_queue());
  f2_res = dpct::get_value(f2_p, dpct::get_default_queue());
  d2_res = dpct::get_value(d2_p, dpct::get_default_queue());

  dpct::dpct_free(f_p);
  dpct::dpct_free(d_p);
  dpct::dpct_free(f2_p);
  dpct::dpct_free(d2_p);

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