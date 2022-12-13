// ====------ test_complex.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <sycl/sycl.hpp>
#include <complex>

int main(){
  sycl::float2* f2_a_ptr = (sycl::float2*)malloc(10 * sizeof(sycl::float2));
  for (int i = 0; i < 10; i++) {
    f2_a_ptr[i].x() = i;
    f2_a_ptr[i].y() = i + 10;
  }

  sycl::float2* f2_b_ptr = (sycl::float2*)malloc(10 * sizeof(sycl::float2));
  memcpy(f2_b_ptr, f2_a_ptr, 10 * sizeof(sycl::float2));

  std::complex<float>* complex_p = (std::complex<float>*)f2_b_ptr;

  printf("original value:\n");
  for (int i = 0; i < 10; i++) {
    printf("%f, %f\n", f2_a_ptr[i].x(), f2_a_ptr[i].y());
  }
  printf("casted value:\n");
  for (int i = 0; i < 10; i++) {
    printf("%f, %f\n", complex_p[i].real(), complex_p[i].imag());
  }

  for (int i = 0; i < 10; i++) {
    if (abs(f2_a_ptr[i].x() - complex_p[i].real()) > 0.01) {
      printf("fail\n");
      return 1;
    }
    if (abs(f2_a_ptr[i].y() - complex_p[i].imag()) > 0.01) {
      printf("fail\n");
      return 1;
    }
  }
  printf("pass\n");
  return 0;
}
