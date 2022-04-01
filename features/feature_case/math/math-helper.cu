// ====------ math-helper.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <algorithm>

#include <stdio.h>

#include "cuda_fp16.h"

__global__ void math_helper() {
  short s;
  unsigned short us;
  int i;
  unsigned int ui;
  long long ll;
  unsigned long long ull;
  __half h;
  float f;
  double d;

  // CHECK: s = dpct::bit_cast<sycl::half, short>(h);
  s = __half_as_short(h);

  // CHECK: us = dpct::bit_cast<sycl::half, unsigned short>(h);
  us = __half_as_ushort(h);


  // CHECK: h = dpct::bit_cast<short, sycl::half>(s);
  h = __short_as_half(s);

  // CHECK: h = dpct::bit_cast<unsigned short, sycl::half>(us);
  h = __ushort_as_half(us);

  // CHECK: ll = dpct::bit_cast<double, long long>(d);
  ll = __double_as_longlong(d);

  // CHECK: i = dpct::bit_cast<float, int>(f);
  i = __float_as_int(f);

  // CHECK: ui = dpct::bit_cast<float, unsigned int>(f);
  ui = __float_as_uint(f);

  // CHECK: f = dpct::bit_cast<int, float>(i);
  f = __int_as_float(i);

  // CHECK: d = dpct::bit_cast<long long, double>(ll);
  d = __longlong_as_double(ll);

  // CHECK: f = dpct::bit_cast<unsigned int, float>(ui);
  f = __uint_as_float(ui);
  // CHECK: dpct::cast_double_to_int(d0);
  __double2hiint(d);
  // CHECK: dpct::cast_double_to_int(d0, false);
  __double2loint(d);
  // CHECK: dpct::cast_ints_to_double(i, i2);
  __hiloint2double(i, i);


  // CHECK: u = dpct::bytewise_max_signed(u, u2);
  ui = __vmaxs4(ui, ui);

  ull = __brevll(ull);

  double *a_d;
  // CHECK: dpct::fast_length((float *)a_d, 5);
  norm(5, a_d);
}


int main() {
  math_helper<<<1, 1>>>();
}
