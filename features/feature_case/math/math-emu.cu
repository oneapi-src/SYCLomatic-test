// ====------ math-emu.cu---------- *- CUDA -* ----===////
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

__global__ void math_emu() {
  float f;
  double d = 0;
  double &d0 = d, &d1 = d, &d2 = d, &d3 = d;
  int i, i2;
  unsigned u, u1, u2;
  float f0, f1, f2, f3;
  long l, l2;
  long long ll, ll2;
  unsigned long long ull, ull2;

  // CHECK: d2 = sycl::frexp(d0, sycl::make_ptr<int, sycl::access::address_space::global_space>(&i));
  d2 = frexp(d0, &i);
  // CHECK: d2 = sycl::frexp((double)i, sycl::make_ptr<int, sycl::access::address_space::global_space>(&i));
  d2 = frexp(i, &i);

  // CHECK: d2 = sycl::modf(d0, sycl::make_ptr<double, sycl::access::address_space::global_space>(&d1));
  d2 = modf(d0, &d1);
  // CHECK: d2 = sycl::modf((double)i, sycl::make_ptr<double, sycl::access::address_space::global_space>(&d1));
  d2 = modf(i, &d1);

  // CHECK: d2 = sycl::remquo(d0, d1, sycl::make_ptr<int, sycl::access::address_space::global_space>(&i));
  d2 = remquo(d0, d1, &i);
  // CHECK: d2 = sycl::remquo((double)i, (double)i, sycl::make_ptr<int, sycl::access::address_space::global_space>(&i));
  d2 = remquo(i, i, &i);
  // CHECK: d2 = sycl::remquo(d0, (double)i, sycl::make_ptr<int, sycl::access::address_space::global_space>(&i));
  d2 = remquo(d0, i, &i);
  // CHECK: d2 = sycl::remquo((double)i, d1, sycl::make_ptr<int, sycl::access::address_space::global_space>(&i));
  d2 = remquo(i, d1, &i);


  // CHECK: d2 = sycl::nan(0u);
  d2 = nan("NaN");

  // CHECK: d0 = sycl::nextafter(d0, d0);
  d0 = nextafter(d0, d0);
  // CHECK: d0 = sycl::nextafter((double)i, (double)i);
  d0 = nextafter(i, i);
  // CHECK: d0 = sycl::nextafter(d0, (double)i);
  d0 = nextafter(d0, i);
  // CHECK: d0 = sycl::nextafter((double)i, d1);
  d0 = nextafter(i, d1);


  // CHECK: /*
  // CHECK-NEXT: DPCT1017:{{[0-9]+}}: The sycl::floor call is used instead of the nearbyintf call. These two calls do not provide exactly the same functionality. Check the potential precision and/or performance issues for the generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: f = sycl::floor(f + 0.5);
  f = nearbyintf(f);

  // CHECK: /*
  // CHECK-NEXT: DPCT1017:{{[0-9]+}}: The sycl::floor call is used instead of the nearbyint call. These two calls do not provide exactly the same functionality. Check the potential precision and/or performance issues for the generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: d = sycl::floor(d + 0.5);
  d = nearbyint(d);

  // CHECK: /*
  // CHECK-NEXT: DPCT1017:{{[0-9]+}}: The sycl::hypot call is used instead of the rhypotf call. These two calls do not provide exactly the same functionality. Check the potential precision and/or performance issues for the generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: f = 1 / sycl::hypot(f, f);
  f = rhypotf(f, f);

  // CHECK: /*
  // CHECK-NEXT: DPCT1017:{{[0-9]+}}: The sycl::sincos call is used instead of the sincospif call. These two calls do not provide exactly the same functionality. Check the potential precision and/or performance issues for the generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: f = sycl::sincos(f * DPCT_PI_F, sycl::make_ptr<float, sycl::access::address_space::global_space>(&f));
  sincospif(f, &f, &f);

  // CHECK: /*
  // CHECK-NEXT: DPCT1017:{{[0-9]+}}: The sycl::sincos call is used instead of the sincospi call. These two calls do not provide exactly the same functionality. Check the potential precision and/or performance issues for the generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: d = sycl::sincos(d * DPCT_PI, sycl::make_ptr<double, sycl::access::address_space::global_space>(&d));
  sincospi(d, &d, &d);


  // CHECK: sycl::exp(d0*d0)*sycl::erfc(d0);
  erfcx(d0);
  // CHECK: sycl::exp(f0*f0)*sycl::erfc(f0);
  erfcxf(f0);
  // CHECK: sycl::fast_length(sycl::float3(d0, d1, d2));
  norm3d(d0, d1, d2);
  // CHECK: sycl::fast_length(sycl::float3(f0, f1, f2));
  norm3df(f0, f1, f2);
  // CHECK: sycl::fast_length(sycl::float4(d0, d1, d2, d3));
  norm4d(d0, d1, d2, d3);
  // CHECK: sycl::fast_length(sycl::float4(f0, f1, f2, f3));
  norm4df(f0, f1, f2, f3);
  // CHECK: sycl::native::recip((float)sycl::cbrt(d0));
  rcbrt(d0);
  // CHECK: sycl::native::recip((float)sycl::cbrt(f0));
  rcbrtf(f0);
  // CHECK: sycl::native::recip(sycl::fast_length(sycl::float3(d0, d1, d2)));
  rnorm3d(d0, d1, d2);
  // CHECK: sycl::native::recip(sycl::fast_length(sycl::float3(f0, f1, f2)));
  rnorm3df(f0, f1, f2);
  // CHECK: sycl::native::recip(sycl::fast_length(sycl::float4(d0, d1, d2, d3)));
  rnorm4d(d0, d1, d2, d3);
  // CHECK: sycl::native::recip(sycl::fast_length(sycl::float4(f0, f1, f2, f3)));
  rnorm4df(f0, f1, f2, f3);
  // CHECK: d0*(2<<l);
  scalbln(d0, l);
  // CHECK: f0*(2<<l);
  scalblnf(f0, l);
  // CHECK: d0*(2<<i);
  scalbn(d0, i);
  // CHECK: f0*(2<<i);
  scalbnf(f0, i);
  // CHECK: dpct::cast_double_to_int(d0);
  __double2hiint(d0);
  // CHECK: dpct::cast_double_to_int(d0, false);
  __double2loint(d0);
  // CHECK: dpct::cast_ints_to_double(i, i2);
  __hiloint2double(i, i2);

  // CHECK: sycl::abs_diff(i, i2)+u;
  __sad(i, i2, u);
  // CHECK: sycl::abs_diff(u, u1)+u2;
  __usad(u, u1, u2);

  // CHECK: /*
  // CHECK-NEXT: DPCT1017:{{[0-9]+}}: The sycl::native::recip call is used instead of the __drcp_rd call. These two calls do not provide exactly the same functionality. Check the potential precision and/or performance issues for the generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::native::recip((float)d0);
  __drcp_rd(d0);
  // CHECK: /*
  // CHECK-NEXT: DPCT1017:{{[0-9]+}}: The sycl::native::recip call is used instead of the __drcp_rn call. These two calls do not provide exactly the same functionality. Check the potential precision and/or performance issues for the generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::native::recip((float)d0);
  __drcp_rn(d0);
  // CHECK: /*
  // CHECK-NEXT: DPCT1017:{{[0-9]+}}: The sycl::native::recip call is used instead of the __drcp_ru call. These two calls do not provide exactly the same functionality. Check the potential precision and/or performance issues for the generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::native::recip((float)d0);
  __drcp_ru(d0);
  // CHECK: /*
  // CHECK-NEXT: DPCT1017:{{[0-9]+}}: The sycl::native::recip call is used instead of the __drcp_rz call. These two calls do not provide exactly the same functionality. Check the potential precision and/or performance issues for the generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::native::recip((float)d0);
  __drcp_rz(d0);

  // CHECK: sycl::mul_hi(ll, ll);
  __mul64hi(ll, ll);
  // CHECK: sycl::rhadd(i, i2);
  __rhadd(i, i2);
  // CHECK: sycl::hadd(u, u2);
  __uhadd(u, u2);
  // CHECK: sycl::mul24(u, u2);
  __umul24(u, u2);
  // CHECK: sycl::mul_hi(ull, ull2);
  __umul64hi(ull, ull2);
  // CHECK: sycl::mul_hi(u, u2);
  __umulhi(u, u2);
  // CHECK: sycl::rhadd(u, u2);
  __urhadd(u, u2);

  // CHECK: u = dpct::bytewise_max_signed(u, u2);
  u = __vmaxs4(u, u2);

  double *a_d;
  // CHECK: 0;
  norm(0, a_d);
  // CHECK: sycl::fast_length((float)a_d[0]);
  norm(1, a_d);
  // CHECK: sycl::fast_length(sycl::float2(a_d[0], a_d[1]));
  norm(2, a_d);
  // CHECK: sycl::fast_length(sycl::float3(a_d[0], a_d[1], a_d[2]));
  norm(3, a_d);
  // CHECK: sycl::fast_length(sycl::float4(a_d[0], a_d[1], a_d[2], a_d[3]));
  norm(4, a_d);
  // CHECK: dpct::fast_length((float *)a_d, 5);
  norm(5, a_d);
}

int main() {
  math_emu<<<1, 1>>>();
}
