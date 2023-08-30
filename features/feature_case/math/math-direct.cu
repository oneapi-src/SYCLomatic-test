// ====------ math-direct.cu---------- *- CUDA -* ----===////
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

__global__ void math_direct() {
  double d = 0;
  double &d0 = d, &d1 = d, &d2 = d;
  int i;

  // Double Precision Mathematical Functions

  // CHECK: d2 = sycl::acos(d0);
  d2 = acos(d0);
  // CHECK: d2 = sycl::acos((double)i);
  d2 = acos(i);

  // CHECK: d2 = sycl::acosh(d0);
  d2 = acosh(d0);
  // CHECK: d2 = sycl::acosh((double)i);
  d2 = acosh(i);

  // CHECK: d2 = sycl::asin(d0);
  d2 = asin(d0);
  // CHECK: d2 = sycl::asin((double)i);
  d2 = asin(i);

  // CHECK: d2 = sycl::asinh(d0);
  d2 = asinh(d0);
  // CHECK: d2 = sycl::asinh((double)i);
  d2 = asinh(i);

  // CHECK: d2 = sycl::atan2(d0, d1);
  d2 = atan2(d0, d1);
  // CHECK: d2 = sycl::atan2((double)i, (double)i);
  d2 = atan2(i, i);
  // CHECK: d2 = sycl::atan2(d0, (double)i);
  d2 = atan2(d0, i);
  // CHECK: d2 = sycl::atan2((double)i, d1);
  d2 = atan2(i, d1);

  // CHECK: d2 = sycl::atan(d0);
  d2 = atan(d0);
  // CHECK: d2 = sycl::atan((double)i);
  d2 = atan(i);

  // CHECK: d2 = sycl::atanh(d0);
  d2 = atanh(d0);
  // CHECK: d2 = sycl::atanh((double)i);
  d2 = atanh(i);

  // CHECK: d2 = sycl::cbrt(d0);
  d2 = cbrt(d0);
  // CHECK: d2 = sycl::cbrt((double)i);
  d2 = cbrt(i);

  // CHECK: d2 = sycl::ceil(d0);
  d2 = ceil(d0);

  // CHECK: d2 = sycl::copysign(d0, d1);
  d2 = copysign(d0, d1);
  // CHECK: d2 = sycl::copysign((double)i, (double)i);
  d2 = copysign(i, i);
  // CHECK: d2 = sycl::copysign(d0, (double)i);
  d2 = copysign(d0, i);
  // CHECK: d2 = sycl::copysign((double)i, d1);
  d2 = copysign(i, d1);

  // CHECK: d2 = sycl::cos(d0);
  d2 = cos(d0);
  // CHECK: d2 = sycl::cos((double)i);
  d2 = cos(i);

  // CHECK: d2 = sycl::cosh(d0);
  d2 = cosh(d0);
  // CHECK: d2 = sycl::cosh((double)i);
  d2 = cosh(i);

  // CHECK: d2 = sycl::cospi(d0);
  d2 = cospi(d0);
  // CHECK: d2 = sycl::cospi((double)i);
  d2 = cospi((double)i);

  // CHECK: d2 = sycl::erfc(d0);
  d2 = erfc(d0);
  // CHECK: d2 = sycl::erfc((double)i);
  d2 = erfc(i);

  // CHECK: d2 = sycl::erf(d0);
  d2 = erf(d0);
  // CHECK: d2 = sycl::erf((double)i);
  d2 = erf(i);

  // CHECK: d2 = sycl::exp10(d0);
  d2 = exp10(d0);
  // CHECK: d2 = sycl::exp10((double)i);
  d2 = exp10((double)i);

  // CHECK: d2 = sycl::exp2(d0);
  d2 = exp2(d0);
  // CHECK: d2 = sycl::exp2((double)i);
  d2 = exp2(i);

  // CHECK: d2 = sycl::exp(d0);
  d2 = exp(d0);
  // CHECK: d2 = sycl::exp((double)i);
  d2 = exp(i);

  // CHECK: d2 = sycl::expm1(d0);
  d2 = expm1(d0);
  // CHECK: d2 = sycl::expm1((double)i);
  d2 = expm1(i);

  // CHECK: d2 = sycl::cos(d0);
  d2 = cos(d0);
  // CHECK: d2 = sycl::cos((double)i);
  d2 = cos(i);

  // CHECK: d2 = sycl::cosh(d0);
  d2 = cosh(d0);
  // CHECK: d2 = sycl::cosh((double)i);
  d2 = cosh(i);

  // CHECK: d2 = sycl::cospi(d0);
  d2 = cospi(d0);
  // CHECK: d2 = sycl::cospi((double)i);
  d2 = cospi((double)i);

  // CHECK: d2 = sycl::erfc(d0);
  d2 = erfc(d0);
  // CHECK: d2 = sycl::erfc((double)i);
  d2 = erfc(i);

  // CHECK: d2 = sycl::erf(d0);
  d2 = erf(d0);
  // CHECK: d2 = sycl::erf((double)i);
  d2 = erf(i);

  // CHECK: d2 = sycl::exp10(d0);
  d2 = exp10(d0);
  // CHECK: d2 = sycl::exp10((double)i);
  d2 = exp10((double)i);

  // CHECK: d2 = sycl::exp2(d0);
  d2 = exp2(d0);
  // CHECK: d2 = sycl::exp2((double)i);
  d2 = exp2(i);

  // CHECK: d2 = sycl::exp(d0);
  d2 = exp(d0);
  // CHECK: d2 = sycl::exp((double)i);
  d2 = exp(i);

  // CHECK: d2 = sycl::expm1(d0);
  d2 = expm1(d0);
  // CHECK: d2 = sycl::expm1((double)i);
  d2 = expm1(i);

  // CHECK: d2 = sycl::fabs(d0);
  d2 = fabs(d0);
  // CHECK: d2 = sycl::fabs((double)i);
  d2 = fabs(i);

  // CHECK: sycl::fabs(d0);
  abs(d0);
  // CHECK: sycl::fabs(d0 * d1);
  abs(d0 * d1);

  // CHECK: d2 = sycl::fdim(d0, d1);
  d2 = fdim(d0, d1);
  // CHECK: d2 = sycl::fdim((double)i, (double)i);
  d2 = fdim(i, i);
  // CHECK: d2 = sycl::fdim(d0, (double)i);
  d2 = fdim(d0, i);
  // CHECK: d2 = sycl::fdim((double)i, d1);
  d2 = fdim(i, d1);

  // CHECK: d2 = sycl::floor(d0);
  d2 = floor(d0);
  // CHECK: d2 = sycl::floor((double)i);
  d2 = floor(i);

  // CHECK: d2 = sycl::fma(d0, d1, d2);
  d2 = fma(d0, d1, d2);
  // CHECK: d2 = sycl::fma((double)i, (double)i, (double)i);
  d2 = fma(i, i, i);
  // CHECK: d2 = sycl::fma(d0, (double)i, (double)i);
  d2 = fma(d0, i, i);
  // CHECK: d2 = sycl::fma((double)i, d1, (double)i);
  d2 = fma(i, d1, i);
  // CHECK: d2 = sycl::fma((double)i, (double)i, d2);
  d2 = fma(i, i, d2);
  // CHECK: d2 = sycl::fma(d0, d1, (double)i);
  d2 = fma(d0, d1, i);
  // CHECK: d2 = sycl::fma(d0, (double)i, d2);
  d2 = fma(d0, i, d2);
  // CHECK: d2 = sycl::fma((double)i, d1, d2);
  d2 = fma(i, d1, d2);

  // CHECK: d2 = sycl::fmax(d0, d1);
  d2 = fmax(d0, d1);
  // CHECK: d2 = sycl::fmax((double)i, (double)i);
  d2 = fmax(i, i);
  // CHECK: d2 = sycl::fmax(d0, (double)i);
  d2 = fmax(d0, i);
  // CHECK: d2 = sycl::fmax((double)i, d1);
  d2 = fmax(i, d1);

  // CHECK: d2 = sycl::fmin(d0, d1);
  d2 = fmin(d0, d1);
  // CHECK: d2 = sycl::fmin((double)i, (double)i);
  d2 = fmin(i, i);
  // CHECK: d2 = sycl::fmin(d0, (double)i);
  d2 = fmin(d0, i);
  // CHECK: d2 = sycl::fmin((double)i, d1);
  d2 = fmin(i, d1);

  // CHECK: d2 = sycl::fmod(d0, d1);
  d2 = fmod(d0, d1);
  // CHECK: d2 = sycl::fmod((double)i, (double)i);
  d2 = fmod(i, i);
  // CHECK: d2 = sycl::fmod(d0, (double)i);
  d2 = fmod(d0, i);
  // CHECK: d2 = sycl::fmod((double)i, d1);
  d2 = fmod(i, d1);

  // CHECK: d2 = sycl::frexp(d0, sycl::make_ptr<int, sycl::access::address_space::global_space>(&i));
  d2 = frexp(d0, &i);
  // CHECK: d2 = sycl::frexp((double)i, sycl::make_ptr<int, sycl::access::address_space::global_space>(&i));
  d2 = frexp(i, &i);

  // CHECK: d2 = sycl::hypot(d0, d1);
  d2 = hypot(d0, d1);
  // CHECK: d2 = sycl::hypot((double)i, (double)i);
  d2 = hypot(i, i);
  // CHECK: d2 = sycl::hypot(d0, (double)i);
  d2 = hypot(d0, i);
  // CHECK: d2 = sycl::hypot((double)i, d1);
  d2 = hypot(i, d1);

  // CHECK: d2 = sycl::ilogb(d0);
  d2 = ilogb(d0);
  // CHECK: d2 = sycl::ilogb((double)i);
  d2 = ilogb(i);

  // CHECK: d2 = sycl::ldexp(d0, i);
  d2 = ldexp(d0, i);
  // CHECK: d2 = sycl::ldexp((double)i, i);
  d2 = ldexp(i, i);

  // CHECK: d2 = sycl::lgamma(d0);
  d2 = lgamma(d0);
  // CHECK: d2 = sycl::lgamma((double)i);
  d2 = lgamma(i);

  // CHECK: d2 = sycl::rint(d0);
  d2 = llrint(d0);
  // CHECK: d2 = sycl::rint((double)i);
  d2 = llrint(i);

  // CHECK: d2 = sycl::round(d0);
  d2 = llround(d0);
  // CHECK: d2 = sycl::round((double)i);
  d2 = llround(i);

  // CHECK: d2 = sycl::log10(d0);
  d2 = log10(d0);
  // CHECK: d2 = sycl::log10((double)i);
  d2 = log10(i);

  // CHECK: d2 = sycl::log1p(d0);
  d2 = log1p(d0);
  // CHECK: d2 = sycl::log1p((double)i);
  d2 = log1p(i);

  // CHECK: d2 = sycl::log2(d0);
  d2 = log2(d0);
  // CHECK: d2 = sycl::log2((double)i);
  d2 = log2(i);

  // CHECK: d2 = sycl::logb(d0);
  d2 = logb(d0);
  // CHECK: d2 = sycl::logb((double)i);
  d2 = logb(i);

  // CHECK: d2 = sycl::rint(d0);
  d2 = lrint(d0);
  // CHECK: d2 = sycl::rint((double)i);
  d2 = lrint(i);

  // CHECK: d2 = sycl::round(d0);
  d2 = lround(d0);
  // CHECK: d2 = sycl::round((double)i);
  d2 = lround(i);

  // CHECK: d2 = sycl::nan(0u);
  d2 = nan("");

  // CHECK: d2 = sycl::pow(d0, d1);
  d2 = pow(d0, d1);
  // CHECK: d2 = sycl::pown((float)i, i);
  d2 = pow(i, i);
  // CHECK: d2 = sycl::pown(d0, i);
  d2 = pow(d0, i);
  // CHECK: d2 = sycl::pow((double)i, d1);
  d2 = pow(i, d1);

  // CHECK: sycl::pown(f, 1);
  float f;
  pow(f, 1);

  // CHECK: d2 = sycl::remainder(d0, d1);
  d2 = remainder(d0, d1);
  // CHECK: d2 = sycl::remainder((double)i, (double)i);
  d2 = remainder(i, i);
  // CHECK: d2 = sycl::remainder(d0, (double)i);
  d2 = remainder(d0, i);
  // CHECK: d2 = sycl::remainder((double)i, d1);
  d2 = remainder(i, d1);

  // CHECK: d2 = sycl::rint(d0);
  d2 = rint(d0);
  // CHECK: d2 = sycl::rint((double)i);
  d2 = rint(i);

  // CHECK: d2 = sycl::round(d0);
  d2 = round(d0);
  // CHECK: d2 = sycl::round((double)i);
  d2 = round(i);

  // CHECK: d2 = sycl::rsqrt(d0);
  d2 = rsqrt(d0);
  // CHECK: d2 = sycl::rsqrt((double)i);
  d2 = rsqrt((double)i);

    // CHECK: d2 = sycl::sin(d0);
  d2 = sin(d0);
  // CHECK: d2 = sycl::sin((double)i);
  d2 = sin(i);

  // CHECK: d2 = sycl::sinh(d0);
  d2 = sinh(d0);
  // CHECK: d2 = sycl::sinh((double)i);
  d2 = sinh(i);

  // CHECK: d2 = sycl::sinpi(d0);
  d2 = sinpi(d0);
  // CHECK: d2 = sycl::sinpi((double)i);
  d2 = sinpi((double)i);

  // CHECK: d2 = sycl::sqrt(d0);
  d2 = sqrt(d0);
  // CHECK: d2 = sycl::sqrt((double)i);
  d2 = sqrt(i);

  // CHECK: d2 = sycl::tan(d0);
  d2 = tan(d0);
  // CHECK: d2 = sycl::tan((double)i);
  d2 = tan(i);

  // CHECK: d2 = sycl::tanh(d0);
  d2 = tanh(d0);
  // CHECK: d2 = sycl::tanh((double)i);
  d2 = tanh(i);

  // CHECK: d2 = sycl::tgamma(d0);
  d2 = tgamma(d0);
  // CHECK: d2 = sycl::tgamma((double)i);
  d2 = tgamma(i);

  // CHECK: d2 = sycl::trunc(d0);
  d2 = trunc(d0);
  // CHECK: d2 = sycl::trunc((double)i);
  d2 = trunc(i);

  // CHECK: d0 = sycl::fmin(d0, d1);
  d0 = fmin(d0, d1);
  // CHECK: d0 = sycl::fmin((double)i, (double)i);
  d0 = fmin(i, i);
  // CHECK: d0 = sycl::fmin(d0, (double)i);
  d0 = fmin(d0, i);
  // CHECK: d0 = sycl::fmin((double)i, d1);
  d0 = fmin(i, d1);

  // CHECK: d0 = sycl::fmax(d0, d1);
  d0 = fmax(d0, d1);
  // CHECK: d0 = sycl::fmax((double)i, (double)i);
  d0 = fmax(i, i);
  // CHECK: d0 = sycl::fmax(d0, (double)i);
  d0 = fmax(d0, i);
  // CHECK: d0 = sycl::fmax((double)i, d1);
  d0 = fmax(i, d1);

  // CHECK: d1 = sycl::floor(d1);
  d1 = floor(d1);
  // CHECK: d1 = sycl::floor((double)i);
  d1 = floor(i);

  // CHECK: d2 = sycl::fma(d0, d1, d2);
  d2 = fma(d0, d1, d2);
  // CHECK: d2 = sycl::fma((double)i, (double)i, (double)i);
  d2 = fma(i, i, i);
  // CHECK: d2 = sycl::fma(d0, (double)i, (double)i);
  d2 = fma(d0, i, i);
  // CHECK: d2 = sycl::fma((double)i, d1, (double)i);
  d2 = fma(i, d1, i);
  // CHECK: d2 = sycl::fma((double)i, (double)i, d2);
  d2 = fma(i, i, d2);
  // CHECK: d2 = sycl::fma(d0, d1, (double)i);
  d2 = fma(d0, d1, i);
  // CHECK: d2 = sycl::fma(d0, (double)i, d2);
  d2 = fma(d0, i, d2);
  // CHECK: d2 = sycl::fma((double)i, d1, d2);
  d2 = fma(i, d1, d2);

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
}

__global__ void math_direct2() {
  int i;
  unsigned u;
  long l;
  long long ll;
  unsigned long long ull;

  // CHECK: i = sycl::clz(i);
  // CHECK-NEXT: i = sycl::clz(ll);
  // CHECK-NEXT: i = sycl::hadd(i, i);
  // CHECK-NEXT: i = sycl::mul24(i, i);
  // CHECK-NEXT: i = sycl::mul_hi(i, i);
  // CHECK-NEXT: i = sycl::popcount(u);
  // CHECK-NEXT: i = sycl::popcount(ull);
  i = __clz(i);
  i = __clzll(ll);
  i = __hadd(i, i);
  i = __mul24(i, i);
  i = __mulhi(i, i);
  i = __popc(u);
  i = __popcll(ull);

  // CHECK: sycl::clz((int)u);
  // CHECK-NEXT: sycl::clz((long long)ull);
  // CHECK-NEXT: sycl::hadd((int)u, (int)u);
  // CHECK-NEXT: sycl::mul24((int)u, (int)u);
  // CHECK-NEXT: sycl::mul_hi((int)u, (int)u);
  __clz(u);
  __clzll(ull);
  __hadd(u, u);
  __mul24(u, u);
  __mulhi(u, u);
}

__global__ void math_direct3() {
  int i = 0;
  // CHECK: sycl::max((unsigned int)i, (unsigned int)item_ct1.get_local_id(2));
  // CHECK-NEXT: sycl::max((unsigned int)i, (unsigned int)item_ct1.get_local_id(1));
  // CHECK-NEXT: sycl::max((unsigned int)i, (unsigned int)item_ct1.get_local_id(0));
  // CHECK-NEXT: sycl::max((unsigned int)item_ct1.get_local_id(2), (unsigned int)i);
  // CHECK-NEXT: sycl::max((unsigned int)item_ct1.get_local_id(1), (unsigned int)i);
  // CHECK-NEXT: sycl::max((unsigned int)item_ct1.get_local_id(0), (unsigned int)i);
  max(i, threadIdx.x);
  max(i, threadIdx.y);
  max(i, threadIdx.z);
  max(threadIdx.x, i);
  max(threadIdx.y, i);
  max(threadIdx.z, i);

  // CHECK: sycl::max((unsigned int)i, (unsigned int)item_ct1.get_group(2));
  // CHECK-NEXT: sycl::max((unsigned int)i, (unsigned int)item_ct1.get_group(1));
  // CHECK-NEXT: sycl::max((unsigned int)i, (unsigned int)item_ct1.get_group(0));
  // CHECK-NEXT: sycl::max((unsigned int)item_ct1.get_group(2), (unsigned int)i);
  // CHECK-NEXT: sycl::max((unsigned int)item_ct1.get_group(1), (unsigned int)i);
  // CHECK-NEXT: sycl::max((unsigned int)item_ct1.get_group(0), (unsigned int)i);
  max(i, blockIdx.x);
  max(i, blockIdx.y);
  max(i, blockIdx.z);
  max(blockIdx.x, i);
  max(blockIdx.y, i);
  max(blockIdx.z, i);

  // CHECK: sycl::max((unsigned int)i, (unsigned int)item_ct1.get_local_range(2));
  // CHECK-NEXT: sycl::max((unsigned int)i, (unsigned int)item_ct1.get_local_range(1));
  // CHECK-NEXT: sycl::max((unsigned int)i, (unsigned int)item_ct1.get_local_range(0));
  // CHECK-NEXT: sycl::max((unsigned int)item_ct1.get_local_range(2), (unsigned int)i);
  // CHECK-NEXT: sycl::max((unsigned int)item_ct1.get_local_range(1), (unsigned int)i);
  // CHECK-NEXT: sycl::max((unsigned int)item_ct1.get_local_range(0), (unsigned int)i);
  max(i, blockDim.x);
  max(i, blockDim.y);
  max(i, blockDim.z);
  max(blockDim.x, i);
  max(blockDim.y, i);
  max(blockDim.z, i);

  // CHECK: sycl::min((unsigned int)i, (unsigned int)item_ct1.get_local_id(2));
  // CHECK-NEXT: sycl::min((unsigned int)i, (unsigned int)item_ct1.get_local_id(1));
  // CHECK-NEXT: sycl::min((unsigned int)i, (unsigned int)item_ct1.get_local_id(0));
  // CHECK-NEXT: sycl::min((unsigned int)item_ct1.get_local_id(2), (unsigned int)i);
  // CHECK-NEXT: sycl::min((unsigned int)item_ct1.get_local_id(1), (unsigned int)i);
  // CHECK-NEXT: sycl::min((unsigned int)item_ct1.get_local_id(0), (unsigned int)i);
  min(i, threadIdx.x);
  min(i, threadIdx.y);
  min(i, threadIdx.z);
  min(threadIdx.x, i);
  min(threadIdx.y, i);
  min(threadIdx.z, i);

  // CHECK: sycl::min((unsigned int)i, (unsigned int)item_ct1.get_group(2));
  // CHECK-NEXT: sycl::min((unsigned int)i, (unsigned int)item_ct1.get_group(1));
  // CHECK-NEXT: sycl::min((unsigned int)i, (unsigned int)item_ct1.get_group(0));
  // CHECK-NEXT: sycl::min((unsigned int)item_ct1.get_group(2), (unsigned int)i);
  // CHECK-NEXT: sycl::min((unsigned int)item_ct1.get_group(1), (unsigned int)i);
  // CHECK-NEXT: sycl::min((unsigned int)item_ct1.get_group(0), (unsigned int)i);
  min(i, blockIdx.x);
  min(i, blockIdx.y);
  min(i, blockIdx.z);
  min(blockIdx.x, i);
  min(blockIdx.y, i);
  min(blockIdx.z, i);

  // CHECK: sycl::min((unsigned int)i, (unsigned int)item_ct1.get_local_range(2));
  // CHECK-NEXT: sycl::min((unsigned int)i, (unsigned int)item_ct1.get_local_range(1));
  // CHECK-NEXT: sycl::min((unsigned int)i, (unsigned int)item_ct1.get_local_range(0));
  // CHECK-NEXT: sycl::min((unsigned int)item_ct1.get_local_range(2), (unsigned int)i);
  // CHECK-NEXT: sycl::min((unsigned int)item_ct1.get_local_range(1), (unsigned int)i);
  // CHECK-NEXT: sycl::min((unsigned int)item_ct1.get_local_range(0), (unsigned int)i);
  min(i, blockDim.x);
  min(i, blockDim.y);
  min(i, blockDim.z);
  min(blockDim.x, i);
  min(blockDim.y, i);
  min(blockDim.z, i);
}

int main() {
  math_direct<<<1, 1>>>();  
  math_direct2<<<1, 1>>>();  
  math_direct3<<<1, 1>>>();  
}
