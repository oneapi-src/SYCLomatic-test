// ===------------------- math.cpp ---------- -*- C++ -* ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <dpct/dpct.hpp>

int main() {
  double d;
  float f;
  int i;
  sycl::half h;
  sycl::half2 h2;
  unsigned u;
  dpct::fast_length(&f, i);
  dpct::length(&d, i);
  dpct::compare(h, h, std::equal_to<>());
  dpct::compare(h2, h2, std::equal_to<>());
  dpct::unordered_compare(h, h, std::equal_to<>());
  dpct::unordered_compare(h2, h2, std::equal_to<>());
  dpct::compare_both(h2, h2, std::equal_to<>());
  dpct::unordered_compare_both(h2, h2, std::equal_to<>());
  dpct::isnan(h2);
  dpct::vectorized_binary<sycl::short2>(u, u, dpct::abs_diff());
  dpct::vectorized_binary<sycl::short2>(u, u, dpct::add_sat());
  dpct::vectorized_binary<sycl::short2>(u, u, dpct::rhadd());
  dpct::vectorized_binary<sycl::short2>(u, u, dpct::hadd());
  dpct::vectorized_binary<sycl::short2>(u, u, dpct::maximum());
  dpct::vectorized_binary<sycl::short2>(u, u, dpct::minimum());
  dpct::vectorized_binary<sycl::short2>(u, u, dpct::sub_sat());
  dpct::vectorized_unary<sycl::short2>(u, dpct::abs());
  dpct::vectorized_sum_abs_diff<sycl::short2>(u, u);
  printf("test pass!\n");
  return 0;
}
