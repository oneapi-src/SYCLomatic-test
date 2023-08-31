// ===----------- math_extend_func.cpp ---------- -*- C++ -* --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <cmath>
#include <cstdint>
#include <dpct/dpct.hpp>
#include <dpct/math.hpp>
#include <limits>
#include <stdio.h>
#include <sycl/sycl.hpp>

#define CHECK(S, REF)                                                          \
  {                                                                            \
    auto ret = S;                                                              \
    if (ret != REF) {                                                          \
      return {#S, REF};                                                        \
    }                                                                          \
  }

const auto INT32MAX = std::numeric_limits<int32_t>::max();
const auto INT32MIN = std::numeric_limits<int32_t>::min();
const auto UINT32MAX = std::numeric_limits<uint32_t>::max();
const auto UINT32MIN = std::numeric_limits<uint32_t>::min();
const int b = 4, c = 5, d = 6;

std::pair<char *, int> vadd() {
  CHECK(dpct::extend_add<int32_t>(3, 4), 7);
  CHECK(dpct::extend_add<uint32_t>(b, c), 9);
  CHECK(dpct::extend_add_sat<int32_t>(b, INT32MAX), INT32MAX);
  CHECK(dpct::extend_add_sat<uint32_t>(UINT32MAX, INT32MAX), UINT32MAX);
  CHECK(dpct::extend_add_sat<int32_t>(b, -20, d, sycl::plus<>()), -10);
  CHECK(dpct::extend_add_sat<int32_t>(b, c, -20, sycl::minimum<>()), -20);
  CHECK(dpct::extend_add_sat<int32_t>(b, (-33), 9, sycl::maximum<>()), 9);

  return {nullptr, 0};
}

std::pair<char *, int> vsub() {
  CHECK(dpct::extend_sub<int32_t>(3, 4), -1);
  CHECK(dpct::extend_sub<uint32_t>(c, b), 1);
  CHECK(dpct::extend_sub_sat<int32_t>(10, INT32MIN), INT32MAX);
  CHECK(dpct::extend_sub_sat<uint32_t>(UINT32MIN, 1), UINT32MIN);
  CHECK(dpct::extend_sub_sat<int32_t>(b, -20, d, sycl::plus<>()), 30);
  CHECK(dpct::extend_sub_sat<int32_t>(b, c, -20, sycl::minimum<>()), -20);
  CHECK(dpct::extend_sub_sat<int32_t>(b, (-33), 9, sycl::maximum<>()), 37);

  return {nullptr, 0};
}

std::pair<char *, int> vabsdiff() {
  CHECK(dpct::extend_absdiff<int32_t>(3, 4), 1);
  CHECK(dpct::extend_absdiff<uint32_t>(c, b), 1);
  CHECK(dpct::extend_absdiff_sat<int32_t>(10, INT32MIN), INT32MAX);
  CHECK(dpct::extend_absdiff_sat<uint32_t>(UINT32MIN, 1), 1);
  CHECK(dpct::extend_absdiff_sat<int32_t>(b, -20, d, sycl::plus<>()), 30);
  CHECK(dpct::extend_absdiff_sat<int32_t>(b, c, -20, sycl::minimum<>()), -20);
  CHECK(dpct::extend_absdiff_sat<int32_t>(b, (-33), 9, sycl::maximum<>()), 37);

  return {nullptr, 0};
}

std::pair<char *, int> vmin() {
  CHECK(dpct::extend_min<int32_t>(3, 4), 3);
  CHECK(dpct::extend_min<uint32_t>(c, b), 4);
  CHECK(dpct::extend_min_sat<int32_t>(UINT32MAX, 1), 1);
  CHECK(dpct::extend_min_sat<uint32_t>(10, (-1)), 0);
  CHECK(dpct::extend_min_sat<int32_t>(b, -20, d, sycl::plus<>()), -14);
  CHECK(dpct::extend_min_sat<int32_t>(b, c, -20, sycl::minimum<>()), -20);
  CHECK(dpct::extend_min_sat<int32_t>(b, (-33), 9, sycl::maximum<>()), 9);

  return {nullptr, 0};
}

std::pair<char *, int> vmax() {
  CHECK(dpct::extend_max<int32_t>(3, 4), 4);
  CHECK(dpct::extend_max<uint32_t>(c, b), 5);
  CHECK(dpct::extend_max_sat<int32_t>(UINT32MAX, 1), INT32MAX);
  CHECK(dpct::extend_max_sat<uint32_t>(UINT32MAX, 1), UINT32MAX);
  CHECK(dpct::extend_max_sat<int32_t>(b, -20, d, sycl::plus<>()), 10);
  CHECK(dpct::extend_max_sat<int32_t>(b, c, -20, sycl::minimum<>()), -20);
  CHECK(dpct::extend_max_sat<int32_t>(b, (-33), 9, sycl::maximum<>()), 9);

  return {nullptr, 0};
}

void test(const sycl::stream &s, int *ec) {
  {
    auto res = vadd();
    if (res.first != nullptr) {
      s << res.first << " = " << res.second << " check failed!\n";
      *ec = 1;
      return;
    }
    s << "vadd check passed!\n";
  }
  {
    auto res = vsub();
    if (res.first != nullptr) {
      s << res.first << " = " << res.second << " check failed!\n";
      *ec = 1;
      return;
    }
    s << "vsub check passed!\n";
  }
  {
    auto res = vabsdiff();
    if (res.first != nullptr) {
      s << res.first << " = " << res.second << " check failed!\n";
      *ec = 1;
      return;
    }
    s << "vabsdiff check passed!\n";
  }
  {
    auto res = vmin();
    if (res.first != nullptr) {
      s << res.first << " = " << res.second << " check failed!\n";
      *ec = 1;
      return;
    }
    s << "vmin check passed!\n";
  }
  {
    auto res = vmax();
    if (res.first != nullptr) {
      s << res.first << " = " << res.second << " check failed!\n";
      *ec = 1;
      return;
    }
    s << "vmax check passed!\n";
  }
}

int main() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  int *ec = nullptr;
  ec = sycl::malloc_shared<int>(1, q_ct1);
  *ec = 0;
  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::stream out(1024, 256, cgh);
    cgh.parallel_for(1, [=](sycl::item<1> it) { test(out, ec); });
  });
  dev_ct1.queues_wait_and_throw();
  return *ec;
}
