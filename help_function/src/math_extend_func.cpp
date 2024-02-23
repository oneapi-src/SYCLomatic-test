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

#define CHECK(NUM, S, REF)                                                     \
  {                                                                            \
    auto ret = S;                                                              \
    if (ret != REF) {                                                          \
      return NUM;                                                              \
    }                                                                          \
  }

const auto INT32MAX = std::numeric_limits<int32_t>::max();
const auto INT32MIN = std::numeric_limits<int32_t>::min();
const auto UINT32MAX = std::numeric_limits<uint32_t>::max();
const auto UINT32MIN = std::numeric_limits<uint32_t>::min();
const int b = 4, c = 5, d = 6;

int vadd() {
  CHECK(1, dpct::extend_add<int32_t>(3, 4), 7);
  CHECK(2, dpct::extend_add<uint32_t>(b, c), 9);
  CHECK(3, dpct::extend_add_sat<int32_t>(b, INT32MAX), INT32MAX);
  CHECK(4, dpct::extend_add_sat<uint32_t>(UINT32MAX, INT32MAX), UINT32MAX);
  CHECK(5, dpct::extend_add_sat<int32_t>(b, -20, d, sycl::plus<>()), -10);
  CHECK(6, dpct::extend_add_sat<int32_t>(b, c, -20, sycl::minimum<>()), -20);
  CHECK(7, dpct::extend_add_sat<int32_t>(b, (-33), 9, sycl::maximum<>()), 9);

  return 0;
}

int vsub() {
  CHECK(101, dpct::extend_sub<int32_t>(3, 4), -1);
  CHECK(102, dpct::extend_sub<uint32_t>(c, b), 1);
  CHECK(103, dpct::extend_sub_sat<int32_t>(10, INT32MIN), INT32MAX);
  CHECK(104, dpct::extend_sub_sat<uint32_t>(UINT32MIN, 1), UINT32MIN);
  CHECK(105, dpct::extend_sub_sat<int32_t>(b, -20, d, sycl::plus<>()), 30);
  CHECK(106, dpct::extend_sub_sat<int32_t>(b, c, -20, sycl::minimum<>()), -20);
  CHECK(107, dpct::extend_sub_sat<int32_t>(b, (-33), 9, sycl::maximum<>()), 37);

  return 0;
}

int vabsdiff() {
  CHECK(201, dpct::extend_absdiff<int32_t>(3, 4), 1);
  CHECK(202, dpct::extend_absdiff<uint32_t>(c, b), 1);
  CHECK(203, dpct::extend_absdiff_sat<int32_t>(10, INT32MIN), INT32MAX);
  CHECK(204, dpct::extend_absdiff_sat<uint32_t>(UINT32MIN, 1), 1);
  CHECK(205, dpct::extend_absdiff_sat<int32_t>(b, -20, d, sycl::plus<>()), 30);
  CHECK(206, dpct::extend_absdiff_sat<int32_t>(b, c, -20, sycl::minimum<>()),
        -20);
  CHECK(207, dpct::extend_absdiff_sat<int32_t>(b, (-33), 9, sycl::maximum<>()),
        37);

  return 0;
}

int vmin() {
  CHECK(301, dpct::extend_min<int32_t>(3, 4), 3);
  CHECK(302, dpct::extend_min<uint32_t>(c, b), 4);
  CHECK(303, dpct::extend_min_sat<int32_t>(UINT32MAX, 1), 1);
  CHECK(304, dpct::extend_min_sat<uint32_t>(10, (-1)), 0);
  CHECK(305, dpct::extend_min_sat<int32_t>(b, -20, d, sycl::plus<>()), -14);
  CHECK(306, dpct::extend_min_sat<int32_t>(b, c, -20, sycl::minimum<>()), -20);
  CHECK(307, dpct::extend_min_sat<int32_t>(b, (-33), 9, sycl::maximum<>()), 9);

  return 0;
}

int vmax() {
  CHECK(401, dpct::extend_max<int32_t>(3, 4), 4);
  CHECK(402, dpct::extend_max<uint32_t>(c, b), 5);
  CHECK(403, dpct::extend_max_sat<int32_t>(UINT32MAX, 1), INT32MAX);
  CHECK(404, dpct::extend_max_sat<uint32_t>(UINT32MAX, 1), UINT32MAX);
  CHECK(405, dpct::extend_max_sat<int32_t>(b, -20, d, sycl::plus<>()), 10);
  CHECK(406, dpct::extend_max_sat<int32_t>(b, c, -20, sycl::minimum<>()), -20);
  CHECK(407, dpct::extend_max_sat<int32_t>(b, (-33), 9, sycl::maximum<>()), 9);

  return 0;
}

void test(int *ec) {
  {
    *ec = vadd();
    if (*ec != 0) {
      return;
    }
  }
  {
    *ec = vsub();
    if (*ec != 0) {
      return;
    }
  }
  {
    *ec = vabsdiff();
    if (*ec != 0) {
      return;
    }
  }
  {
    *ec = vmin();
    if (*ec != 0) {
      return;
    }
  }
  {
    *ec = vmax();
    if (*ec != 0) {
      return;
    }
  }
}

int main() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  int *ec = nullptr;
  ec = sycl::malloc_shared<int>(1, q_ct1);
  *ec = 0;
  q_ct1.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(1, [=](sycl::item<1> it) { test(ec); });
  });
  dev_ct1.queues_wait_and_throw();
  if (*ec != 0)
    printf("test %d failed\n", *ec);
  return *ec;
}
