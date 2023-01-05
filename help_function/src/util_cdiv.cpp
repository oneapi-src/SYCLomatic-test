//===-------------- util_cdiv.cpp --------------------*- C++ -*----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------===//
//test_feature:Util_cdiv

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <complex>

#include <cmath>

template <typename T>
bool check(T x, float e[], int& index) {
    float precison = 0.001f;
    if((std::abs(x.x() - e[index++]) < precison) && (std::abs(x.y() - e[index++]) < precison)) {
        return true;
    }
    return false;
}

template <>
bool check<float>(float x, float e[], int& index) {
  float precison = 0.001f;
  if(std::abs(x - e[index++]) < precison) {
      return true;
  }
  return false;
}

template <>
bool check<double>(double x, float e[], int& index) {
    float precison = 0.001f;
  if(std::abs(x - e[index++]) < precison) {
      return true;
  }
  return false;
}

void kernel(int *result) {

    sycl::float2 f1, f2;
    sycl::double2 d1, d2;

    f1 = sycl::float2(1.8, -2.7);
    f2 = sycl::float2(-3.6, 4.5);
    d1 = sycl::double2(5.4, -6.3);
    d2 = sycl::double2(-7.2, 8.1);

    int index = 0;
    bool r = true;
    float expect[4] = {-0.765517, 0.013793, -0.560976, 0.048780};

    auto a1 = dpct::cdiv(d1, d2);
    r = r && check(a1, expect, index);

    auto a2 = dpct::cdiv(f1, f2);
    r = r && check(a2, expect, index);

    *result = r;
}

int main() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

    sycl::float2 f1, f2;
    sycl::double2 d1, d2;

    f1 = sycl::float2(1.8, -2.7);
    f2 = sycl::float2(-3.6, 4.5);
    d1 = sycl::double2(5.4, -6.3);
    d2 = sycl::double2(-7.2, 8.1);
    int index = 0;
    bool r = true;
    float expect[4] = {-0.765517, 0.013793, -0.560976, 0.048780};

    auto a1 = dpct::cdiv(d1, d2);
    r = r && check(a1, expect, index);

    auto a2 = dpct::cdiv(f1, f2);
    r = r && check(a2, expect, index);


    int *result = nullptr;
    result = sycl::malloc_shared<int>(1, q_ct1);
    *result = 0;

    q_ct1.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
      [=](sycl::nd_item<3> item_ct1) {
        kernel(result);
      });
    dev_ct1.queues_wait_and_throw();

    if(*result && r) {
      std::cout << "pass" << std::endl;
    } else {
      std::cout << "fail" << std::endl;
      exit(-1);
    }
    return 0;
}
