// ===--- r2cc2r_many_2d_outofplace_basic.dp.cpp --------------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===---------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <oneapi/mkl.hpp>
#include <dpct/fft_utils.hpp>
#include "common.h"
#include <cstring>
#include <iostream>

bool r2cc2r_many_2d_outofplace_basic() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  dpct::fft::fft_engine *plan_fwd;
  float forward_idata_h[2/*n0*/ * 3/*n1*/ * 2/*batch*/];
  set_value(forward_idata_h, 6);
  set_value(forward_idata_h + 6, 6);

  float* forward_idata_d;
  sycl::float2 *forward_odata_d;
  float* backward_odata_d;
  forward_idata_d =
      (float *)sycl::malloc_device(sizeof(float) * 2 * 3 * 2, q_ct1);
  forward_odata_d = (sycl::float2 *)sycl::malloc_device(
      2 * 2 * sizeof(sycl::float2) * (3 / 2 + 1), q_ct1);
  backward_odata_d =
      (float *)sycl::malloc_device(sizeof(float) * 2 * 3 * 2, q_ct1);
  q_ct1.memcpy(forward_idata_d, forward_idata_h, sizeof(float) * 2 * 3 * 2)
      .wait();

  int n[2] = {2, 3};
  plan_fwd = dpct::fft::fft_engine::create(
      &q_ct1, 2, n, nullptr, 0, 0, nullptr, 0, 0,
      dpct::fft::fft_type::real_float_to_complex_float, 2);
  plan_fwd->compute<float, sycl::float2>(forward_idata_d, forward_odata_d,
                                         dpct::fft::fft_direction::forward);
  dev_ct1.queues_wait_and_throw();
  sycl::float2 forward_odata_h[8];
  q_ct1
      .memcpy(forward_odata_h, forward_odata_d,
              2 * 2 * sizeof(sycl::float2) * (3 / 2 + 1))
      .wait();

  sycl::float2 forward_odata_ref[8];
  forward_odata_ref[0] = sycl::float2{15, 0};
  forward_odata_ref[1] = sycl::float2{-3, 1.73205};
  forward_odata_ref[2] = sycl::float2{-9, 0};
  forward_odata_ref[3] = sycl::float2{0, 0};
  forward_odata_ref[4] = sycl::float2{15, 0};
  forward_odata_ref[5] = sycl::float2{-3, 1.73205};
  forward_odata_ref[6] = sycl::float2{-9, 0};
  forward_odata_ref[7] = sycl::float2{0, 0};

  dpct::fft::fft_engine::destroy(plan_fwd);

  if (!compare(forward_odata_ref, forward_odata_h, 8)) {
    std::cout << "forward_odata_h:" << std::endl;
    print_values(forward_odata_h, 8);
    std::cout << "forward_odata_ref:" << std::endl;
    print_values(forward_odata_ref, 8);

    sycl::free(forward_idata_d, q_ct1);
    sycl::free(forward_odata_d, q_ct1);
    sycl::free(backward_odata_d, q_ct1);

    return false;
  }

  dpct::fft::fft_engine *plan_bwd;
  plan_bwd = dpct::fft::fft_engine::create(
      &q_ct1, 2, n, nullptr, 0, 0, nullptr, 0, 0,
      dpct::fft::fft_type::complex_float_to_real_float, 2);
  plan_bwd->compute<sycl::float2, float>(forward_odata_d, backward_odata_d,
                                         dpct::fft::fft_direction::backward);
  dev_ct1.queues_wait_and_throw();
  float backward_odata_h[12];
  q_ct1.memcpy(backward_odata_h, backward_odata_d, sizeof(float) * 12).wait();

  float backward_odata_ref[12];
  backward_odata_ref[0] =  0;
  backward_odata_ref[1] =  6;
  backward_odata_ref[2] =  12;
  backward_odata_ref[3] =  18;
  backward_odata_ref[4] =  24;
  backward_odata_ref[5] =  30;
  backward_odata_ref[6] =  0;
  backward_odata_ref[7] =  6;
  backward_odata_ref[8] =  12;
  backward_odata_ref[9] =  18;
  backward_odata_ref[10] = 24;
  backward_odata_ref[11] = 30;

  sycl::free(forward_idata_d, q_ct1);
  sycl::free(forward_odata_d, q_ct1);
  sycl::free(backward_odata_d, q_ct1);

  dpct::fft::fft_engine::destroy(plan_bwd);

  if (!compare(backward_odata_ref, backward_odata_h, 12)) {
    std::cout << "backward_odata_h:" << std::endl;
    print_values(backward_odata_h, 12);
    std::cout << "backward_odata_ref:" << std::endl;
    print_values(backward_odata_ref, 12);
    return false;
  }
  return true;
}


#ifdef DEBUG_FFT
int main() {
#define FUNC r2cc2r_many_2d_outofplace_basic
  bool res = FUNC();
  cudaDeviceSynchronize();
  if (!res) {
    std::cout << "Fail" << std::endl;
    return -1;
  }
  std::cout << "Pass" << std::endl;
  return 0;
}
#endif

