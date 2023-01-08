// ===--- r2cc2r_1d_outofplace.dp.cpp -------------------------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===---------------------------------------------------------------------===//

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <oneapi/mkl.hpp>
#include <dpct/fft_utils.hpp>
#include "common.h"
#include <cstring>
#include <iostream>

bool r2cc2r_1d_outofplace() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  dpct::fft::fft_engine *plan_fwd;
  float forward_idata_h[14];
  set_value(forward_idata_h, 7);
  set_value(forward_idata_h + 7, 7);

  float* forward_idata_d;
  sycl::float2 *forward_odata_d;
  float* backward_odata_d;
  forward_idata_d = (float *)sycl::malloc_device(2 * sizeof(float) * 7, q_ct1);
  forward_odata_d = (sycl::float2 *)sycl::malloc_device(
      2 * sizeof(sycl::float2) * (7 / 2 + 1), q_ct1);
  backward_odata_d = (float *)sycl::malloc_device(2 * sizeof(float) * 7, q_ct1);
  q_ct1.memcpy(forward_idata_d, forward_idata_h, 2 * sizeof(float) * 7).wait();

  plan_fwd = dpct::fft::fft_engine::create(
      &q_ct1, 7, dpct::fft::fft_type::real_float_to_complex_float, 2);
  plan_fwd->compute<float, sycl::float2>(forward_idata_d, forward_odata_d,
                                         dpct::fft::fft_direction::forward);
  dev_ct1.queues_wait_and_throw();
  sycl::float2 forward_odata_h[8];
  q_ct1
      .memcpy(forward_odata_h, forward_odata_d,
              2 * sizeof(sycl::float2) * (7 / 2 + 1))
      .wait();

  sycl::float2 forward_odata_ref[8];
  forward_odata_ref[0] = sycl::float2{21, 0};
  forward_odata_ref[1] = sycl::float2{-3.5, 7.26783};
  forward_odata_ref[2] = sycl::float2{-3.5, 2.79116};
  forward_odata_ref[3] = sycl::float2{-3.5, 0.798852};
  forward_odata_ref[4] = sycl::float2{21, 0};
  forward_odata_ref[5] = sycl::float2{-3.5, 7.26783};
  forward_odata_ref[6] = sycl::float2{-3.5, 2.79116};
  forward_odata_ref[7] = sycl::float2{-3.5, 0.798852};

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
      &q_ct1, 7, dpct::fft::fft_type::complex_float_to_real_float, 2);
  plan_bwd->compute<sycl::float2, float>(forward_odata_d, backward_odata_d,
                                         dpct::fft::fft_direction::backward);
  dev_ct1.queues_wait_and_throw();
  float backward_odata_h[14];
  q_ct1.memcpy(backward_odata_h, backward_odata_d, 2 * sizeof(float) * 7)
      .wait();

  float backward_odata_ref[14];
  backward_odata_ref[0] = 0;
  backward_odata_ref[1] = 7;
  backward_odata_ref[2] = 14;
  backward_odata_ref[3] = 21;
  backward_odata_ref[4] = 28;
  backward_odata_ref[5] = 35;
  backward_odata_ref[6] = 42;
  backward_odata_ref[7] = 0;
  backward_odata_ref[8] = 7;
  backward_odata_ref[9] = 14;
  backward_odata_ref[10] = 21;
  backward_odata_ref[11] = 28;
  backward_odata_ref[12] = 35;
  backward_odata_ref[13] = 42;

  sycl::free(forward_idata_d, q_ct1);
  sycl::free(forward_odata_d, q_ct1);
  sycl::free(backward_odata_d, q_ct1);

  dpct::fft::fft_engine::destroy(plan_bwd);

  if (!compare(backward_odata_ref, backward_odata_h, 14)) {
    std::cout << "backward_odata_h:" << std::endl;
    print_values(backward_odata_h, 14);
    std::cout << "backward_odata_ref:" << std::endl;
    print_values(backward_odata_ref, 14);
    return false;
  }
  return true;
}



#ifdef DEBUG_FFT
int main() {
#define FUNC r2cc2r_1d_outofplace
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

