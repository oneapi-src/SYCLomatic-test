// ===--- r2cc2r_2d_outofplace.dp.cpp -------------------------*- C++ -*---===//
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

bool r2cc2r_2d_outofplace() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  dpct::fft::fft_engine *plan_fwd;
  float forward_idata_h[4][5];
  set_value((float*)forward_idata_h, 20);

  float* forward_idata_d;
  sycl::float2 *forward_odata_d;
  float* backward_odata_d;
  forward_idata_d = sycl::malloc_device<float>(20, q_ct1);
  forward_odata_d = (sycl::float2 *)sycl::malloc_device(
      sizeof(sycl::float2) * (5 / 2 + 1) * 4, q_ct1);
  backward_odata_d = sycl::malloc_device<float>(20, q_ct1);
  q_ct1.memcpy(forward_idata_d, forward_idata_h, sizeof(float) * 20).wait();

  plan_fwd = dpct::fft::fft_engine::create(
      &q_ct1, 4, 5, dpct::fft::fft_type::real_float_to_complex_float);
  plan_fwd->compute<float, sycl::float2>(forward_idata_d, forward_odata_d,
                                         dpct::fft::fft_direction::forward);
  dev_ct1.queues_wait_and_throw();
  sycl::float2 forward_odata_h[12];
  q_ct1
      .memcpy(forward_odata_h, forward_odata_d,
              sizeof(sycl::float2) * (5 / 2 + 1) * 4)
      .wait();

  sycl::float2 forward_odata_ref[12];
  forward_odata_ref[0] = sycl::float2{190, 0};
  forward_odata_ref[1] = sycl::float2{-10, 13.7638};
  forward_odata_ref[2] = sycl::float2{-10, 3.2492};
  forward_odata_ref[3] = sycl::float2{-50, 50};
  forward_odata_ref[4] = sycl::float2{0, 0};
  forward_odata_ref[5] = sycl::float2{0, 0};
  forward_odata_ref[6] = sycl::float2{-50, 0};
  forward_odata_ref[7] = sycl::float2{0, 0};
  forward_odata_ref[8] = sycl::float2{0, 0};
  forward_odata_ref[9] = sycl::float2{-50, -50};
  forward_odata_ref[10] = sycl::float2{0, 0};
  forward_odata_ref[11] = sycl::float2{0, 0};

  dpct::fft::fft_engine::destroy(plan_fwd);

  if (!compare(forward_odata_ref, forward_odata_h, 12)) {
    std::cout << "forward_odata_h:" << std::endl;
    print_values(forward_odata_h, 12);
    std::cout << "forward_odata_ref:" << std::endl;
    print_values(forward_odata_ref, 12);

    sycl::free(forward_idata_d, q_ct1);
    sycl::free(forward_odata_d, q_ct1);
    sycl::free(backward_odata_d, q_ct1);

    return false;
  }

  dpct::fft::fft_engine *plan_bwd;
  plan_bwd = dpct::fft::fft_engine::create(
      &q_ct1, 4, 5, dpct::fft::fft_type::complex_float_to_real_float);
  plan_bwd->compute<sycl::float2, float>(forward_odata_d, backward_odata_d,
                                         dpct::fft::fft_direction::backward);
  dev_ct1.queues_wait_and_throw();
  float backward_odata_h[20];
  q_ct1.memcpy(backward_odata_h, backward_odata_d, sizeof(float) * 20).wait();

  float backward_odata_ref[20];
  backward_odata_ref[0] =  0;
  backward_odata_ref[1] =  20;
  backward_odata_ref[2] =  40;
  backward_odata_ref[3] =  60;
  backward_odata_ref[4] =  80;
  backward_odata_ref[5] =  100;
  backward_odata_ref[6] =  120;
  backward_odata_ref[7] =  140;
  backward_odata_ref[8] =  160;
  backward_odata_ref[9] =  180;
  backward_odata_ref[10] = 200;
  backward_odata_ref[11] = 220;
  backward_odata_ref[12] = 240;
  backward_odata_ref[13] = 260;
  backward_odata_ref[14] = 280;
  backward_odata_ref[15] = 300;
  backward_odata_ref[16] = 320;
  backward_odata_ref[17] = 340;
  backward_odata_ref[18] = 360;
  backward_odata_ref[19] = 380;

  sycl::free(forward_idata_d, q_ct1);
  sycl::free(forward_odata_d, q_ct1);
  sycl::free(backward_odata_d, q_ct1);

  dpct::fft::fft_engine::destroy(plan_bwd);

  if (!compare(backward_odata_ref, backward_odata_h, 20)) {
    std::cout << "backward_odata_h:" << std::endl;
    print_values(backward_odata_h, 20);
    std::cout << "backward_odata_ref:" << std::endl;
    print_values(backward_odata_ref, 20);
    return false;
  }
  return true;
}


#ifdef DEBUG_FFT
int main() {
#define FUNC r2cc2r_2d_outofplace
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

