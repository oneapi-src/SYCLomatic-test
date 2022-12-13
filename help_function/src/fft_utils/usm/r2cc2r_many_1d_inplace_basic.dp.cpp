// ===--- r2cc2r_many_1d_inplace_basic.dp.cpp -----------------*- C++ -*---===//
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

bool r2cc2r_many_1d_inplace_basic() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  dpct::fft::fft_engine *plan_fwd;
  plan_fwd = dpct::fft::fft_engine::create();
  float forward_idata_h[24];
  set_value(forward_idata_h, 10);
  set_value(forward_idata_h + 12, 10);

  float* data_d;
  data_d = sycl::malloc_device<float>(24, q_ct1);
  q_ct1.memcpy(data_d, forward_idata_h, sizeof(float) * 24).wait();

  int n[1] = {10};
  size_t workSize;
  plan_fwd->commit(&q_ct1, 1, n, nullptr, 0, 0, nullptr, 0, 0,
                   dpct::fft::fft_type::real_float_to_complex_float, 2,
                   nullptr);
  plan_fwd->compute<float, sycl::float2>(data_d, (sycl::float2 *)data_d,
                                         dpct::fft::fft_direction::forward);
  dev_ct1.queues_wait_and_throw();
  sycl::float2 forward_odata_h[12];
  q_ct1.memcpy(forward_odata_h, data_d, sizeof(float) * 24).wait();

  sycl::float2 forward_odata_ref[12];
  forward_odata_ref[0] = sycl::float2{45, 0};
  forward_odata_ref[1] = sycl::float2{-5, 15.3884};
  forward_odata_ref[2] = sycl::float2{-5, 6.88191};
  forward_odata_ref[3] = sycl::float2{-5, 3.63271};
  forward_odata_ref[4] = sycl::float2{-5, 1.6246};
  forward_odata_ref[5] = sycl::float2{-5, 0};
  forward_odata_ref[6] = sycl::float2{45, 0};
  forward_odata_ref[7] = sycl::float2{-5, 15.3884};
  forward_odata_ref[8] = sycl::float2{-5, 6.88191};
  forward_odata_ref[9] = sycl::float2{-5, 3.63271};
  forward_odata_ref[10] = sycl::float2{-5, 1.6246};
  forward_odata_ref[11] = sycl::float2{-5, 0};

  dpct::fft::fft_engine::destroy(plan_fwd);

  if (!compare(forward_odata_ref, forward_odata_h, 12)) {
    std::cout << "forward_odata_h:" << std::endl;
    print_values(forward_odata_h, 12);
    std::cout << "forward_odata_ref:" << std::endl;
    print_values(forward_odata_ref, 12);

    sycl::free(data_d, q_ct1);

    return false;
  }

  dpct::fft::fft_engine *plan_bwd;
  plan_bwd = dpct::fft::fft_engine::create();
  plan_bwd->commit(&q_ct1, 1, n, nullptr, 0, 0, nullptr, 0, 0,
                   dpct::fft::fft_type::complex_float_to_real_float, 2,
                   nullptr);
  plan_bwd->compute<sycl::float2, float>((sycl::float2 *)data_d, data_d,
                                         dpct::fft::fft_direction::backward);
  dev_ct1.queues_wait_and_throw();
  float backward_odata_h[24];
  q_ct1.memcpy(backward_odata_h, data_d, sizeof(float) * 24).wait();

  float backward_odata_ref[24];
  backward_odata_ref[0] =  0;
  backward_odata_ref[1] =  10;
  backward_odata_ref[2] =  20;
  backward_odata_ref[3] =  30;
  backward_odata_ref[4] =  40;
  backward_odata_ref[5] =  50;
  backward_odata_ref[6] =  60;
  backward_odata_ref[7] =  70;
  backward_odata_ref[8] =  80;
  backward_odata_ref[9] =  90;
  backward_odata_ref[10] = -5;
  backward_odata_ref[11] = 0;
  backward_odata_ref[12] = 0;
  backward_odata_ref[13] = 10;
  backward_odata_ref[14] = 20;
  backward_odata_ref[15] = 30;
  backward_odata_ref[16] = 40;
  backward_odata_ref[17] = 50;
  backward_odata_ref[18] = 60;
  backward_odata_ref[19] = 70;
  backward_odata_ref[20] = 80;
  backward_odata_ref[21] = 90;
  backward_odata_ref[22] = -5;
  backward_odata_ref[23] = 0;

  sycl::free(data_d, q_ct1);
  dpct::fft::fft_engine::destroy(plan_bwd);

  std::vector<int> indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                              12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
  if (!compare(backward_odata_ref, backward_odata_h, indices)) {
    std::cout << "backward_odata_h:" << std::endl;
    print_values(backward_odata_h, indices);
    std::cout << "backward_odata_ref:" << std::endl;
    print_values(backward_odata_ref, indices);
    return false;
  }
  return true;
}

#ifdef DEBUG_FFT
int main() {
#define FUNC r2cc2r_many_1d_inplace_basic
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

