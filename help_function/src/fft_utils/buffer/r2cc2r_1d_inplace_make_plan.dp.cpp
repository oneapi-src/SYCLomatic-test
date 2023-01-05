// ===--- r2cc2r_1d_inplace_make_plan.dp.cpp ------------------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===---------------------------------------------------------------------===//

#define DPCT_USM_LEVEL_NONE
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <oneapi/mkl.hpp>
#include <dpct/fft_utils.hpp>
#include "common.h"
#include <cstring>
#include <iostream>

bool r2cc2r_1d_inplace_make_plan() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  dpct::fft::fft_engine *plan_fwd;
  plan_fwd = dpct::fft::fft_engine::create();
  float forward_idata_h[16];
  set_value(forward_idata_h, 7);
  set_value(forward_idata_h + 8, 7);

  float* data_d;
  data_d = (float *)dpct::dpct_malloc(sizeof(float) * 16);
  dpct::dpct_memcpy(data_d, forward_idata_h, sizeof(float) * 16,
                    dpct::host_to_device);

  size_t workSize;
  plan_fwd->commit(&q_ct1, 7, dpct::fft::fft_type::real_float_to_complex_float,
                   2, nullptr);
  plan_fwd->compute<float, sycl::float2>(data_d, (sycl::float2 *)data_d,
                                         dpct::fft::fft_direction::forward);
  dev_ct1.queues_wait_and_throw();
  sycl::float2 forward_odata_h[8];
  dpct::dpct_memcpy(forward_odata_h, data_d, sizeof(float) * 16,
                    dpct::device_to_host);

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

    dpct::dpct_free(data_d);

    return false;
  }

  dpct::fft::fft_engine *plan_bwd;
  plan_bwd = dpct::fft::fft_engine::create();
  plan_bwd->commit(&q_ct1, 7, dpct::fft::fft_type::complex_float_to_real_float,
                   2, nullptr);
  plan_bwd->compute<sycl::float2, float>((sycl::float2 *)data_d, data_d,
                                         dpct::fft::fft_direction::backward);
  dev_ct1.queues_wait_and_throw();
  float backward_odata_h[16];
  dpct::dpct_memcpy(backward_odata_h, data_d, sizeof(float) * 16,
                    dpct::device_to_host);

  float backward_odata_ref[16];
  backward_odata_ref[0] = 0;
  backward_odata_ref[1] = 7;
  backward_odata_ref[2] = 14;
  backward_odata_ref[3] = 21;
  backward_odata_ref[4] = 28;
  backward_odata_ref[5] = 35;
  backward_odata_ref[6] = 42;
  backward_odata_ref[7] = 0.798852;
  backward_odata_ref[8] = 0;
  backward_odata_ref[9] = 7;
  backward_odata_ref[10] = 14;
  backward_odata_ref[11] = 21;
  backward_odata_ref[12] = 28;
  backward_odata_ref[13] = 35;
  backward_odata_ref[14] = 42;
  backward_odata_ref[15] = 0.798852;

  dpct::dpct_free(data_d);
  dpct::fft::fft_engine::destroy(plan_bwd);

  std::vector<int> indices = {0, 1, 2,  3,  4,  5,  6,
                              8, 9, 10, 11, 12, 13, 14};
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
#define FUNC r2cc2r_1d_inplace_make_plan
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

