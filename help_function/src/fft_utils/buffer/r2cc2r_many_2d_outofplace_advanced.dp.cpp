// ===--- r2cc2r_many_2d_outofplace_advanced.dp.cpp -----------*- C++ -*---===//
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
#include <dpct/lib_common_utils.hpp>

// forward
// input
// +---+---+---+---+---+---+     -+
// | r | 0 | r | 0 | r | 0 |      |
// +---+---+---+---+---+---+      |
// | r | 0 | r | 0 | r | 0 |      batch0
// +---+---+---+---+---+---+     -+
// | r | 0 | r | 0 | r | 0 |      |
// +---+---+---+---+---+---+      |
// | r | 0 | r | 0 | r | 0 |      batch1
// +---+---+---+---+---+---+     -+
// |__________n2___________|
// |________nembed2________|
// output
// +---+---+---+---+ -+
// |   c   |   c   |  |
// +---+---+---+---+  batch0
// |   c   |   c   |  |
// +---+---+---+---+ -+
// |   c   |   c   |  |
// +---+---+---+---+  batch1
// |   c   |   c   |  |
// +---+---+---+---+ -+
// |______n2_______|
// |____nembed2____|
bool r2cc2r_many_2d_outofplace_advanced() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  dpct::fft::fft_engine *plan_fwd;
  plan_fwd = dpct::fft::fft_engine::create();
  float forward_idata_h[24];
  std::memset(forward_idata_h, 0, sizeof(float) * 24);
  forward_idata_h[0]  = 0;
  forward_idata_h[2]  = 1;
  forward_idata_h[4]  = 2;
  forward_idata_h[6]  = 3;
  forward_idata_h[8]  = 4;
  forward_idata_h[10] = 5;
  forward_idata_h[12] = 0;
  forward_idata_h[14] = 1;
  forward_idata_h[16] = 2;
  forward_idata_h[18] = 3;
  forward_idata_h[20] = 4;
  forward_idata_h[22] = 5;

  float* forward_idata_d;
  sycl::float2 *forward_odata_d;
  float* backward_odata_d;
  forward_idata_d = (float *)dpct::dpct_malloc(sizeof(float) * 24);
  forward_odata_d = (sycl::float2 *)dpct::dpct_malloc(sizeof(sycl::float2) * 8);
  backward_odata_d = (float *)dpct::dpct_malloc(sizeof(float) * 24);
  dpct::dpct_memcpy(forward_idata_d, forward_idata_h, sizeof(float) * 24,
                    dpct::host_to_device);

  long long int n[2] = {2, 3};
  long long int inembed[2] = {2, 3};
  long long int onembed[2] = {2, 2};
  size_t workSize;
  plan_fwd->commit(&q_ct1, 2, n, inembed, 2, 12,
                   dpct::library_data_t::real_float, onembed, 1, 4,
                   dpct::library_data_t::complex_float, 2, nullptr);
  plan_fwd->compute<void, void>(forward_idata_d, forward_odata_d,
                                dpct::fft::fft_direction::forward);
  dev_ct1.queues_wait_and_throw();
  sycl::float2 forward_odata_h[8];
  dpct::dpct_memcpy(forward_odata_h, forward_odata_d, sizeof(sycl::float2) * 8,
                    dpct::device_to_host);

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

    dpct::dpct_free(forward_idata_d);
    dpct::dpct_free(forward_odata_d);
    dpct::dpct_free(backward_odata_d);

    return false;
  }

  dpct::fft::fft_engine *plan_bwd;
  plan_bwd = dpct::fft::fft_engine::create();
  plan_bwd->commit(&q_ct1, 2, n, onembed, 1, 4,
                   dpct::library_data_t::complex_float, inembed, 2, 12,
                   dpct::library_data_t::real_float, 2, nullptr);
  plan_bwd->compute<void, void>(forward_odata_d, backward_odata_d,
                                dpct::fft::fft_direction::backward);
  dev_ct1.queues_wait_and_throw();
  float backward_odata_h[24];
  dpct::dpct_memcpy(backward_odata_h, backward_odata_d, sizeof(float) * 24,
                    dpct::device_to_host);

  float backward_odata_ref[24];
  backward_odata_ref[0]  = 0;
  backward_odata_ref[2]  = 6;
  backward_odata_ref[4]  = 12;
  backward_odata_ref[6]  = 18;
  backward_odata_ref[8]  = 24;
  backward_odata_ref[10] = 30;
  backward_odata_ref[12] = 0;
  backward_odata_ref[14] = 6;
  backward_odata_ref[16] = 12;
  backward_odata_ref[18] = 18;
  backward_odata_ref[20] = 24;
  backward_odata_ref[22] = 30;

  dpct::dpct_free(forward_idata_d);
  dpct::dpct_free(forward_odata_d);
  dpct::dpct_free(backward_odata_d);

  dpct::fft::fft_engine::destroy(plan_bwd);

  std::vector<int> indices = {0, 2, 4,
                              6, 8, 10,
                              12, 14, 16,
                              18, 20, 22};
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
#define FUNC r2cc2r_many_2d_outofplace_advanced
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

