// ===--- r2cc2r_many_3d_outofplace_basic.dp.cpp --------------*- C++ -*---===//
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

bool r2cc2r_many_3d_outofplace_basic() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  dpct::fft::fft_engine *plan_fwd;
  float forward_idata_h[4/*n0*/ * 2/*n1*/ * 3/*n2*/ * 2/*batch*/];
  set_value(forward_idata_h, 24);
  set_value(forward_idata_h + 24, 24);

  float* forward_idata_d;
  sycl::float2 *forward_odata_d;
  float* backward_odata_d;
  forward_idata_d =
      (float *)sycl::malloc_device(2 * sizeof(float) * 4 * 2 * 3, q_ct1);
  forward_odata_d = (sycl::float2 *)sycl::malloc_device(
      2 * sizeof(sycl::float2) * 4 * 2 * (3 / 2 + 1), q_ct1);
  backward_odata_d =
      (float *)sycl::malloc_device(2 * sizeof(float) * 4 * 2 * 3, q_ct1);
  q_ct1.memcpy(forward_idata_d, forward_idata_h, 2 * sizeof(float) * 4 * 2 * 3)
      .wait();

  int n[3] = {4, 2, 3};
  plan_fwd = dpct::fft::fft_engine::create(
      &q_ct1, 3, n, nullptr, 0, 0, nullptr, 0, 0,
      dpct::fft::fft_type::real_float_to_complex_float, 2);
  plan_fwd->compute<float, sycl::float2>(forward_idata_d, forward_odata_d,
                                         dpct::fft::fft_direction::forward);
  dev_ct1.queues_wait_and_throw();
  sycl::float2 forward_odata_h[32];
  q_ct1
      .memcpy(forward_odata_h, forward_odata_d,
              2 * sizeof(sycl::float2) * 4 * 2 * (3 / 2 + 1))
      .wait();

  sycl::float2 forward_odata_ref[32];
  forward_odata_ref[0] = sycl::float2{276, 0};
  forward_odata_ref[1] = sycl::float2{-12, 6.9282};
  forward_odata_ref[2] = sycl::float2{-36, 0};
  forward_odata_ref[3] = sycl::float2{0, 0};
  forward_odata_ref[4] = sycl::float2{-72, 72};
  forward_odata_ref[5] = sycl::float2{0, 0};
  forward_odata_ref[6] = sycl::float2{0, 0};
  forward_odata_ref[7] = sycl::float2{0, 0};
  forward_odata_ref[8] = sycl::float2{-72, 0};
  forward_odata_ref[9] = sycl::float2{0, 0};
  forward_odata_ref[10] = sycl::float2{0, 0};
  forward_odata_ref[11] = sycl::float2{0, 0};
  forward_odata_ref[12] = sycl::float2{-72, -72};
  forward_odata_ref[13] = sycl::float2{0, 0};
  forward_odata_ref[14] = sycl::float2{0, 0};
  forward_odata_ref[15] = sycl::float2{0, 0};
  forward_odata_ref[16] = sycl::float2{276, 0};
  forward_odata_ref[17] = sycl::float2{-12, 6.9282};
  forward_odata_ref[18] = sycl::float2{-36, 0};
  forward_odata_ref[19] = sycl::float2{0, 0};
  forward_odata_ref[20] = sycl::float2{-72, 72};
  forward_odata_ref[21] = sycl::float2{0, 0};
  forward_odata_ref[22] = sycl::float2{0, 0};
  forward_odata_ref[23] = sycl::float2{0, 0};
  forward_odata_ref[24] = sycl::float2{-72, 0};
  forward_odata_ref[25] = sycl::float2{0, 0};
  forward_odata_ref[26] = sycl::float2{0, 0};
  forward_odata_ref[27] = sycl::float2{0, 0};
  forward_odata_ref[28] = sycl::float2{-72, -72};
  forward_odata_ref[29] = sycl::float2{0, 0};
  forward_odata_ref[30] = sycl::float2{0, 0};
  forward_odata_ref[31] = sycl::float2{0, 0};

  dpct::fft::fft_engine::destroy(plan_fwd);

  if (!compare(forward_odata_ref, forward_odata_h, 32)) {
    std::cout << "forward_odata_h:" << std::endl;
    print_values(forward_odata_h, 32);
    std::cout << "forward_odata_ref:" << std::endl;
    print_values(forward_odata_ref, 32);

    sycl::free(forward_idata_d, q_ct1);
    sycl::free(forward_odata_d, q_ct1);
    sycl::free(backward_odata_d, q_ct1);

    return false;
  }

  dpct::fft::fft_engine *plan_bwd;
  plan_bwd = dpct::fft::fft_engine::create(
      &q_ct1, 3, n, nullptr, 0, 0, nullptr, 0, 0,
      dpct::fft::fft_type::complex_float_to_real_float, 2);
  plan_bwd->compute<sycl::float2, float>(forward_odata_d, backward_odata_d,
                                         dpct::fft::fft_direction::backward);
  dev_ct1.queues_wait_and_throw();
  float backward_odata_h[48];
  q_ct1.memcpy(backward_odata_h, backward_odata_d, sizeof(float) * 48).wait();

  float backward_odata_ref[48];
  backward_odata_ref[0] =  0;
  backward_odata_ref[1] =  24;
  backward_odata_ref[2] =  48;
  backward_odata_ref[3] =  72;
  backward_odata_ref[4] =  96;
  backward_odata_ref[5] =  120;
  backward_odata_ref[6] =  144;
  backward_odata_ref[7] =  168;
  backward_odata_ref[8] =  192;
  backward_odata_ref[9] =  216;
  backward_odata_ref[10] = 240;
  backward_odata_ref[11] = 264;
  backward_odata_ref[12] = 288;
  backward_odata_ref[13] = 312;
  backward_odata_ref[14] = 336;
  backward_odata_ref[15] = 360;
  backward_odata_ref[16] = 384;
  backward_odata_ref[17] = 408;
  backward_odata_ref[18] = 432;
  backward_odata_ref[19] = 456;
  backward_odata_ref[20] = 480;
  backward_odata_ref[21] = 504;
  backward_odata_ref[22] = 528;
  backward_odata_ref[23] = 552;
  backward_odata_ref[24] = 0;
  backward_odata_ref[25] = 24;
  backward_odata_ref[26] = 48;
  backward_odata_ref[27] = 72;
  backward_odata_ref[28] = 96;
  backward_odata_ref[29] = 120;
  backward_odata_ref[30] = 144;
  backward_odata_ref[31] = 168;
  backward_odata_ref[32] = 192;
  backward_odata_ref[33] = 216;
  backward_odata_ref[34] = 240;
  backward_odata_ref[35] = 264;
  backward_odata_ref[36] = 288;
  backward_odata_ref[37] = 312;
  backward_odata_ref[38] = 336;
  backward_odata_ref[39] = 360;
  backward_odata_ref[40] = 384;
  backward_odata_ref[41] = 408;
  backward_odata_ref[42] = 432;
  backward_odata_ref[43] = 456;
  backward_odata_ref[44] = 480;
  backward_odata_ref[45] = 504;
  backward_odata_ref[46] = 528;
  backward_odata_ref[47] = 552;

  sycl::free(forward_idata_d, q_ct1);
  sycl::free(forward_odata_d, q_ct1);
  sycl::free(backward_odata_d, q_ct1);

  dpct::fft::fft_engine::destroy(plan_bwd);

  if (!compare(backward_odata_ref, backward_odata_h, 48)) {
    std::cout << "backward_odata_h:" << std::endl;
    print_values(backward_odata_h, 48);
    std::cout << "backward_odata_ref:" << std::endl;
    print_values(backward_odata_ref, 48);
    return false;
  }
  return true;
}


#ifdef DEBUG_FFT
int main() {
#define FUNC r2cc2r_many_3d_outofplace_basic
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

