// ===--- d2zz2d_3d_inplace_make_plan.dp.cpp ------------------*- C++ -*---===//
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

bool d2zz2d_3d_inplace_make_plan() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  dpct::fft::fft_engine *plan_fwd;
  plan_fwd = dpct::fft::fft_engine::create();
  double forward_idata_h[36];
  set_value(forward_idata_h, 2, 3, 5, 6);

  double* data_d;
  data_d = (double *)dpct::dpct_malloc(sizeof(double) * 36);
  dpct::dpct_memcpy(data_d, forward_idata_h, sizeof(double) * 36,
                    dpct::host_to_device);

  size_t workSize;
  plan_fwd->commit(&q_ct1, 2, 3, 5,
                   dpct::fft::fft_type::real_double_to_complex_double, nullptr);
  plan_fwd->compute<double, sycl::double2>(data_d, (sycl::double2 *)data_d,
                                           dpct::fft::fft_direction::forward);
  dev_ct1.queues_wait_and_throw();
  sycl::double2 forward_odata_h[18];
  dpct::dpct_memcpy(forward_odata_h, data_d, sizeof(double) * 36,
                    dpct::device_to_host);

  sycl::double2 forward_odata_ref[18];
  forward_odata_ref[0] = sycl::double2{435, 0};
  forward_odata_ref[1] = sycl::double2{-15, 20.6457};
  forward_odata_ref[2] = sycl::double2{-15, 4.8738};
  forward_odata_ref[3] = sycl::double2{-75, 43.3013};
  forward_odata_ref[4] = sycl::double2{0, 0};
  forward_odata_ref[5] = sycl::double2{0, 0};
  forward_odata_ref[6] = sycl::double2{-75, -43.3013};
  forward_odata_ref[7] = sycl::double2{0, 0};
  forward_odata_ref[8] = sycl::double2{0, 0};
  forward_odata_ref[9] = sycl::double2{-225, 0};
  forward_odata_ref[10] = sycl::double2{0, 0};
  forward_odata_ref[11] = sycl::double2{0, 0};
  forward_odata_ref[12] = sycl::double2{0, 0};
  forward_odata_ref[13] = sycl::double2{0, 0};
  forward_odata_ref[14] = sycl::double2{0, 0};
  forward_odata_ref[15] = sycl::double2{0, 0};
  forward_odata_ref[16] = sycl::double2{0, 0};
  forward_odata_ref[17] = sycl::double2{0, 0};

  dpct::fft::fft_engine::destroy(plan_fwd);

  if (!compare(forward_odata_ref, forward_odata_h, 18)) {
    std::cout << "forward_odata_h:" << std::endl;
    print_values(forward_odata_h, 18);
    std::cout << "forward_odata_ref:" << std::endl;
    print_values(forward_odata_ref, 18);

    dpct::dpct_free(data_d);

    return false;
  }

  dpct::fft::fft_engine *plan_bwd;
  plan_bwd = dpct::fft::fft_engine::create();
  plan_bwd->commit(&q_ct1, 2, 3, 5,
                   dpct::fft::fft_type::complex_double_to_real_double, nullptr);
  plan_bwd->compute<sycl::double2, double>((sycl::double2 *)data_d, data_d,
                                           dpct::fft::fft_direction::backward);
  dev_ct1.queues_wait_and_throw();
  double backward_odata_h[36];
  dpct::dpct_memcpy(backward_odata_h, data_d, sizeof(double) * 36,
                    dpct::device_to_host);

  double backward_odata_ref[36];
  backward_odata_ref[0]  = 0;
  backward_odata_ref[1]  = 30;
  backward_odata_ref[2]  = 60;
  backward_odata_ref[3]  = 90;
  backward_odata_ref[4]  = 120;
  backward_odata_ref[5]  = 4.8738;
  backward_odata_ref[6]  = 150;
  backward_odata_ref[7]  = 180;
  backward_odata_ref[8]  = 210;
  backward_odata_ref[9]  = 240;
  backward_odata_ref[10] = 270;
  backward_odata_ref[11] = 4.8738;
  backward_odata_ref[12] = 300;
  backward_odata_ref[13] = 330;
  backward_odata_ref[14] = 360;
  backward_odata_ref[15] = 390;
  backward_odata_ref[16] = 420;
  backward_odata_ref[17] = 4.8738;
  backward_odata_ref[18] = 450;
  backward_odata_ref[19] = 480;
  backward_odata_ref[20] = 510;
  backward_odata_ref[21] = 540;
  backward_odata_ref[22] = 570;
  backward_odata_ref[23] = 4.8738;
  backward_odata_ref[24] = 600;
  backward_odata_ref[25] = 630;
  backward_odata_ref[26] = 660;
  backward_odata_ref[27] = 690;
  backward_odata_ref[28] = 720;
  backward_odata_ref[29] = 4.8738;
  backward_odata_ref[30] = 750;
  backward_odata_ref[31] = 780;
  backward_odata_ref[32] = 810;
  backward_odata_ref[33] = 840;
  backward_odata_ref[34] = 870;
  backward_odata_ref[35] = 4.8738;

  dpct::dpct_free(data_d);
  dpct::fft::fft_engine::destroy(plan_bwd);

  std::vector<int> indices = {0, 1, 2, 3, 4,
                              6, 7, 8, 9, 10,
                              12, 13, 14, 15 ,16,
                              18, 19, 20, 21, 22,
                              24, 25, 26, 27, 28,
                              30, 31, 32, 33, 34};
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
#define FUNC d2zz2d_3d_inplace_make_plan
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

