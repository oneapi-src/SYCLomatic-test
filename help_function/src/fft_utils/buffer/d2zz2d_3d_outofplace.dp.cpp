// ===--- d2zz2d_3d_outofplace.dp.cpp -------------------------*- C++ -*---===//
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

bool d2zz2d_3d_outofplace() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  dpct::fft::fft_engine *plan_fwd;
  double forward_idata_h[2][3][5];
  set_value((double*)forward_idata_h, 30);

  double* forward_idata_d;
  sycl::double2 *forward_odata_d;
  double* backward_odata_d;
  forward_idata_d = (double *)dpct::dpct_malloc(sizeof(double) * 30);
  forward_odata_d = (sycl::double2 *)dpct::dpct_malloc(sizeof(sycl::double2) *
                                                       (5 / 2 + 1) * 2 * 3);
  backward_odata_d = (double *)dpct::dpct_malloc(sizeof(double) * 30);
  dpct::dpct_memcpy(forward_idata_d, forward_idata_h, sizeof(double) * 30,
                    dpct::host_to_device);

  plan_fwd = dpct::fft::fft_engine::create(
      &q_ct1, 2, 3, 5, dpct::fft::fft_type::real_double_to_complex_double);
  plan_fwd->compute<double, sycl::double2>(forward_idata_d, forward_odata_d,
                                           dpct::fft::fft_direction::forward);
  dev_ct1.queues_wait_and_throw();
  sycl::double2 forward_odata_h[18];
  dpct::dpct_memcpy(forward_odata_h, forward_odata_d,
                    sizeof(sycl::double2) * (5 / 2 + 1) * 2 * 3,
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

    dpct::dpct_free(forward_idata_d);
    dpct::dpct_free(forward_odata_d);
    dpct::dpct_free(backward_odata_d);

    return false;
  }

  dpct::fft::fft_engine *plan_bwd;
  plan_bwd = dpct::fft::fft_engine::create(
      &q_ct1, 2, 3, 5, dpct::fft::fft_type::complex_double_to_real_double);
  plan_bwd->compute<sycl::double2, double>(forward_odata_d, backward_odata_d,
                                           dpct::fft::fft_direction::backward);
  dev_ct1.queues_wait_and_throw();
  double backward_odata_h[30];
  dpct::dpct_memcpy(backward_odata_h, backward_odata_d, sizeof(double) * 30,
                    dpct::device_to_host);

  double backward_odata_ref[30];
  backward_odata_ref[0] =  0;
  backward_odata_ref[1] =  30;
  backward_odata_ref[2] =  60;
  backward_odata_ref[3] =  90;
  backward_odata_ref[4] =  120;
  backward_odata_ref[5] =  150;
  backward_odata_ref[6] =  180;
  backward_odata_ref[7] =  210;
  backward_odata_ref[8] =  240;
  backward_odata_ref[9] =  270;
  backward_odata_ref[10] = 300;
  backward_odata_ref[11] = 330;
  backward_odata_ref[12] = 360;
  backward_odata_ref[13] = 390;
  backward_odata_ref[14] = 420;
  backward_odata_ref[15] = 450;
  backward_odata_ref[16] = 480;
  backward_odata_ref[17] = 510;
  backward_odata_ref[18] = 540;
  backward_odata_ref[19] = 570;
  backward_odata_ref[20] = 600;
  backward_odata_ref[21] = 630;
  backward_odata_ref[22] = 660;
  backward_odata_ref[23] = 690;
  backward_odata_ref[24] = 720;
  backward_odata_ref[25] = 750;
  backward_odata_ref[26] = 780;
  backward_odata_ref[27] = 810;
  backward_odata_ref[28] = 840;
  backward_odata_ref[29] = 870;

  dpct::dpct_free(forward_idata_d);
  dpct::dpct_free(forward_odata_d);
  dpct::dpct_free(backward_odata_d);

  dpct::fft::fft_engine::destroy(plan_bwd);

  if (!compare(backward_odata_ref, backward_odata_h, 30)) {
    std::cout << "backward_odata_h:" << std::endl;
    print_values(backward_odata_h, 30);
    std::cout << "backward_odata_ref:" << std::endl;
    print_values(backward_odata_ref, 30);
    return false;
  }
  return true;
}


#ifdef DEBUG_FFT
int main() {
#define FUNC d2zz2d_3d_outofplace
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

