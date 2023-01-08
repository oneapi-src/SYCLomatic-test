// ===--- d2zz2d_2d_inplace.dp.cpp ----------------------------*- C++ -*---===//
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

bool d2zz2d_2d_inplace() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  dpct::fft::fft_engine *plan_fwd;
  double forward_idata_h[24];
  set_value(forward_idata_h, 4, 5, 6);

  double* data_d;
  data_d = sycl::malloc_device<double>(24, q_ct1);
  q_ct1.memcpy(data_d, forward_idata_h, sizeof(double) * 24).wait();

  plan_fwd = dpct::fft::fft_engine::create(
      &q_ct1, 4, 5, dpct::fft::fft_type::real_double_to_complex_double);
  plan_fwd->compute<double, sycl::double2>(data_d, (sycl::double2 *)data_d,
                                           dpct::fft::fft_direction::forward);
  dev_ct1.queues_wait_and_throw();
  sycl::double2 forward_odata_h[12];
  q_ct1.memcpy(forward_odata_h, data_d, sizeof(double) * 24).wait();

  sycl::double2 forward_odata_ref[12];
  forward_odata_ref[0] = sycl::double2{190, 0};
  forward_odata_ref[1] = sycl::double2{-10, 13.7638};
  forward_odata_ref[2] = sycl::double2{-10, 3.2492};
  forward_odata_ref[3] = sycl::double2{-50, 50};
  forward_odata_ref[4] = sycl::double2{0, 0};
  forward_odata_ref[5] = sycl::double2{0, 0};
  forward_odata_ref[6] = sycl::double2{-50, 0};
  forward_odata_ref[7] = sycl::double2{0, 0};
  forward_odata_ref[8] = sycl::double2{0, 0};
  forward_odata_ref[9] = sycl::double2{-50, -50};
  forward_odata_ref[10] = sycl::double2{0, 0};
  forward_odata_ref[11] = sycl::double2{0, 0};

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
  plan_bwd = dpct::fft::fft_engine::create(
      &q_ct1, 4, 5, dpct::fft::fft_type::complex_double_to_real_double);
  plan_bwd->compute<sycl::double2, double>((sycl::double2 *)data_d, data_d,
                                           dpct::fft::fft_direction::backward);
  dev_ct1.queues_wait_and_throw();
  double backward_odata_h[24];
  q_ct1.memcpy(backward_odata_h, data_d, sizeof(double) * 24).wait();

  double backward_odata_ref[24];
  backward_odata_ref[0] =  0;
  backward_odata_ref[1] =  20;
  backward_odata_ref[2] =  40;
  backward_odata_ref[3] =  60;
  backward_odata_ref[4] =  80;
  backward_odata_ref[5] =  3.2492;
  backward_odata_ref[6] =  100;
  backward_odata_ref[7] =  120;
  backward_odata_ref[8] =  140;
  backward_odata_ref[9] =  160;
  backward_odata_ref[10] = 180;
  backward_odata_ref[11] = 3.2492;
  backward_odata_ref[12] = 200;
  backward_odata_ref[13] = 220;
  backward_odata_ref[14] = 240;
  backward_odata_ref[15] = 260;
  backward_odata_ref[16] = 280;
  backward_odata_ref[17] = 3.2492;
  backward_odata_ref[18] = 300;
  backward_odata_ref[19] = 320;
  backward_odata_ref[20] = 340;
  backward_odata_ref[21] = 360;
  backward_odata_ref[22] = 380;
  backward_odata_ref[23] = 3.2492;

  sycl::free(data_d, q_ct1);
  dpct::fft::fft_engine::destroy(plan_bwd);

  std::vector<int> indices = {0, 1, 2, 3, 4,
                              6, 7, 8, 9, 10,
                              12, 13, 14, 15 ,16,
                              18, 19, 20, 21, 22};
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
#define FUNC d2zz2d_2d_inplace
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

