// ===--- d2zz2d_many_3d_outofplace_basic.dp.cpp --------------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===---------------------------------------------------------------------===//

#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <oneapi/mkl.hpp>
#include <dpct/fft_utils.hpp>
#include "common.h"
#include <cstring>
#include <iostream>

bool d2zz2d_many_3d_outofplace_basic() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  dpct::fft::fft_engine *plan_fwd;
  double forward_idata_h[4/*n0*/ * 2/*n1*/ * 3/*n2*/ * 2/*batch*/];
  set_value(forward_idata_h, 24);
  set_value(forward_idata_h + 24, 24);

  double* forward_idata_d;
  sycl::double2 *forward_odata_d;
  double* backward_odata_d;
  forward_idata_d = (double *)dpct::dpct_malloc(2 * sizeof(double) * 4 * 2 * 3);
  forward_odata_d = (sycl::double2 *)dpct::dpct_malloc(
      2 * sizeof(sycl::double2) * 4 * 2 * (3 / 2 + 1));
  backward_odata_d =
      (double *)dpct::dpct_malloc(2 * sizeof(double) * 4 * 2 * 3);
  dpct::dpct_memcpy(forward_idata_d, forward_idata_h,
                    2 * sizeof(double) * 4 * 2 * 3, dpct::host_to_device);

  int n[3] = {4, 2, 3};
  plan_fwd = dpct::fft::fft_engine::create(
      &q_ct1, 3, n, nullptr, 0, 0, nullptr, 0, 0,
      dpct::fft::fft_type::real_double_to_complex_double, 2);
  plan_fwd->compute<double, sycl::double2>(forward_idata_d, forward_odata_d,
                                           dpct::fft::fft_direction::forward);
  dev_ct1.queues_wait_and_throw();
  sycl::double2 forward_odata_h[32];
  dpct::dpct_memcpy(forward_odata_h, forward_odata_d,
                    2 * sizeof(sycl::double2) * 4 * 2 * (3 / 2 + 1),
                    dpct::device_to_host);

  sycl::double2 forward_odata_ref[32];
  forward_odata_ref[0] = sycl::double2{276, 0};
  forward_odata_ref[1] = sycl::double2{-12, 6.9282};
  forward_odata_ref[2] = sycl::double2{-36, 0};
  forward_odata_ref[3] = sycl::double2{0, 0};
  forward_odata_ref[4] = sycl::double2{-72, 72};
  forward_odata_ref[5] = sycl::double2{0, 0};
  forward_odata_ref[6] = sycl::double2{0, 0};
  forward_odata_ref[7] = sycl::double2{0, 0};
  forward_odata_ref[8] = sycl::double2{-72, 0};
  forward_odata_ref[9] = sycl::double2{0, 0};
  forward_odata_ref[10] = sycl::double2{0, 0};
  forward_odata_ref[11] = sycl::double2{0, 0};
  forward_odata_ref[12] = sycl::double2{-72, -72};
  forward_odata_ref[13] = sycl::double2{0, 0};
  forward_odata_ref[14] = sycl::double2{0, 0};
  forward_odata_ref[15] = sycl::double2{0, 0};
  forward_odata_ref[16] = sycl::double2{276, 0};
  forward_odata_ref[17] = sycl::double2{-12, 6.9282};
  forward_odata_ref[18] = sycl::double2{-36, 0};
  forward_odata_ref[19] = sycl::double2{0, 0};
  forward_odata_ref[20] = sycl::double2{-72, 72};
  forward_odata_ref[21] = sycl::double2{0, 0};
  forward_odata_ref[22] = sycl::double2{0, 0};
  forward_odata_ref[23] = sycl::double2{0, 0};
  forward_odata_ref[24] = sycl::double2{-72, 0};
  forward_odata_ref[25] = sycl::double2{0, 0};
  forward_odata_ref[26] = sycl::double2{0, 0};
  forward_odata_ref[27] = sycl::double2{0, 0};
  forward_odata_ref[28] = sycl::double2{-72, -72};
  forward_odata_ref[29] = sycl::double2{0, 0};
  forward_odata_ref[30] = sycl::double2{0, 0};
  forward_odata_ref[31] = sycl::double2{0, 0};

  dpct::fft::fft_engine::destroy(plan_fwd);

  if (!compare(forward_odata_ref, forward_odata_h, 32)) {
    std::cout << "forward_odata_h:" << std::endl;
    print_values(forward_odata_h, 32);
    std::cout << "forward_odata_ref:" << std::endl;
    print_values(forward_odata_ref, 32);

    dpct::dpct_free(forward_idata_d);
    dpct::dpct_free(forward_odata_d);
    dpct::dpct_free(backward_odata_d);

    return false;
  }

  dpct::fft::fft_engine *plan_bwd;
  plan_bwd = dpct::fft::fft_engine::create(
      &q_ct1, 3, n, nullptr, 0, 0, nullptr, 0, 0,
      dpct::fft::fft_type::complex_double_to_real_double, 2);
  plan_bwd->compute<sycl::double2, double>(forward_odata_d, backward_odata_d,
                                           dpct::fft::fft_direction::backward);
  dev_ct1.queues_wait_and_throw();
  double backward_odata_h[48];
  dpct::dpct_memcpy(backward_odata_h, backward_odata_d, sizeof(double) * 48,
                    dpct::device_to_host);

  double backward_odata_ref[48];
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

  dpct::dpct_free(forward_idata_d);
  dpct::dpct_free(forward_odata_d);
  dpct::dpct_free(backward_odata_d);

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
#define FUNC d2zz2d_many_3d_outofplace_basic
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

