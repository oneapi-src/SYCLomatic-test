// ===--- d2zz2d_1d_outofplace_make_plan.dp.cpp ---------------*- C++ -*---===//
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

bool d2zz2d_1d_outofplace_make_plan() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  dpct::fft::fft_engine *plan_fwd;
  plan_fwd = dpct::fft::fft_engine::create();
  double forward_idata_h[14];
  set_value(forward_idata_h, 7);
  set_value(forward_idata_h + 7, 7);

  double* forward_idata_d;
  sycl::double2 *forward_odata_d;
  double* backward_odata_d;
  forward_idata_d = (double *)dpct::dpct_malloc(2 * sizeof(double) * 7);
  forward_odata_d = (sycl::double2 *)dpct::dpct_malloc(
      2 * sizeof(sycl::double2) * (7 / 2 + 1));
  backward_odata_d = (double *)dpct::dpct_malloc(2 * sizeof(double) * 7);
  dpct::dpct_memcpy(forward_idata_d, forward_idata_h, 2 * sizeof(double) * 7,
                    dpct::host_to_device);

  size_t workSize;
  plan_fwd->commit(&q_ct1, 7,
                   dpct::fft::fft_type::real_double_to_complex_double, 2,
                   nullptr);
  plan_fwd->compute<double, sycl::double2>(forward_idata_d, forward_odata_d,
                                           dpct::fft::fft_direction::forward);
  dev_ct1.queues_wait_and_throw();
  sycl::double2 forward_odata_h[8];
  dpct::dpct_memcpy(forward_odata_h, forward_odata_d,
                    2 * sizeof(sycl::double2) * (7 / 2 + 1),
                    dpct::device_to_host);

  sycl::double2 forward_odata_ref[8];
  forward_odata_ref[0] = sycl::double2{21, 0};
  forward_odata_ref[1] = sycl::double2{-3.5, 7.26783};
  forward_odata_ref[2] = sycl::double2{-3.5, 2.79116};
  forward_odata_ref[3] = sycl::double2{-3.5, 0.798852};
  forward_odata_ref[4] = sycl::double2{21, 0};
  forward_odata_ref[5] = sycl::double2{-3.5, 7.26783};
  forward_odata_ref[6] = sycl::double2{-3.5, 2.79116};
  forward_odata_ref[7] = sycl::double2{-3.5, 0.798852};

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
  plan_bwd->commit(&q_ct1, 7,
                   dpct::fft::fft_type::complex_double_to_real_double, 2,
                   nullptr);
  plan_bwd->compute<sycl::double2, double>(forward_odata_d, backward_odata_d,
                                           dpct::fft::fft_direction::backward);
  dev_ct1.queues_wait_and_throw();
  double backward_odata_h[14];
  dpct::dpct_memcpy(backward_odata_h, backward_odata_d, 2 * sizeof(double) * 7,
                    dpct::device_to_host);

  double backward_odata_ref[14];
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

  dpct::dpct_free(forward_idata_d);
  dpct::dpct_free(forward_odata_d);
  dpct::dpct_free(backward_odata_d);

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
#define FUNC d2zz2d_1d_outofplace_make_plan
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

