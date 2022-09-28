// ===--- z2z_1d_inplace.dp.cpp -------------------------------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===---------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <oneapi/mkl.hpp>
#include <dpct/fft_utils.hpp>
#include "common.h"
#include <cstring>
#include <iostream>

bool z2z_1d_inplace() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  dpct::fft::fft_engine *plan_fwd;
  sycl::double2 forward_idata_h[14];
  set_value((double*)forward_idata_h, 14);
  set_value((double*)forward_idata_h + 14, 14);

  sycl::double2 *data_d;
  data_d = (sycl::double2 *)sycl::malloc_device(2 * sizeof(sycl::double2) * 7,
                                                q_ct1);
  q_ct1.memcpy(data_d, forward_idata_h, 2 * sizeof(sycl::double2) * 7).wait();

  plan_fwd = dpct::fft::fft_engine::create(
      &q_ct1, 7, dpct::fft::fft_type::complex_double_to_complex_double, 2);
  plan_fwd->compute<sycl::double2, sycl::double2>(
      data_d, data_d, dpct::fft::fft_direction::forward);
  dev_ct1.queues_wait_and_throw();
  sycl::double2 forward_odata_h[14];
  q_ct1.memcpy(forward_odata_h, data_d, 2 * sizeof(sycl::double2) * 7).wait();

  sycl::double2 forward_odata_ref[14];
  forward_odata_ref[0] = sycl::double2{42, 49};
  forward_odata_ref[1] = sycl::double2{-21.5356, 7.53565};
  forward_odata_ref[2] = sycl::double2{-12.5823, -1.41769};
  forward_odata_ref[3] = sycl::double2{-8.5977, -5.4023};
  forward_odata_ref[4] = sycl::double2{-5.4023, -8.5977};
  forward_odata_ref[5] = sycl::double2{-1.41769, -12.5823};
  forward_odata_ref[6] = sycl::double2{7.53565, -21.5356};
  forward_odata_ref[7] = sycl::double2{42, 49};
  forward_odata_ref[8] = sycl::double2{-21.5356, 7.53565};
  forward_odata_ref[9] = sycl::double2{-12.5823, -1.41769};
  forward_odata_ref[10] = sycl::double2{-8.5977, -5.4023};
  forward_odata_ref[11] = sycl::double2{-5.4023, -8.5977};
  forward_odata_ref[12] = sycl::double2{-1.41769, -12.5823};
  forward_odata_ref[13] = sycl::double2{7.53565, -21.5356};

  dpct::fft::fft_engine::destroy(plan_fwd);

  if (!compare(forward_odata_ref, forward_odata_h, 14)) {
    std::cout << "forward_odata_h:" << std::endl;
    print_values(forward_odata_h, 14);
    std::cout << "forward_odata_ref:" << std::endl;
    print_values(forward_odata_ref, 14);

    sycl::free(data_d, q_ct1);

    return false;
  }

  dpct::fft::fft_engine *plan_bwd;
  plan_bwd = dpct::fft::fft_engine::create(
      &q_ct1, 7, dpct::fft::fft_type::complex_double_to_complex_double, 2);
  plan_bwd->compute<sycl::double2, sycl::double2>(
      data_d, data_d, dpct::fft::fft_direction::backward);
  dev_ct1.queues_wait_and_throw();
  sycl::double2 backward_odata_h[14];
  q_ct1.memcpy(backward_odata_h, data_d, 2 * sizeof(sycl::double2) * 7).wait();

  sycl::double2 backward_odata_ref[14];
  backward_odata_ref[0] = sycl::double2{0, 7};
  backward_odata_ref[1] = sycl::double2{14, 21};
  backward_odata_ref[2] = sycl::double2{28, 35};
  backward_odata_ref[3] = sycl::double2{42, 49};
  backward_odata_ref[4] = sycl::double2{56, 63};
  backward_odata_ref[5] = sycl::double2{70, 77};
  backward_odata_ref[6] = sycl::double2{84, 91};
  backward_odata_ref[7] = sycl::double2{0, 7};
  backward_odata_ref[8] = sycl::double2{14, 21};
  backward_odata_ref[9] = sycl::double2{28, 35};
  backward_odata_ref[10] = sycl::double2{42, 49};
  backward_odata_ref[11] = sycl::double2{56, 63};
  backward_odata_ref[12] = sycl::double2{70, 77};
  backward_odata_ref[13] = sycl::double2{84, 91};

  sycl::free(data_d, q_ct1);
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
#define FUNC z2z_1d_inplace
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

