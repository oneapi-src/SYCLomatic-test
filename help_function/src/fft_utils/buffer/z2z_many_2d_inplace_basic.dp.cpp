// ===--- z2z_many_2d_inplace_basic.dp.cpp --------------------*- C++ -*---===//
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

bool z2z_many_2d_inplace_basic() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  dpct::fft::fft_engine *plan_fwd;
  plan_fwd = dpct::fft::fft_engine::create();
  sycl::double2 forward_idata_h[2 /*n0*/ * 3 /*n1*/ * 2 /*batch*/];
  set_value((double*)forward_idata_h, 12);
  set_value((double*)forward_idata_h + 12, 12);

  sycl::double2 *data_d;
  data_d = (sycl::double2 *)dpct::dpct_malloc(sizeof(sycl::double2) * 12);
  dpct::dpct_memcpy(data_d, forward_idata_h, sizeof(sycl::double2) * 12,
                    dpct::host_to_device);

  int n[2] = {2, 3};
  size_t workSize;
  plan_fwd->commit(&q_ct1, 2, n, nullptr, 0, 0, nullptr, 0, 0,
                   dpct::fft::fft_type::complex_double_to_complex_double, 2,
                   nullptr);
  plan_fwd->compute<sycl::double2, sycl::double2>(
      data_d, data_d, dpct::fft::fft_direction::forward);
  dev_ct1.queues_wait_and_throw();
  sycl::double2 forward_odata_h[12];
  dpct::dpct_memcpy(forward_odata_h, data_d, sizeof(sycl::double2) * 12,
                    dpct::device_to_host);

  sycl::double2 forward_odata_ref[12];
  forward_odata_ref[0] = sycl::double2{30, 36};
  forward_odata_ref[1] = sycl::double2{-9.4641, -2.5359};
  forward_odata_ref[2] = sycl::double2{-2.5359, -9.4641};
  forward_odata_ref[3] = sycl::double2{-18, -18};
  forward_odata_ref[4] = sycl::double2{0, 0};
  forward_odata_ref[5] = sycl::double2{0, 0};
  forward_odata_ref[6] = sycl::double2{30, 36};
  forward_odata_ref[7] = sycl::double2{-9.4641, -2.5359};
  forward_odata_ref[8] = sycl::double2{-2.5359, -9.4641};
  forward_odata_ref[9] = sycl::double2{-18, -18};
  forward_odata_ref[10] = sycl::double2{0, 0};
  forward_odata_ref[11] = sycl::double2{0, 0};

  dpct::fft::fft_engine::destroy(plan_fwd);

  if (!compare(forward_odata_ref, forward_odata_h, 12)) {
    std::cout << "forward_odata_h:" << std::endl;
    print_values(forward_odata_h, 12);
    std::cout << "forward_odata_ref:" << std::endl;
    print_values(forward_odata_ref, 12);

    dpct::dpct_free(data_d);

    return false;
  }

  dpct::fft::fft_engine *plan_bwd;
  plan_bwd = dpct::fft::fft_engine::create();
  plan_bwd->commit(&q_ct1, 2, n, nullptr, 0, 0, nullptr, 0, 0,
                   dpct::fft::fft_type::complex_double_to_complex_double, 2,
                   nullptr);
  plan_bwd->compute<sycl::double2, sycl::double2>(
      data_d, data_d, dpct::fft::fft_direction::backward);
  dev_ct1.queues_wait_and_throw();
  sycl::double2 backward_odata_h[12];
  dpct::dpct_memcpy(backward_odata_h, data_d, sizeof(sycl::double2) * 12,
                    dpct::device_to_host);

  sycl::double2 backward_odata_ref[12];
  backward_odata_ref[0] = sycl::double2{0, 6};
  backward_odata_ref[1] = sycl::double2{12, 18};
  backward_odata_ref[2] = sycl::double2{24, 30};
  backward_odata_ref[3] = sycl::double2{36, 42};
  backward_odata_ref[4] = sycl::double2{48, 54};
  backward_odata_ref[5] = sycl::double2{60, 66};
  backward_odata_ref[6] = sycl::double2{0, 6};
  backward_odata_ref[7] = sycl::double2{12, 18};
  backward_odata_ref[8] = sycl::double2{24, 30};
  backward_odata_ref[9] = sycl::double2{36, 42};
  backward_odata_ref[10] = sycl::double2{48, 54};
  backward_odata_ref[11] = sycl::double2{60, 66};

  dpct::dpct_free(data_d);
  dpct::fft::fft_engine::destroy(plan_bwd);

  if (!compare(backward_odata_ref, backward_odata_h, 12)) {
    std::cout << "backward_odata_h:" << std::endl;
    print_values(backward_odata_h, 12);
    std::cout << "backward_odata_ref:" << std::endl;
    print_values(backward_odata_ref, 12);
    return false;
  }
  return true;
}

#ifdef DEBUG_FFT
int main() {
#define FUNC z2z_many_2d_inplace_basic
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

