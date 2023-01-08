// ===--- c2c_1d_inplace_make_plan.dp.cpp ---------------------*- C++ -*---===//
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

bool c2c_1d_inplace_make_plan() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  dpct::fft::fft_engine *plan_fwd;
  plan_fwd = dpct::fft::fft_engine::create();
  sycl::float2 forward_idata_h[14];
  set_value((float*)forward_idata_h, 14);
  set_value((float*)forward_idata_h + 14, 14);

  sycl::float2 *data_d;
  data_d = (sycl::float2 *)dpct::dpct_malloc(2 * sizeof(sycl::float2) * 7);
  dpct::dpct_memcpy(data_d, forward_idata_h, 2 * sizeof(sycl::float2) * 7,
                    dpct::host_to_device);

  size_t workSize;
  plan_fwd->commit(&q_ct1, 7,
                   dpct::fft::fft_type::complex_float_to_complex_float, 2,
                   nullptr);
  plan_fwd->compute<sycl::float2, sycl::float2>(
      data_d, data_d, dpct::fft::fft_direction::forward);
  dev_ct1.queues_wait_and_throw();
  sycl::float2 forward_odata_h[14];
  dpct::dpct_memcpy(forward_odata_h, data_d, 2 * sizeof(sycl::float2) * 7,
                    dpct::device_to_host);

  sycl::float2 forward_odata_ref[14];
  forward_odata_ref[0] = sycl::float2{42, 49};
  forward_odata_ref[1] = sycl::float2{-21.5356, 7.53565};
  forward_odata_ref[2] = sycl::float2{-12.5823, -1.41769};
  forward_odata_ref[3] = sycl::float2{-8.5977, -5.4023};
  forward_odata_ref[4] = sycl::float2{-5.4023, -8.5977};
  forward_odata_ref[5] = sycl::float2{-1.41769, -12.5823};
  forward_odata_ref[6] = sycl::float2{7.53565, -21.5356};
  forward_odata_ref[7] = sycl::float2{42, 49};
  forward_odata_ref[8] = sycl::float2{-21.5356, 7.53565};
  forward_odata_ref[9] = sycl::float2{-12.5823, -1.41769};
  forward_odata_ref[10] = sycl::float2{-8.5977, -5.4023};
  forward_odata_ref[11] = sycl::float2{-5.4023, -8.5977};
  forward_odata_ref[12] = sycl::float2{-1.41769, -12.5823};
  forward_odata_ref[13] = sycl::float2{7.53565, -21.5356};

  dpct::fft::fft_engine::destroy(plan_fwd);

  if (!compare(forward_odata_ref, forward_odata_h, 14)) {
    std::cout << "forward_odata_h:" << std::endl;
    print_values(forward_odata_h, 14);
    std::cout << "forward_odata_ref:" << std::endl;
    print_values(forward_odata_ref, 14);

    dpct::dpct_free(data_d);

    return false;
  }

  dpct::fft::fft_engine *plan_bwd;
  plan_bwd = dpct::fft::fft_engine::create();
  plan_bwd->commit(&q_ct1, 7,
                   dpct::fft::fft_type::complex_float_to_complex_float, 2,
                   nullptr);
  plan_bwd->compute<sycl::float2, sycl::float2>(
      data_d, data_d, dpct::fft::fft_direction::backward);
  dev_ct1.queues_wait_and_throw();
  sycl::float2 backward_odata_h[14];
  dpct::dpct_memcpy(backward_odata_h, data_d, 2 * sizeof(sycl::float2) * 7,
                    dpct::device_to_host);

  sycl::float2 backward_odata_ref[14];
  backward_odata_ref[0] = sycl::float2{0, 7};
  backward_odata_ref[1] = sycl::float2{14, 21};
  backward_odata_ref[2] = sycl::float2{28, 35};
  backward_odata_ref[3] = sycl::float2{42, 49};
  backward_odata_ref[4] = sycl::float2{56, 63};
  backward_odata_ref[5] = sycl::float2{70, 77};
  backward_odata_ref[6] = sycl::float2{84, 91};
  backward_odata_ref[7] = sycl::float2{0, 7};
  backward_odata_ref[8] = sycl::float2{14, 21};
  backward_odata_ref[9] = sycl::float2{28, 35};
  backward_odata_ref[10] = sycl::float2{42, 49};
  backward_odata_ref[11] = sycl::float2{56, 63};
  backward_odata_ref[12] = sycl::float2{70, 77};
  backward_odata_ref[13] = sycl::float2{84, 91};

  dpct::dpct_free(data_d);
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
#define FUNC c2c_1d_inplace_make_plan
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

