// ===--- c2c_many_1d_outofplace_basic.dp.cpp -----------------*- C++ -*---===//
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

bool c2c_many_1d_outofplace_basic() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  dpct::fft::fft_engine *plan_fwd;
  sycl::float2 forward_idata_h[10];
  set_value((float*)forward_idata_h, 10);
  set_value((float*)forward_idata_h + 10, 10);

  sycl::float2 *forward_idata_d;
  sycl::float2 *forward_odata_d;
  sycl::float2 *backward_odata_d;
  forward_idata_d =
      (sycl::float2 *)dpct::dpct_malloc(sizeof(sycl::float2) * 10);
  forward_odata_d =
      (sycl::float2 *)dpct::dpct_malloc(sizeof(sycl::float2) * 10);
  backward_odata_d =
      (sycl::float2 *)dpct::dpct_malloc(sizeof(sycl::float2) * 10);
  dpct::dpct_memcpy(forward_idata_d, forward_idata_h, sizeof(sycl::float2) * 10,
                    dpct::host_to_device);

  int n[1] = {5};
  plan_fwd = dpct::fft::fft_engine::create(
      &q_ct1, 1, n, nullptr, 0, 0, nullptr, 0, 0,
      dpct::fft::fft_type::complex_float_to_complex_float, 2);
  plan_fwd->compute<sycl::float2, sycl::float2>(
      forward_idata_d, forward_odata_d, dpct::fft::fft_direction::forward);
  dev_ct1.queues_wait_and_throw();
  sycl::float2 forward_odata_h[10];
  dpct::dpct_memcpy(forward_odata_h, forward_odata_d, sizeof(sycl::float2) * 10,
                    dpct::device_to_host);

  sycl::float2 forward_odata_ref[10];
  forward_odata_ref[0] = sycl::float2{20, 25};
  forward_odata_ref[1] = sycl::float2{-11.8819, 1.88191};
  forward_odata_ref[2] = sycl::float2{-6.6246, -3.3754};
  forward_odata_ref[3] = sycl::float2{-3.3754, -6.6246};
  forward_odata_ref[4] = sycl::float2{1.88191, -11.8819};
  forward_odata_ref[5] = sycl::float2{20, 25};
  forward_odata_ref[6] = sycl::float2{-11.8819, 1.88191};
  forward_odata_ref[7] = sycl::float2{-6.6246, -3.3754};
  forward_odata_ref[8] = sycl::float2{-3.3754, -6.6246};
  forward_odata_ref[9] = sycl::float2{1.88191, -11.8819};

  dpct::fft::fft_engine::destroy(plan_fwd);

  if (!compare(forward_odata_ref, forward_odata_h, 10)) {
    std::cout << "forward_odata_h:" << std::endl;
    print_values(forward_odata_h, 10);
    std::cout << "forward_odata_ref:" << std::endl;
    print_values(forward_odata_ref, 10);

    dpct::dpct_free(forward_idata_d);
    dpct::dpct_free(forward_odata_d);
    dpct::dpct_free(backward_odata_d);

    return false;
  }

  dpct::fft::fft_engine *plan_bwd;
  plan_bwd = dpct::fft::fft_engine::create(
      &q_ct1, 1, n, nullptr, 0, 0, nullptr, 0, 0,
      dpct::fft::fft_type::complex_float_to_complex_float, 2);
  plan_bwd->compute<sycl::float2, sycl::float2>(
      forward_odata_d, backward_odata_d, dpct::fft::fft_direction::backward);
  dev_ct1.queues_wait_and_throw();
  sycl::float2 backward_odata_h[10];
  dpct::dpct_memcpy(backward_odata_h, backward_odata_d,
                    sizeof(sycl::float2) * 10, dpct::device_to_host);

  sycl::float2 backward_odata_ref[10];
  backward_odata_ref[0] = sycl::float2{0, 5};
  backward_odata_ref[1] = sycl::float2{10, 15};
  backward_odata_ref[2] = sycl::float2{20, 25};
  backward_odata_ref[3] = sycl::float2{30, 35};
  backward_odata_ref[4] = sycl::float2{40, 45};
  backward_odata_ref[5] = sycl::float2{0, 5};
  backward_odata_ref[6] = sycl::float2{10, 15};
  backward_odata_ref[7] = sycl::float2{20, 25};
  backward_odata_ref[8] = sycl::float2{30, 35};
  backward_odata_ref[9] = sycl::float2{40, 45};

  dpct::dpct_free(forward_idata_d);
  dpct::dpct_free(forward_odata_d);
  dpct::dpct_free(backward_odata_d);

  dpct::fft::fft_engine::destroy(plan_bwd);

  if (!compare(backward_odata_ref, backward_odata_h, 10)) {
    std::cout << "backward_odata_h:" << std::endl;
    print_values(backward_odata_h, 10);
    std::cout << "backward_odata_ref:" << std::endl;
    print_values(backward_odata_ref, 10);
    return false;
  }
  return true;
}


#ifdef DEBUG_FFT
int main() {
#define FUNC c2c_many_1d_outofplace_basic
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

