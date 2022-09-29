// ===--- d2zz2d_many_1d_outofplace_advanced.dp.cpp -----------*- C++ -*---===//
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
#include <dpct/lib_common_utils.hpp>

// forward
// input
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
// | r | 0 | r | 0 | r | 0 | r | 0 | r | 0 | 0 | 0 | 0 | r | 0 | r | 0 | r | 0 | r | 0 | r | 0 | 0 | 0 | 0 |
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
// |___________________n___________________|       |   |___________________n___________________|       |   |
// |___________________nembed______________________|   |___________________nembed______________________|   |
// |_______________________batch0______________________|________________________batch1_____________________|
// output
// +---+---+---+---+---+---+---+---+---+---+---+---+
// |   c   |   c   |   c   |   c   |   c   |   c   |
// +---+---+---+---+---+---+---+---+---+---+---+---+
// |___________n___________|___________n___________|
// |_________nembed________|_________nembed________|
// |_________batch0________|_________batch1________|
bool d2zz2d_many_1d_outofplace_advanced() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  dpct::fft::fft_engine *plan_fwd;
  plan_fwd = dpct::fft::fft_engine::create();
  double forward_idata_h[26];
  std::memset(forward_idata_h, 0, sizeof(double) * 26);
  set_value_with_stride(forward_idata_h, 5, 2);
  set_value_with_stride(forward_idata_h + 13, 5, 2);

  double* forward_idata_d;
  sycl::double2 *forward_odata_d;
  double* backward_odata_d;
  forward_idata_d = (double *)dpct::dpct_malloc(sizeof(double) * 26);
  forward_odata_d =
      (sycl::double2 *)dpct::dpct_malloc(sizeof(sycl::double2) * 6);
  backward_odata_d = (double *)dpct::dpct_malloc(sizeof(double) * 26);
  dpct::dpct_memcpy(forward_idata_d, forward_idata_h, sizeof(double) * 26,
                    dpct::host_to_device);

  long long int n[1] = {5};
  long long int inembed[1] = {12};
  long long int onembed[1] = {3};
  size_t workSize;
  plan_fwd->commit(&q_ct1, 1, n, inembed, 2, 13,
                   dpct::library_data_t::real_double, onembed, 1, 3,
                   dpct::library_data_t::complex_double, 2, nullptr);
  plan_fwd->compute<void, void>(forward_idata_d, forward_odata_d,
                                dpct::fft::fft_direction::forward);
  dev_ct1.queues_wait_and_throw();
  sycl::double2 forward_odata_h[6];
  dpct::dpct_memcpy(forward_odata_h, forward_odata_d, sizeof(sycl::double2) * 6,
                    dpct::device_to_host);

  sycl::double2 forward_odata_ref[6];
  forward_odata_ref[0] = sycl::double2{10, 0};
  forward_odata_ref[1] = sycl::double2{-2.5, 3.44095};
  forward_odata_ref[2] = sycl::double2{-2.5, 0.812299};
  forward_odata_ref[3] = sycl::double2{10, 0};
  forward_odata_ref[4] = sycl::double2{-2.5, 3.44095};
  forward_odata_ref[5] = sycl::double2{-2.5, 0.812299};

  dpct::fft::fft_engine::destroy(plan_fwd);

  if (!compare(forward_odata_ref, forward_odata_h, 6)) {
    std::cout << "forward_odata_h:" << std::endl;
    print_values(forward_odata_h, 6);
    std::cout << "forward_odata_ref:" << std::endl;
    print_values(forward_odata_ref, 6);

    dpct::dpct_free(forward_idata_d);
    dpct::dpct_free(forward_odata_d);
    dpct::dpct_free(backward_odata_d);

    return false;
  }

  dpct::fft::fft_engine *plan_bwd;
  plan_bwd = dpct::fft::fft_engine::create();
  plan_bwd->commit(&q_ct1, 1, n, onembed, 1, 3,
                   dpct::library_data_t::complex_double, inembed, 2, 13,
                   dpct::library_data_t::real_double, 2, nullptr);
  plan_bwd->compute<void, void>(forward_odata_d, backward_odata_d,
                                dpct::fft::fft_direction::backward);
  dev_ct1.queues_wait_and_throw();
  double backward_odata_h[26];
  dpct::dpct_memcpy(backward_odata_h, backward_odata_d, sizeof(double) * 26,
                    dpct::device_to_host);

  double backward_odata_ref[26];
  backward_odata_ref[0]  = 0;
  backward_odata_ref[1]  = 0;
  backward_odata_ref[2]  = 5;
  backward_odata_ref[3]  = 0;
  backward_odata_ref[4]  = 10;
  backward_odata_ref[5]  = 0;
  backward_odata_ref[6]  = 15;
  backward_odata_ref[7]  = 0;
  backward_odata_ref[8]  = 20;
  backward_odata_ref[9]  = 0;
  backward_odata_ref[10] = 0;
  backward_odata_ref[11] = 0;
  backward_odata_ref[12] = 0;
  backward_odata_ref[13] = 0;
  backward_odata_ref[14] = 0;
  backward_odata_ref[15] = 5;
  backward_odata_ref[16] = 0;
  backward_odata_ref[17] = 10;
  backward_odata_ref[18] = 0;
  backward_odata_ref[19] = 15;
  backward_odata_ref[20] = 0;
  backward_odata_ref[21] = 20;
  backward_odata_ref[22] = 0;
  backward_odata_ref[23] = 0;
  backward_odata_ref[24] = 0;
  backward_odata_ref[25] = 0;

  dpct::dpct_free(forward_idata_d);
  dpct::dpct_free(forward_odata_d);
  dpct::dpct_free(backward_odata_d);

  dpct::fft::fft_engine::destroy(plan_bwd);

  std::vector<int> indices = {0, 2, 4, 6, 8,
                              13, 15, 17, 19, 21};
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
#define FUNC d2zz2d_many_1d_outofplace_advanced
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

