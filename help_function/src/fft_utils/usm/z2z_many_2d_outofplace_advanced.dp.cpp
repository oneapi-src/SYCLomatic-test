// ===--- z2z_many_2d_outofplace_advanced.dp.cpp --------------*- C++ -*---===//
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
#include <dpct/lib_common_utils.hpp>

// forward
// input
// +---+---+---+---+---+---+      -+
// |   c   |   c   |   c   |       |
// +---+---+---+---+---+---+       |
// |   c   |   c   |   c   |       batch0
// +---+---+---+---+---+---+      -+
// |   c   |   c   |   c   |       |
// +---+---+---+---+---+---+       |
// |   c   |   c   |   c   |       batch1
// +---+---+---+---+---+---+      -+
// |___________n2__________|
// |________nembed2________|
// output
// +---+---+---+---+---+---+ -+
// |   c   |   c   |   c   |  |
// +---+---+---+---+---+---+  batch0
// |   c   |   c   |   c   |  |
// +---+---+---+---+---+---+ -+
// |   c   |   c   |   c   |  |
// +---+---+---+---+---+---+  batch1
// |   c   |   c   |   c   |  |
// +---+---+---+---+---+---+ -+
// |__________n2___________|
// |________nembed2________|
bool z2z_many_2d_outofplace_advanced() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  dpct::fft::fft_engine *plan_fwd;
  plan_fwd = dpct::fft::fft_engine::create();
  sycl::double2 forward_idata_h[12];
  std::memset(forward_idata_h, 0, sizeof(sycl::double2) * 12);
  forward_idata_h[0] = sycl::double2{0, 1};
  forward_idata_h[1] = sycl::double2{2, 3};
  forward_idata_h[2] = sycl::double2{4, 5};
  forward_idata_h[3] = sycl::double2{6, 7};
  forward_idata_h[4] = sycl::double2{8, 9};
  forward_idata_h[5] = sycl::double2{10, 11};
  forward_idata_h[6] = sycl::double2{0, 1};
  forward_idata_h[7] = sycl::double2{2, 3};
  forward_idata_h[8] = sycl::double2{4, 5};
  forward_idata_h[9] = sycl::double2{6, 7};
  forward_idata_h[10] = sycl::double2{8, 9};
  forward_idata_h[11] = sycl::double2{10, 11};

  sycl::double2 *forward_idata_d;
  sycl::double2 *forward_odata_d;
  sycl::double2 *backward_odata_d;
  forward_idata_d = sycl::malloc_device<sycl::double2>(12, q_ct1);
  forward_odata_d = sycl::malloc_device<sycl::double2>(12, q_ct1);
  backward_odata_d = sycl::malloc_device<sycl::double2>(12, q_ct1);
  q_ct1.memcpy(forward_idata_d, forward_idata_h, sizeof(sycl::double2) * 12)
      .wait();

  size_t workSize;
  long long int n[2] = {2, 3};
  long long int inembed[2] = {2, 3};
  long long int onembed[2] = {2, 3};
  plan_fwd->commit(&q_ct1, 2, n, inembed, 1, 6,
                   dpct::library_data_t::complex_double, onembed, 1, 6,
                   dpct::library_data_t::complex_double, 2, nullptr);
  plan_fwd->compute<void, void>(forward_idata_d, forward_odata_d,
                                dpct::fft::fft_direction::forward);
  dev_ct1.queues_wait_and_throw();
  sycl::double2 forward_odata_h[12];
  q_ct1.memcpy(forward_odata_h, forward_odata_d, sizeof(sycl::double2) * 12)
      .wait();

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

    sycl::free(forward_idata_d, q_ct1);
    sycl::free(forward_odata_d, q_ct1);
    sycl::free(backward_odata_d, q_ct1);

    return false;
  }

  dpct::fft::fft_engine *plan_bwd;
  plan_bwd = dpct::fft::fft_engine::create();
  plan_bwd->commit(&q_ct1, 2, n, onembed, 1, 6,
                   dpct::library_data_t::complex_double, inembed, 1, 6,
                   dpct::library_data_t::complex_double, 2, nullptr);
  plan_bwd->compute<void, void>(forward_odata_d, backward_odata_d,
                                dpct::fft::fft_direction::backward);
  dev_ct1.queues_wait_and_throw();
  sycl::double2 backward_odata_h[12];
  q_ct1.memcpy(backward_odata_h, backward_odata_d, sizeof(sycl::double2) * 12)
      .wait();

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

  sycl::free(forward_idata_d, q_ct1);
  sycl::free(forward_odata_d, q_ct1);
  sycl::free(backward_odata_d, q_ct1);

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
#define FUNC z2z_many_2d_outofplace_advanced
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

