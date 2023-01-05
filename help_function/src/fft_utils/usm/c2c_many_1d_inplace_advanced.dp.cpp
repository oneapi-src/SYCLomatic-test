// ===--- c2c_many_1d_inplace_advanced.dp.cpp -----------------*- C++ -*---===//
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



// forward
// input
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
// |   c   |   0   |   c   |   0   |   c   |   0   |   0   |   0   |   0   |   c   |   0   |   c   |   0   |   c   |   0   |   0   |   0   |   0   |
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
// |________________________n______________________|               |       |________________________n______________________|               |       |
// |_____________________________nembed____________________________|       |_____________________________nembed____________________________|       |
// |___________________________________batch0______________________________|___________________________________batch1______________________________|
// output
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
// |   c   |   0   |   c   |   0   |   c   |   0   |   0   |   0   |   0   |   c   |   0   |   c   |   0   |   c   |   0   |   0   |   0   |   0   |
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
// |________________________n______________________|               |       |________________________n______________________|               |       |
// |_____________________________nembed____________________________|       |_____________________________nembed____________________________|       |
// |___________________________________batch0______________________________|___________________________________batch1______________________________|
bool c2c_many_1d_inplace_advanced() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  dpct::fft::fft_engine *plan_fwd;
  plan_fwd = dpct::fft::fft_engine::create();
  sycl::float2 forward_idata_h[18];
  std::memset(forward_idata_h, 0, sizeof(sycl::float2) * 18);
  forward_idata_h[0] = sycl::float2{0, 1};
  forward_idata_h[2] = sycl::float2{2, 3};
  forward_idata_h[4] = sycl::float2{4, 5};
  forward_idata_h[9] = sycl::float2{0, 1};
  forward_idata_h[11] = sycl::float2{2, 3};
  forward_idata_h[13] = sycl::float2{4, 5};

  sycl::float2 *data_d;
  data_d = sycl::malloc_device<sycl::float2>(18, q_ct1);
  q_ct1.memcpy(data_d, forward_idata_h, sizeof(sycl::float2) * 18).wait();

  size_t workSize;
  long long int n[1] = {3};
  long long int inembed[1] = {4};
  long long int onembed[1] = {4};
  plan_fwd->commit(&q_ct1, 1, n, inembed, 2, 9, onembed, 2, 9,
                   dpct::fft::fft_type::complex_float_to_complex_float, 2,
                   nullptr);
  plan_fwd->compute<sycl::float2, sycl::float2>(
      data_d, data_d, dpct::fft::fft_direction::forward);
  dev_ct1.queues_wait_and_throw();
  sycl::float2 forward_odata_h[18];
  q_ct1.memcpy(forward_odata_h, data_d, sizeof(sycl::float2) * 18).wait();

  sycl::float2 forward_odata_ref[18];
  forward_odata_ref[0] = sycl::float2{6, 9};
  forward_odata_ref[1] = sycl::float2{2, 3};
  forward_odata_ref[2] = sycl::float2{-4.73205, -1.26795};
  forward_odata_ref[3] = sycl::float2{0, 1};
  forward_odata_ref[4] = sycl::float2{-1.26795, -4.73205};
  forward_odata_ref[5] = sycl::float2{4, 5};
  forward_odata_ref[6] = sycl::float2{0, 0};
  forward_odata_ref[7] = sycl::float2{0, 0};
  forward_odata_ref[8] = sycl::float2{0, 0};
  forward_odata_ref[9] = sycl::float2{6, 9};
  forward_odata_ref[10] = sycl::float2{0, 0};
  forward_odata_ref[11] = sycl::float2{-4.73205, -1.26795};
  forward_odata_ref[12] = sycl::float2{0, 0};
  forward_odata_ref[13] = sycl::float2{-1.26795, -4.73205};
  forward_odata_ref[14] = sycl::float2{0, 0};
  forward_odata_ref[15] = sycl::float2{0, 0};
  forward_odata_ref[16] = sycl::float2{0, 0};
  forward_odata_ref[17] = sycl::float2{0, 0};

  dpct::fft::fft_engine::destroy(plan_fwd);

  std::vector<int> indices = {0, 2, 4,
                              9, 11, 13};
  if (!compare(forward_odata_ref, forward_odata_h, indices)) {
    std::cout << "forward_odata_h:" << std::endl;
    print_values(forward_odata_h, indices);
    std::cout << "forward_odata_ref:" << std::endl;
    print_values(forward_odata_ref, indices);

    sycl::free(data_d, q_ct1);

    return false;
  }

  dpct::fft::fft_engine *plan_bwd;
  plan_bwd = dpct::fft::fft_engine::create();
  plan_bwd->commit(&q_ct1, 1, n, onembed, 2, 9, inembed, 2, 9,
                   dpct::fft::fft_type::complex_float_to_complex_float, 2,
                   nullptr);
  plan_bwd->compute<sycl::float2, sycl::float2>(
      data_d, data_d, dpct::fft::fft_direction::backward);
  dev_ct1.queues_wait_and_throw();
  sycl::float2 backward_odata_h[18];
  q_ct1.memcpy(backward_odata_h, data_d, sizeof(sycl::float2) * 18).wait();

  sycl::float2 backward_odata_ref[18];
  backward_odata_ref[0] = sycl::float2{0, 3};
  backward_odata_ref[1] = sycl::float2{0, 0};
  backward_odata_ref[2] = sycl::float2{6, 9};
  backward_odata_ref[3] = sycl::float2{0, 0};
  backward_odata_ref[4] = sycl::float2{12, 15};
  backward_odata_ref[5] = sycl::float2{0, 0};
  backward_odata_ref[6] = sycl::float2{0, 0};
  backward_odata_ref[7] = sycl::float2{0, 0};
  backward_odata_ref[8] = sycl::float2{0, 0};
  backward_odata_ref[9] = sycl::float2{0, 3};
  backward_odata_ref[10] = sycl::float2{0, 0};
  backward_odata_ref[11] = sycl::float2{6, 9};
  backward_odata_ref[12] = sycl::float2{0, 0};
  backward_odata_ref[13] = sycl::float2{12, 15};
  backward_odata_ref[14] = sycl::float2{0, 0};
  backward_odata_ref[15] = sycl::float2{0, 0};
  backward_odata_ref[16] = sycl::float2{0, 0};
  backward_odata_ref[17] = sycl::float2{0, 0};

  sycl::free(data_d, q_ct1);
  dpct::fft::fft_engine::destroy(plan_bwd);

  std::vector<int> bwd_indices = {0, 2, 4,
                                9, 11, 13};
  if (!compare(backward_odata_ref, backward_odata_h, bwd_indices)) {
    std::cout << "backward_odata_h:" << std::endl;
    print_values(backward_odata_h, bwd_indices);
    std::cout << "backward_odata_ref:" << std::endl;
    print_values(backward_odata_ref, bwd_indices);
    return false;
  }
  return true;
}


#ifdef DEBUG_FFT
int main() {
#define FUNC c2c_many_1d_inplace_advanced
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

