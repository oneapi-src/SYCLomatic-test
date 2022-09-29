// ===--- c2c_many_2d_inplace_advanced.dp.cpp -----------------*- C++ -*---===//
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


// forward
// input
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+         -+
// |   c   |   0   |   c   |   0   |   c   |   0   |   0   |   0   |          |
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+          |
// |   c   |   0   |   c   |   0   |   c   |   0   |   0   |   0   |          batch0
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+  |  
// |   0   |   0   |   0   |   0   |   0   |   0   |   0   |   0   |   0   |  |
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+ -+
// |   c   |   0   |   c   |   0   |   c   |   0   |   0   |   0   |          |
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+          |
// |   c   |   0   |   c   |   0   |   c   |   0   |   0   |   0   |          batch1
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+  |  
// |   0   |   0   |   0   |   0   |   0   |   0   |   0   |   0   |   0   |  |
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+ -+
// |___________n2__________|
// |________nembed2________|
// output
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+         -+
// |   c   |   0   |   c   |   0   |   c   |   0   |   0   |   0   |          |
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+          |
// |   c   |   0   |   c   |   0   |   c   |   0   |   0   |   0   |          batch0
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+  |  
// |   0   |   0   |   0   |   0   |   0   |   0   |   0   |   0   |   0   |  |
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+ -+
// |   c   |   0   |   c   |   0   |   c   |   0   |   0   |   0   |          |
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+          |
// |   c   |   0   |   c   |   0   |   c   |   0   |   0   |   0   |          batch1
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+  |  
// |   0   |   0   |   0   |   0   |   0   |   0   |   0   |   0   |   0   |  |
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+ -+
// |_______________________n2______________________|               |
// |____________________________nembed2____________________________|
bool c2c_many_2d_inplace_advanced() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  dpct::fft::fft_engine *plan_fwd;
  plan_fwd = dpct::fft::fft_engine::create();
  sycl::float2 forward_idata_h[50];
  std::memset(forward_idata_h, 0, sizeof(sycl::float2) * 50);
  forward_idata_h[0] = sycl::float2{0, 1};
  forward_idata_h[2] = sycl::float2{2, 3};
  forward_idata_h[4] = sycl::float2{4, 5};
  forward_idata_h[8] = sycl::float2{6, 7};
  forward_idata_h[10] = sycl::float2{8, 9};
  forward_idata_h[12] = sycl::float2{10, 11};
  forward_idata_h[25] = sycl::float2{0, 1};
  forward_idata_h[27] = sycl::float2{2, 3};
  forward_idata_h[29] = sycl::float2{4, 5};
  forward_idata_h[33] = sycl::float2{6, 7};
  forward_idata_h[35] = sycl::float2{8, 9};
  forward_idata_h[37] = sycl::float2{10, 11};

  sycl::float2 *data_d;
  data_d = sycl::malloc_device<sycl::float2>(50, q_ct1);
  q_ct1.memcpy(data_d, forward_idata_h, sizeof(sycl::float2) * 50).wait();

  size_t workSize;
  long long int n[2] = {2, 3};
  long long int inembed[2] = {3, 4};
  long long int onembed[2] = {3, 4};
  plan_fwd->commit(&q_ct1, 2, n, inembed, 2, 25, onembed, 2, 25,
                   dpct::fft::fft_type::complex_float_to_complex_float, 2,
                   nullptr);
  plan_fwd->compute<sycl::float2, sycl::float2>(
      data_d, data_d, dpct::fft::fft_direction::forward);
  dev_ct1.queues_wait_and_throw();
  sycl::float2 forward_odata_h[50];
  q_ct1.memcpy(forward_odata_h, data_d, sizeof(sycl::float2) * 50).wait();

  sycl::float2 forward_odata_ref[50];
  forward_odata_ref[0] = sycl::float2{30, 36};
  forward_odata_ref[1] = sycl::float2{2, 3};
  forward_odata_ref[2] = sycl::float2{-9.4641, -2.5359};
  forward_odata_ref[3] = sycl::float2{6, 7};
  forward_odata_ref[4] = sycl::float2{-2.5359, -9.4641};
  forward_odata_ref[5] = sycl::float2{10, 11};
  forward_odata_ref[6] = sycl::float2{0, 0};
  forward_odata_ref[7] = sycl::float2{0, 0};
  forward_odata_ref[8] = sycl::float2{-18, -18};
  forward_odata_ref[9] = sycl::float2{0, 0};
  forward_odata_ref[10] = sycl::float2{0, 0};
  forward_odata_ref[11] = sycl::float2{0, 0};
  forward_odata_ref[12] = sycl::float2{0, 0};
  forward_odata_ref[13] = sycl::float2{0, 0};
  forward_odata_ref[14] = sycl::float2{0, 0};
  forward_odata_ref[15] = sycl::float2{0, 0};
  forward_odata_ref[16] = sycl::float2{0, 0};
  forward_odata_ref[17] = sycl::float2{0, 0};
  forward_odata_ref[18] = sycl::float2{0, 0};
  forward_odata_ref[19] = sycl::float2{0, 0};
  forward_odata_ref[20] = sycl::float2{0, 0};
  forward_odata_ref[21] = sycl::float2{0, 0};
  forward_odata_ref[22] = sycl::float2{0, 0};
  forward_odata_ref[23] = sycl::float2{0, 0};
  forward_odata_ref[24] = sycl::float2{0, 0};
  forward_odata_ref[25] = sycl::float2{30, 36};
  forward_odata_ref[26] = sycl::float2{2, 3};
  forward_odata_ref[27] = sycl::float2{-9.4641, -2.5359};
  forward_odata_ref[28] = sycl::float2{6, 7};
  forward_odata_ref[29] = sycl::float2{-2.5359, -9.4641};
  forward_odata_ref[30] = sycl::float2{10, 11};
  forward_odata_ref[31] = sycl::float2{0, 0};
  forward_odata_ref[32] = sycl::float2{0, 0};
  forward_odata_ref[33] = sycl::float2{-18, -18};
  forward_odata_ref[34] = sycl::float2{0, 0};
  forward_odata_ref[35] = sycl::float2{0, 0};
  forward_odata_ref[36] = sycl::float2{0, 0};
  forward_odata_ref[37] = sycl::float2{0, 0};

  dpct::fft::fft_engine::destroy(plan_fwd);

  std::vector<int> indices = {0, 2, 4,
                              8, 10, 12,
                              25, 27, 29,
                              33, 35, 37};
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
  plan_bwd->commit(&q_ct1, 2, n, onembed, 2, 25, inembed, 2, 25,
                   dpct::fft::fft_type::complex_float_to_complex_float, 2,
                   nullptr);
  plan_bwd->compute<sycl::float2, sycl::float2>(
      data_d, data_d, dpct::fft::fft_direction::backward);
  dev_ct1.queues_wait_and_throw();
  sycl::float2 backward_odata_h[50];
  q_ct1.memcpy(backward_odata_h, data_d, sizeof(sycl::float2) * 50).wait();

  sycl::float2 backward_odata_ref[50];
  backward_odata_ref[0] = sycl::float2{0, 6};
  backward_odata_ref[2] = sycl::float2{12, 18};
  backward_odata_ref[4] = sycl::float2{24, 30};
  backward_odata_ref[8] = sycl::float2{36, 42};
  backward_odata_ref[10] = sycl::float2{48, 54};
  backward_odata_ref[12] = sycl::float2{60, 66};
  backward_odata_ref[25] = sycl::float2{0, 6};
  backward_odata_ref[27] = sycl::float2{12, 18};
  backward_odata_ref[29] = sycl::float2{24, 30};
  backward_odata_ref[33] = sycl::float2{36, 42};
  backward_odata_ref[35] = sycl::float2{48, 54};
  backward_odata_ref[37] = sycl::float2{60, 66};

  sycl::free(data_d, q_ct1);
  dpct::fft::fft_engine::destroy(plan_bwd);

  std::vector<int> indices_bwd = {0, 2, 4, 8, 10, 12,
                                  25, 27, 29, 33, 35, 37};
  if (!compare(backward_odata_ref, backward_odata_h, indices_bwd)) {
    std::cout << "backward_odata_h:" << std::endl;
    print_values(backward_odata_h, indices_bwd);
    std::cout << "backward_odata_ref:" << std::endl;
    print_values(backward_odata_ref, indices_bwd);
    return false;
  }
  return true;
}



#ifdef DEBUG_FFT
int main() {
#define FUNC c2c_many_2d_inplace_advanced
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

