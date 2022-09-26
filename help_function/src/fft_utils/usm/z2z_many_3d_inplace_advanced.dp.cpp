// ===--- z2z_many_3d_inplace_advanced.dp.cpp -----------------*- C++ -*---===//
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
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+           -+--+          --+-----+            -----+
// |   c   |       |   c   |       |   c   |       |   c   |       |       |       |            |  |            |     |                 |
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+            n2 |            |     |                 |
// |   c   |       |   c   |       |   c   |       |   c   |       |       |       |            |  nembed2      |     |                 |
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+            |  |            |     |                 |
// |   c   |       |   c   |       |   c   |       |   c   |       |       |       |            |  |            |     |                 |
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+           -+--+            n1    |                 |
// |   c   |       |   c   |       |   c   |       |   c   |       |       |       |            |  |            |     |                 |
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+            n2 |            |     |                 a batch
// |   c   |       |   c   |       |   c   |       |   c   |       |       |       |            |  nembed2      |   nembed1             |
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+            |  |            |     |                 |
// |   c   |       |   c   |       |   c   |       |   c   |       |       |       |            |  |            |     |                 |
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+           -+--+          --+-----+            -----+
// |_______________________________n3______________________________|               |
// |____________________________________nembed3____________________________________|
// output
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+           -+--+          --+-----+            -----+
// |   c   |       |   c   |       |   c   |       |   c   |       |       |       |            |  |            |     |                 |
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+            n2 |            |     |                 |
// |   c   |       |   c   |       |   c   |       |   c   |       |       |       |            |  nembed2      |     |                 |
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+            |  |            |     |                 |
// |   c   |       |   c   |       |   c   |       |   c   |       |       |       |            |  |            |     |                 |
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+           -+--+            n1    |                 |
// |   c   |       |   c   |       |   c   |       |   c   |       |       |       |            |  |            |     |                 |
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+            n2 |            |     |                 a batch
// |   c   |       |   c   |       |   c   |       |   c   |       |       |       |            |  nembed2      |   nembed1             |
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+            |  |            |     |                 |
// |   c   |       |   c   |       |   c   |       |   c   |       |       |       |            |  |            |     |                 |
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+           -+--+          --+-----+            -----+
// |_______________________________n3______________________________|               |
// |____________________________________nembed3____________________________________|
bool z2z_many_3d_inplace_advanced() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  dpct::fft::fft_engine *plan_fwd;
  plan_fwd = dpct::fft::fft_engine::create();
  sycl::double2 forward_idata_h[120];
  std::memset(forward_idata_h, 0, sizeof(sycl::double2) * 120);
  forward_idata_h[0] = sycl::double2{0, 1};
  forward_idata_h[2] = sycl::double2{2, 3};
  forward_idata_h[4] = sycl::double2{4, 5};
  forward_idata_h[6] = sycl::double2{6, 7};
  forward_idata_h[10] = sycl::double2{8, 9};
  forward_idata_h[12] = sycl::double2{10, 11};
  forward_idata_h[14] = sycl::double2{12, 13};
  forward_idata_h[16] = sycl::double2{14, 15};
  forward_idata_h[20] = sycl::double2{16, 17};
  forward_idata_h[22] = sycl::double2{18, 19};
  forward_idata_h[24] = sycl::double2{20, 21};
  forward_idata_h[26] = sycl::double2{22, 23};
  forward_idata_h[30] = sycl::double2{24, 25};
  forward_idata_h[32] = sycl::double2{26, 27};
  forward_idata_h[34] = sycl::double2{28, 29};
  forward_idata_h[36] = sycl::double2{30, 31};
  forward_idata_h[40] = sycl::double2{32, 33};
  forward_idata_h[42] = sycl::double2{34, 35};
  forward_idata_h[44] = sycl::double2{36, 37};
  forward_idata_h[46] = sycl::double2{38, 39};
  forward_idata_h[50] = sycl::double2{40, 41};
  forward_idata_h[52] = sycl::double2{42, 43};
  forward_idata_h[54] = sycl::double2{44, 45};
  forward_idata_h[56] = sycl::double2{46, 47};
  std::memcpy(forward_idata_h + 60, forward_idata_h,
              sizeof(sycl::double2) * 60);

  sycl::double2 *data_d;
  data_d = sycl::malloc_device<sycl::double2>(120, q_ct1);
  q_ct1.memcpy(data_d, forward_idata_h, sizeof(sycl::double2) * 120).wait();

  size_t workSize;
  long long int n[3] = {2, 3, 4};
  long long int inembed[3] = {2, 3, 5};
  long long int onembed[3] = {2, 3, 5};
  plan_fwd->commit(&q_ct1, 3, n, inembed, 2, 60, onembed, 2, 60,
                   dpct::fft::fft_type::complex_double_to_complex_double, 2,
                   nullptr);
  plan_fwd->compute<sycl::double2, sycl::double2>(
      data_d, data_d, dpct::fft::fft_direction::forward);
  dev_ct1.queues_wait_and_throw();
  sycl::double2 forward_odata_h[120];
  q_ct1.memcpy(forward_odata_h, data_d, sizeof(sycl::double2) * 120).wait();

  sycl::double2 forward_odata_ref[120];
  forward_odata_ref[0] = sycl::double2{552, 576};
  forward_odata_ref[2] = sycl::double2{-48, 0};
  forward_odata_ref[4] = sycl::double2{-24, -24};
  forward_odata_ref[6] = sycl::double2{0, -48};
  forward_odata_ref[10] = sycl::double2{-151.426, -40.5744};
  forward_odata_ref[12] = sycl::double2{0, 0};
  forward_odata_ref[14] = sycl::double2{0, 0};
  forward_odata_ref[16] = sycl::double2{0, 0};
  forward_odata_ref[20] = sycl::double2{-40.5744, -151.426};
  forward_odata_ref[22] = sycl::double2{0, 0};
  forward_odata_ref[24] = sycl::double2{0, 0};
  forward_odata_ref[26] = sycl::double2{0, 0};
  forward_odata_ref[30] = sycl::double2{-288, -288};
  forward_odata_ref[32] = sycl::double2{0, 0};
  forward_odata_ref[34] = sycl::double2{0, 0};
  forward_odata_ref[36] = sycl::double2{0, 0};
  forward_odata_ref[40] = sycl::double2{0, 0};
  forward_odata_ref[42] = sycl::double2{0, 0};
  forward_odata_ref[44] = sycl::double2{0, 0};
  forward_odata_ref[46] = sycl::double2{0, 0};
  forward_odata_ref[50] = sycl::double2{0, 0};
  forward_odata_ref[52] = sycl::double2{0, 0};
  forward_odata_ref[54] = sycl::double2{0, 0};
  forward_odata_ref[56] = sycl::double2{0, 0};
  std::memcpy(forward_odata_ref + 60, forward_odata_ref,
              60 * sizeof(sycl::double2));

  dpct::fft::fft_engine::destroy(plan_fwd);

  std::vector<int> indices = {0, 2, 4, 6,
                              10, 12, 14, 16,
                              20, 22, 24, 26,
                              30, 32, 34, 36,
                              40, 42, 44, 46,
                              50, 52, 54, 56,
                              60, 62, 64, 66,
                              70, 72, 74, 76,
                              80, 82, 84, 86,
                              90, 92, 94, 96,
                              100, 102, 104, 106,
                              110, 112, 114, 116
                              };
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
  plan_bwd->commit(&q_ct1, 3, n, onembed, 2, 60, inembed, 2, 60,
                   dpct::fft::fft_type::complex_double_to_complex_double, 2,
                   nullptr);
  plan_bwd->compute<sycl::double2, sycl::double2>(
      data_d, data_d, dpct::fft::fft_direction::backward);
  dev_ct1.queues_wait_and_throw();
  sycl::double2 backward_odata_h[120];
  q_ct1.memcpy(backward_odata_h, data_d, sizeof(sycl::double2) * 120).wait();

  sycl::double2 backward_odata_ref[120];
  backward_odata_ref[0] = sycl::double2{0, 24};
  backward_odata_ref[2] = sycl::double2{48, 72};
  backward_odata_ref[4] = sycl::double2{96, 120};
  backward_odata_ref[6] = sycl::double2{144, 168};
  backward_odata_ref[10] = sycl::double2{192, 216};
  backward_odata_ref[12] = sycl::double2{240, 264};
  backward_odata_ref[14] = sycl::double2{288, 312};
  backward_odata_ref[16] = sycl::double2{336, 360};
  backward_odata_ref[20] = sycl::double2{384, 408};
  backward_odata_ref[22] = sycl::double2{432, 456};
  backward_odata_ref[24] = sycl::double2{480, 504};
  backward_odata_ref[26] = sycl::double2{528, 552};
  backward_odata_ref[30] = sycl::double2{576, 600};
  backward_odata_ref[32] = sycl::double2{624, 648};
  backward_odata_ref[34] = sycl::double2{672, 696};
  backward_odata_ref[36] = sycl::double2{720, 744};
  backward_odata_ref[40] = sycl::double2{768, 792};
  backward_odata_ref[42] = sycl::double2{816, 840};
  backward_odata_ref[44] = sycl::double2{864, 888};
  backward_odata_ref[46] = sycl::double2{912, 936};
  backward_odata_ref[50] = sycl::double2{960, 984};
  backward_odata_ref[52] = sycl::double2{1008, 1032};
  backward_odata_ref[54] = sycl::double2{1056, 1080};
  backward_odata_ref[56] = sycl::double2{1104, 1128};
  std::memcpy(backward_odata_ref + 60, backward_odata_ref,
              60 * sizeof(sycl::double2));

  sycl::free(data_d, q_ct1);
  dpct::fft::fft_engine::destroy(plan_bwd);

  std::vector<int> bwd_indices = {0, 2, 4, 6,
                                10, 12, 14, 16,
                                20, 22, 24, 26,
                                30, 32, 34, 36,
                                40, 42, 44, 46,
                                50, 52, 54, 56,
                                60, 62, 64, 66,
                                70, 72, 74, 76,
                                80, 82, 84, 86,
                                90, 92, 94, 96,
                                100, 102, 104, 106,
                                110, 112, 114, 116
                                };
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
#define FUNC z2z_many_3d_inplace_advanced
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

