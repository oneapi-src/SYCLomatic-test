// ===--- z2z_many_3d_inplace_advanced.cu --------------------*- CUDA -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===---------------------------------------------------------------------===//

#include "cufft.h"
#include "cufftXt.h"
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
  cufftHandle plan_fwd;
  cufftCreate(&plan_fwd);
  double2 forward_idata_h[120];
  std::memset(forward_idata_h, 0, sizeof(double2) * 120);
  forward_idata_h[0] = double2{0, 1};
  forward_idata_h[2] = double2{2, 3};
  forward_idata_h[4] = double2{4, 5};
  forward_idata_h[6] = double2{6, 7};
  forward_idata_h[10] = double2{8, 9};
  forward_idata_h[12] = double2{10, 11};
  forward_idata_h[14] = double2{12, 13};
  forward_idata_h[16] = double2{14, 15};
  forward_idata_h[20] = double2{16, 17};
  forward_idata_h[22] = double2{18, 19};
  forward_idata_h[24] = double2{20, 21};
  forward_idata_h[26] = double2{22, 23};
  forward_idata_h[30] = double2{24, 25};
  forward_idata_h[32] = double2{26, 27};
  forward_idata_h[34] = double2{28, 29};
  forward_idata_h[36] = double2{30, 31};
  forward_idata_h[40] = double2{32, 33};
  forward_idata_h[42] = double2{34, 35};
  forward_idata_h[44] = double2{36, 37};
  forward_idata_h[46] = double2{38, 39};
  forward_idata_h[50] = double2{40, 41};
  forward_idata_h[52] = double2{42, 43};
  forward_idata_h[54] = double2{44, 45};
  forward_idata_h[56] = double2{46, 47};
  std::memcpy(forward_idata_h + 60, forward_idata_h, sizeof(double2) * 60);

  double2* data_d;
  cudaMalloc(&data_d, sizeof(double2) * 120);
  cudaMemcpy(data_d, forward_idata_h, sizeof(double2) * 120, cudaMemcpyHostToDevice);

  size_t workSize;
  long long int n[3] = {2, 3, 4};
  long long int inembed[3] = {2, 3, 5};
  long long int onembed[3] = {2, 3, 5};
  cufftMakePlanMany64(plan_fwd, 3, n, inembed, 2, 60, onembed, 2, 60, CUFFT_Z2Z, 2, &workSize);
  cufftExecZ2Z(plan_fwd, data_d, data_d, CUFFT_FORWARD);
  cudaDeviceSynchronize();
  double2 forward_odata_h[120];
  cudaMemcpy(forward_odata_h, data_d, sizeof(double2) * 120, cudaMemcpyDeviceToHost);

  double2 forward_odata_ref[120];
  forward_odata_ref[0]  = double2{552,576};
  forward_odata_ref[2]  = double2{-48,0};
  forward_odata_ref[4]  = double2{-24,-24};
  forward_odata_ref[6]  = double2{0,-48};
  forward_odata_ref[10] = double2{-151.426,-40.5744};
  forward_odata_ref[12] = double2{0,0};
  forward_odata_ref[14] = double2{0,0};
  forward_odata_ref[16] = double2{0,0};
  forward_odata_ref[20] = double2{-40.5744,-151.426};
  forward_odata_ref[22] = double2{0,0};
  forward_odata_ref[24] = double2{0,0};
  forward_odata_ref[26] = double2{0,0};
  forward_odata_ref[30] = double2{-288,-288};
  forward_odata_ref[32] = double2{0,0};
  forward_odata_ref[34] = double2{0,0};
  forward_odata_ref[36] = double2{0,0};
  forward_odata_ref[40] = double2{0,0};
  forward_odata_ref[42] = double2{0,0};
  forward_odata_ref[44] = double2{0,0};
  forward_odata_ref[46] = double2{0,0};
  forward_odata_ref[50] = double2{0,0};
  forward_odata_ref[52] = double2{0,0};
  forward_odata_ref[54] = double2{0,0};
  forward_odata_ref[56] = double2{0,0};
  std::memcpy(forward_odata_ref + 60, forward_odata_ref, 60 * sizeof(double2));

  cufftDestroy(plan_fwd);

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

    cudaFree(data_d);

    return false;
  }

  cufftHandle plan_bwd;
  cufftCreate(&plan_bwd);
  cufftMakePlanMany64(plan_bwd, 3, n, onembed, 2, 60, inembed, 2, 60, CUFFT_Z2Z, 2, &workSize);
  cufftExecZ2Z(plan_bwd, data_d, data_d, CUFFT_INVERSE);
  cudaDeviceSynchronize();
  double2 backward_odata_h[120];
  cudaMemcpy(backward_odata_h, data_d, sizeof(double2) * 120, cudaMemcpyDeviceToHost);

  double2 backward_odata_ref[120];
  backward_odata_ref[0]  = double2{0,24};
  backward_odata_ref[2]  = double2{48,72};
  backward_odata_ref[4]  = double2{96,120};
  backward_odata_ref[6]  = double2{144,168};
  backward_odata_ref[10] = double2{192,216};
  backward_odata_ref[12] = double2{240,264};
  backward_odata_ref[14] = double2{288,312};
  backward_odata_ref[16] = double2{336,360};
  backward_odata_ref[20] = double2{384,408};
  backward_odata_ref[22] = double2{432,456};
  backward_odata_ref[24] = double2{480,504};
  backward_odata_ref[26] = double2{528,552};
  backward_odata_ref[30] = double2{576,600};
  backward_odata_ref[32] = double2{624,648};
  backward_odata_ref[34] = double2{672,696};
  backward_odata_ref[36] = double2{720,744};
  backward_odata_ref[40] = double2{768,792};
  backward_odata_ref[42] = double2{816,840};
  backward_odata_ref[44] = double2{864,888};
  backward_odata_ref[46] = double2{912,936};
  backward_odata_ref[50] = double2{960,984};
  backward_odata_ref[52] = double2{1008,1032};
  backward_odata_ref[54] = double2{1056,1080};
  backward_odata_ref[56] = double2{1104,1128};
  std::memcpy(backward_odata_ref + 60, backward_odata_ref, 60 * sizeof(double2));

  cudaFree(data_d);
  cufftDestroy(plan_bwd);

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

