// ===--- d2zz2d_many_3d_inplace_advanced.cu -----------------*- CUDA -*---===//
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
// +---+---+---+---+
// | r | r | r |   |
// +---+---+---+---+
// | r | r | r |   |
// +---+---+---+---+
// | r | r | r |   |
// +---+---+---+---+
// | r | r | r |   |
// +---+---+---+---+
// | r | r | r |   |
// +---+---+---+---+
// | r | r | r |   |
// +---+---+---+---+
// | r | r | r |   |
// +---+---+---+---+
// | r | r | r |   |
// +---+---+---+---+
// output
// +---+---+---+---+           -+ -+          ---+-------+     -----------+
// |   c   |   c   |            |  |             |       |                |
// +---+---+---+---+            n2 |             |       |                |
// |   c   |   c   |            |  nembed2       |       |                |
// +---+---+---+---+           -+--+             |       |                |
// |   c   |   c   |            |  |             |       |                |
// +---+---+---+---+            n2 |             |       |                |
// |   c   |   c   |            |  nembed2       |       |                |
// +---+---+---+---+           -+--+             n1      |                a batch
// |   c   |   c   |            |  |             |       |                |
// +---+---+---+---+            n2 |             |    nembed1(except      |
// |   c   |   c   |            |  nembed2       |    the last element)   |
// +---+---+---+---+           -+--+             |       |                |
// |   c   |   c   |            |  |             |       |                |
// +---+---+---+---+            n2 |             |       |                |
// |   c   |   c   |            |  nembed2       |       |                |
// +---+---+---+---+           -+--+          ---+-------+     -----------+
// |______n3_______|
// |____nembed3____|
bool d2zz2d_many_3d_inplace_advanced() {
  cufftHandle plan_fwd;
  cufftCreate(&plan_fwd);
  double forward_idata_h[64];
  std::memset(forward_idata_h, 0, sizeof(double) * 64);
  forward_idata_h[0] = 0;
  forward_idata_h[1] = 1;
  forward_idata_h[2] = 2;
  forward_idata_h[4] = 3;
  forward_idata_h[5] = 4;
  forward_idata_h[6] = 5;
  forward_idata_h[8] = 6;
  forward_idata_h[9] = 7;
  forward_idata_h[10] = 8;
  forward_idata_h[12] = 9;
  forward_idata_h[13] = 10;
  forward_idata_h[14] = 11;
  forward_idata_h[16] = 12;
  forward_idata_h[17] = 13;
  forward_idata_h[18] = 14;
  forward_idata_h[20] = 15;
  forward_idata_h[21] = 16;
  forward_idata_h[22] = 17;
  forward_idata_h[24] = 18;
  forward_idata_h[25] = 19;
  forward_idata_h[26] = 20;
  forward_idata_h[28] = 21;
  forward_idata_h[29] = 22;
  forward_idata_h[30] = 23;
  std::memcpy(forward_idata_h + 32, forward_idata_h, sizeof(double) * 32);

  double* data_d;
  cudaMalloc(&data_d, sizeof(double) * 64);
  cudaMemcpy(data_d, forward_idata_h, sizeof(double) * 64, cudaMemcpyHostToDevice);

  size_t workSize;
  long long int n[3] = {4, 2, 3};
  long long int inembed[3] = {4, 2, 4};
  long long int onembed[3] = {4, 2, 2};
  cufftMakePlanMany64(plan_fwd, 3, n, inembed, 1, 32, onembed, 1, 16, CUFFT_D2Z, 2, &workSize);
  cufftExecD2Z(plan_fwd, data_d, (double2*)data_d);
  cudaDeviceSynchronize();
  double2 forward_odata_h[32];
  cudaMemcpy(forward_odata_h, data_d, sizeof(double) * 64, cudaMemcpyDeviceToHost);

  double2 forward_odata_ref[32] = {
    double2{276,0},
    double2{-12,6.9282},
    double2{-36,0},
    double2{0,0},
    double2{-72,72},
    double2{0,0},
    double2{0,0},
    double2{0,0},
    double2{-72,0},
    double2{0,0},
    double2{0,0},
    double2{0,0},
    double2{-72,-72},
    double2{0,0},
    double2{0,0},
    double2{0,0},
    double2{276,0},
    double2{-12,6.9282},
    double2{-36,0},
    double2{0,0},
    double2{-72,72},
    double2{0,0},
    double2{0,0},
    double2{0,0},
    double2{-72,0},
    double2{0,0},
    double2{0,0},
    double2{0,0},
    double2{-72,-72},
    double2{0,0},
    double2{0,0},
    double2{0,0}
  };

  cufftDestroy(plan_fwd);

  if (!compare(forward_odata_ref, forward_odata_h, 32)) {
    std::cout << "forward_odata_h:" << std::endl;
    print_values(forward_odata_h, 32);
    std::cout << "forward_odata_ref:" << std::endl;
    print_values(forward_odata_ref, 32);

    cudaFree(data_d);

    return false;
  }

  cufftHandle plan_bwd;
  cufftCreate(&plan_bwd);
  cufftMakePlanMany64(plan_bwd, 3, n, onembed, 1, 16, inembed, 1, 32, CUFFT_Z2D, 2, &workSize);
  cufftExecZ2D(plan_bwd, (double2*)data_d, data_d);
  cudaDeviceSynchronize();
  double backward_odata_h[64];
  cudaMemcpy(backward_odata_h, data_d, sizeof(double) * 64, cudaMemcpyDeviceToHost);

  double backward_odata_ref[64];
  backward_odata_ref[0]  = 0;
  backward_odata_ref[1]  = 24;
  backward_odata_ref[2]  = 48;
  backward_odata_ref[4]  = 72;
  backward_odata_ref[5]  = 96;
  backward_odata_ref[6]  = 120;
  backward_odata_ref[8]  = 144;
  backward_odata_ref[9]  = 168;
  backward_odata_ref[10] = 192;
  backward_odata_ref[12] = 216;
  backward_odata_ref[13] = 240;
  backward_odata_ref[14] = 264;
  backward_odata_ref[16] = 288;
  backward_odata_ref[17] = 312;
  backward_odata_ref[18] = 336;
  backward_odata_ref[20] = 360;
  backward_odata_ref[21] = 384;
  backward_odata_ref[22] = 408;
  backward_odata_ref[24] = 432;
  backward_odata_ref[25] = 456;
  backward_odata_ref[26] = 480;
  backward_odata_ref[28] = 504;
  backward_odata_ref[29] = 528;
  backward_odata_ref[30] = 552;
  std::memcpy(backward_odata_ref + 32, backward_odata_ref, sizeof(double) * 32);
  cudaFree(data_d);
  cufftDestroy(plan_bwd);

  std::vector<int> indices = {0, 1, 2,
                              4, 5, 6,
                              8, 9, 10,
                              12, 13, 14,
                              16, 17, 18,
                              20, 21, 22,
                              24, 25, 26,
                              28, 29, 30,
                              32, 33, 34,
                              36, 37, 38,
                              40, 41, 42,
                              44, 45, 46,
                              48, 49, 50,
                              52, 53, 54,
                              56, 57, 58,
                              60, 61, 62
                              };
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
#define FUNC d2zz2d_many_3d_inplace_advanced
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

