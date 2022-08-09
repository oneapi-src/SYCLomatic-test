// ===--- d2zz2d_3d_outofplace_make_plan.cu ------------------*- CUDA -*---===//
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


bool d2zz2d_3d_outofplace_make_plan() {
  cufftHandle plan_fwd;
  cufftCreate(&plan_fwd);
  double forward_idata_h[2][3][5];
  set_value((double*)forward_idata_h, 30);

  double* forward_idata_d;
  double2* forward_odata_d;
  double* backward_odata_d;
  cudaMalloc(&forward_idata_d, sizeof(double) * 30);
  cudaMalloc(&forward_odata_d, sizeof(double2) * (5/2+1) * 2 * 3);
  cudaMalloc(&backward_odata_d, sizeof(double) * 30);
  cudaMemcpy(forward_idata_d, forward_idata_h, sizeof(double) * 30, cudaMemcpyHostToDevice);

  size_t workSize;
  cufftMakePlan3d(plan_fwd, 2 ,3 ,5, CUFFT_D2Z, &workSize);
  cufftExecD2Z(plan_fwd, forward_idata_d, forward_odata_d);
  cudaDeviceSynchronize();
  double2 forward_odata_h[18];
  cudaMemcpy(forward_odata_h, forward_odata_d, sizeof(double2) * (5/2+1) * 2 * 3, cudaMemcpyDeviceToHost);

  double2 forward_odata_ref[18];
  forward_odata_ref[0] =  double2{435,0};
  forward_odata_ref[1] =  double2{-15,20.6457};
  forward_odata_ref[2] =  double2{-15,4.8738};
  forward_odata_ref[3] =  double2{-75,43.3013};
  forward_odata_ref[4] =  double2{0,0};
  forward_odata_ref[5] =  double2{0,0};
  forward_odata_ref[6] =  double2{-75,-43.3013};
  forward_odata_ref[7] =  double2{0,0};
  forward_odata_ref[8] =  double2{0,0};
  forward_odata_ref[9] =  double2{-225,0};
  forward_odata_ref[10] = double2{0,0};
  forward_odata_ref[11] = double2{0,0};
  forward_odata_ref[12] = double2{0,0};
  forward_odata_ref[13] = double2{0,0};
  forward_odata_ref[14] = double2{0,0};
  forward_odata_ref[15] = double2{0,0};
  forward_odata_ref[16] = double2{0,0};
  forward_odata_ref[17] = double2{0,0};

  cufftDestroy(plan_fwd);

  if (!compare(forward_odata_ref, forward_odata_h, 18)) {
    std::cout << "forward_odata_h:" << std::endl;
    print_values(forward_odata_h, 18);
    std::cout << "forward_odata_ref:" << std::endl;
    print_values(forward_odata_ref, 18);

    cudaFree(forward_idata_d);
    cudaFree(forward_odata_d);
    cudaFree(backward_odata_d);

    return false;
  }

  cufftHandle plan_bwd;
  cufftCreate(&plan_bwd);
  cufftMakePlan3d(plan_bwd, 2 ,3 ,5, CUFFT_Z2D, &workSize);
  cufftExecZ2D(plan_bwd, forward_odata_d, backward_odata_d);
  cudaDeviceSynchronize();
  double backward_odata_h[30];
  cudaMemcpy(backward_odata_h, backward_odata_d, sizeof(double) * 30, cudaMemcpyDeviceToHost);

  double backward_odata_ref[30];
  backward_odata_ref[0] =  0;
  backward_odata_ref[1] =  30;
  backward_odata_ref[2] =  60;
  backward_odata_ref[3] =  90;
  backward_odata_ref[4] =  120;
  backward_odata_ref[5] =  150;
  backward_odata_ref[6] =  180;
  backward_odata_ref[7] =  210;
  backward_odata_ref[8] =  240;
  backward_odata_ref[9] =  270;
  backward_odata_ref[10] = 300;
  backward_odata_ref[11] = 330;
  backward_odata_ref[12] = 360;
  backward_odata_ref[13] = 390;
  backward_odata_ref[14] = 420;
  backward_odata_ref[15] = 450;
  backward_odata_ref[16] = 480;
  backward_odata_ref[17] = 510;
  backward_odata_ref[18] = 540;
  backward_odata_ref[19] = 570;
  backward_odata_ref[20] = 600;
  backward_odata_ref[21] = 630;
  backward_odata_ref[22] = 660;
  backward_odata_ref[23] = 690;
  backward_odata_ref[24] = 720;
  backward_odata_ref[25] = 750;
  backward_odata_ref[26] = 780;
  backward_odata_ref[27] = 810;
  backward_odata_ref[28] = 840;
  backward_odata_ref[29] = 870;

  cudaFree(forward_idata_d);
  cudaFree(forward_odata_d);
  cudaFree(backward_odata_d);

  cufftDestroy(plan_bwd);

  if (!compare(backward_odata_ref, backward_odata_h, 30)) {
    std::cout << "backward_odata_h:" << std::endl;
    print_values(backward_odata_h, 30);
    std::cout << "backward_odata_ref:" << std::endl;
    print_values(backward_odata_ref, 30);
    return false;
  }
  return true;
}

#ifdef DEBUG_FFT
int main() {
#define FUNC d2zz2d_3d_outofplace_make_plan
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

