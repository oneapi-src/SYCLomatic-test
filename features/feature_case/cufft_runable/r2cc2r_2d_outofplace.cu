// ===--- r2cc2r_2d_outofplace.cu ----------------------------*- CUDA -*---===//
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

bool r2cc2r_2d_outofplace() {
  cufftHandle plan_fwd;
  float forward_idata_h[4][5];
  set_value((float*)forward_idata_h, 20);

  float* forward_idata_d;
  float2* forward_odata_d;
  float* backward_odata_d;
  cudaMalloc(&forward_idata_d, sizeof(float) * 20);
  cudaMalloc(&forward_odata_d, sizeof(float2) * (5/2+1) * 4);
  cudaMalloc(&backward_odata_d, sizeof(float) * 20);
  cudaMemcpy(forward_idata_d, forward_idata_h, sizeof(float) * 20, cudaMemcpyHostToDevice);

  cufftPlan2d(&plan_fwd, 4, 5, CUFFT_R2C);
  cufftExecR2C(plan_fwd, forward_idata_d, forward_odata_d);
  cudaDeviceSynchronize();
  float2 forward_odata_h[12];
  cudaMemcpy(forward_odata_h, forward_odata_d, sizeof(float2) * (5/2+1) * 4, cudaMemcpyDeviceToHost);

  float2 forward_odata_ref[12];
  forward_odata_ref[0] =  float2{190,0};
  forward_odata_ref[1] =  float2{-10,13.7638};
  forward_odata_ref[2] =  float2{-10,3.2492};
  forward_odata_ref[3] =  float2{-50,50};
  forward_odata_ref[4] =  float2{0,0};
  forward_odata_ref[5] =  float2{0,0};
  forward_odata_ref[6] =  float2{-50,0};
  forward_odata_ref[7] =  float2{0,0};
  forward_odata_ref[8] =  float2{0,0};
  forward_odata_ref[9] =  float2{-50,-50};
  forward_odata_ref[10] = float2{0,0};
  forward_odata_ref[11] = float2{0,0};

  cufftDestroy(plan_fwd);

  if (!compare(forward_odata_ref, forward_odata_h, 12)) {
    std::cout << "forward_odata_h:" << std::endl;
    print_values(forward_odata_h, 12);
    std::cout << "forward_odata_ref:" << std::endl;
    print_values(forward_odata_ref, 12);

    cudaFree(forward_idata_d);
    cudaFree(forward_odata_d);
    cudaFree(backward_odata_d);

    return false;
  }

  cufftHandle plan_bwd;
  cufftPlan2d(&plan_bwd, 4, 5, CUFFT_C2R);
  cufftExecC2R(plan_bwd, forward_odata_d, backward_odata_d);
  cudaDeviceSynchronize();
  float backward_odata_h[20];
  cudaMemcpy(backward_odata_h, backward_odata_d, sizeof(float) * 20, cudaMemcpyDeviceToHost);

  float backward_odata_ref[20];
  backward_odata_ref[0] =  0;
  backward_odata_ref[1] =  20;
  backward_odata_ref[2] =  40;
  backward_odata_ref[3] =  60;
  backward_odata_ref[4] =  80;
  backward_odata_ref[5] =  100;
  backward_odata_ref[6] =  120;
  backward_odata_ref[7] =  140;
  backward_odata_ref[8] =  160;
  backward_odata_ref[9] =  180;
  backward_odata_ref[10] = 200;
  backward_odata_ref[11] = 220;
  backward_odata_ref[12] = 240;
  backward_odata_ref[13] = 260;
  backward_odata_ref[14] = 280;
  backward_odata_ref[15] = 300;
  backward_odata_ref[16] = 320;
  backward_odata_ref[17] = 340;
  backward_odata_ref[18] = 360;
  backward_odata_ref[19] = 380;

  cudaFree(forward_idata_d);
  cudaFree(forward_odata_d);
  cudaFree(backward_odata_d);

  cufftDestroy(plan_bwd);

  if (!compare(backward_odata_ref, backward_odata_h, 20)) {
    std::cout << "backward_odata_h:" << std::endl;
    print_values(backward_odata_h, 20);
    std::cout << "backward_odata_ref:" << std::endl;
    print_values(backward_odata_ref, 20);
    return false;
  }
  return true;
}


#ifdef DEBUG_FFT
int main() {
#define FUNC r2cc2r_2d_outofplace
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

