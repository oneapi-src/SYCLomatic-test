// ===--- r2cc2r_2d_inplace_make_plan.cu ---------------------*- CUDA -*---===//
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


bool r2cc2r_2d_inplace_make_plan() {
  cufftHandle plan_fwd;
  cufftCreate(&plan_fwd);
  float forward_idata_h[24];
  set_value(forward_idata_h, 4, 5, 6);

  float* data_d;
  cudaMalloc(&data_d, sizeof(float) * 24);
  cudaMemcpy(data_d, forward_idata_h, sizeof(float) * 24, cudaMemcpyHostToDevice);

  size_t workSize;
  cufftMakePlan2d(plan_fwd, 4, 5, CUFFT_R2C, &workSize);
  cufftExecR2C(plan_fwd, data_d, (float2*)data_d);
  cudaDeviceSynchronize();
  float2 forward_odata_h[12];
  cudaMemcpy(forward_odata_h, data_d, sizeof(float) * 24, cudaMemcpyDeviceToHost);

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

    cudaFree(data_d);

    return false;
  }

  cufftHandle plan_bwd;
  cufftCreate(&plan_bwd);
  cufftMakePlan2d(plan_bwd, 4, 5, CUFFT_C2R, &workSize);
  cufftExecC2R(plan_bwd, (float2*)data_d, data_d);
  cudaDeviceSynchronize();
  float backward_odata_h[24];
  cudaMemcpy(backward_odata_h, data_d, sizeof(float) * 24, cudaMemcpyDeviceToHost);

  float backward_odata_ref[24];
  backward_odata_ref[0] =  0;
  backward_odata_ref[1] =  20;
  backward_odata_ref[2] =  40;
  backward_odata_ref[3] =  60;
  backward_odata_ref[4] =  80;
  backward_odata_ref[5] =  3.2492;
  backward_odata_ref[6] =  100;
  backward_odata_ref[7] =  120;
  backward_odata_ref[8] =  140;
  backward_odata_ref[9] =  160;
  backward_odata_ref[10] = 180;
  backward_odata_ref[11] = 3.2492;
  backward_odata_ref[12] = 200;
  backward_odata_ref[13] = 220;
  backward_odata_ref[14] = 240;
  backward_odata_ref[15] = 260;
  backward_odata_ref[16] = 280;
  backward_odata_ref[17] = 3.2492;
  backward_odata_ref[18] = 300;
  backward_odata_ref[19] = 320;
  backward_odata_ref[20] = 340;
  backward_odata_ref[21] = 360;
  backward_odata_ref[22] = 380;
  backward_odata_ref[23] = 3.2492;

  cudaFree(data_d);
  cufftDestroy(plan_bwd);

  std::vector<int> indices = {0, 1, 2, 3, 4,
                              6, 7, 8, 9, 10,
                              12, 13, 14, 15 ,16,
                              18, 19, 20, 21, 22};
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
#define FUNC r2cc2r_2d_inplace_make_plan
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

