// ===--- r2cc2r_many_1d_outofplace_basic.cu -----------------*- CUDA -*---===//
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

bool r2cc2r_many_1d_outofplace_basic() {
  cufftHandle plan_fwd;
  float forward_idata_h[20];
  set_value(forward_idata_h, 10);
  set_value(forward_idata_h + 10, 10);

  float* forward_idata_d;
  float2* forward_odata_d;
  float* backward_odata_d;
  cudaMalloc(&forward_idata_d, sizeof(float) * 20);
  cudaMalloc(&forward_odata_d, 2 * sizeof(float2) * (10/2+1));
  cudaMalloc(&backward_odata_d, sizeof(float) * 20);
  cudaMemcpy(forward_idata_d, forward_idata_h, sizeof(float) * 20, cudaMemcpyHostToDevice);

  int n[1] = {10};
  cufftPlanMany(&plan_fwd, 1, n, nullptr, 0, 0, nullptr, 0, 0, CUFFT_R2C, 2);
  cufftExecR2C(plan_fwd, forward_idata_d, forward_odata_d);
  cudaDeviceSynchronize();
  float2 forward_odata_h[12];
  cudaMemcpy(forward_odata_h, forward_odata_d, 2 * sizeof(float2) * (10/2+1), cudaMemcpyDeviceToHost);

  float2 forward_odata_ref[12];
  forward_odata_ref[0] =  float2{45,0};
  forward_odata_ref[1] =  float2{-5,15.3884};
  forward_odata_ref[2] =  float2{-5,6.88191};
  forward_odata_ref[3] =  float2{-5,3.63271};
  forward_odata_ref[4] =  float2{-5,1.6246};
  forward_odata_ref[5] =  float2{-5,0};
  forward_odata_ref[6] =  float2{45,0};
  forward_odata_ref[7] =  float2{-5,15.3884};
  forward_odata_ref[8] =  float2{-5,6.88191};
  forward_odata_ref[9] =  float2{-5,3.63271};
  forward_odata_ref[10] = float2{-5,1.6246};
  forward_odata_ref[11] = float2{-5,0};

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
  cufftPlanMany(&plan_bwd, 1, n, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2R, 2);
  cufftExecC2R(plan_bwd, forward_odata_d, backward_odata_d);
  cudaDeviceSynchronize();
  float backward_odata_h[20];
  cudaMemcpy(backward_odata_h, backward_odata_d, sizeof(float) * 20, cudaMemcpyDeviceToHost);

  float backward_odata_ref[20];
  backward_odata_ref[0] =  0;
  backward_odata_ref[1] =  10;
  backward_odata_ref[2] =  20;
  backward_odata_ref[3] =  30;
  backward_odata_ref[4] =  40;
  backward_odata_ref[5] =  50;
  backward_odata_ref[6] =  60;
  backward_odata_ref[7] =  70;
  backward_odata_ref[8] =  80;
  backward_odata_ref[9] =  90;
  backward_odata_ref[10] = 0;
  backward_odata_ref[11] = 10;
  backward_odata_ref[12] = 20;
  backward_odata_ref[13] = 30;
  backward_odata_ref[14] = 40;
  backward_odata_ref[15] = 50;
  backward_odata_ref[16] = 60;
  backward_odata_ref[17] = 70;
  backward_odata_ref[18] = 80;
  backward_odata_ref[19] = 90;

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
#define FUNC r2cc2r_many_1d_outofplace_basic
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

