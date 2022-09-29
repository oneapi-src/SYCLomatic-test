// ===--- r2cc2r_1d_outofplace_make_plan.cu ------------------*- CUDA -*---===//
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


bool r2cc2r_1d_outofplace_make_plan() {
  cufftHandle plan_fwd;
  cufftCreate(&plan_fwd);
  float forward_idata_h[14];
  set_value(forward_idata_h, 7);
  set_value(forward_idata_h + 7, 7);

  float* forward_idata_d;
  float2* forward_odata_d;
  float* backward_odata_d;
  cudaMalloc(&forward_idata_d, 2 * sizeof(float) * 7);
  cudaMalloc(&forward_odata_d, 2 * sizeof(float2) * (7/2+1));
  cudaMalloc(&backward_odata_d, 2 * sizeof(float) * 7);
  cudaMemcpy(forward_idata_d, forward_idata_h, 2 * sizeof(float) * 7, cudaMemcpyHostToDevice);

  size_t workSize;
  cufftMakePlan1d(plan_fwd, 7, CUFFT_R2C, 2, &workSize);
  cufftExecR2C(plan_fwd, forward_idata_d, forward_odata_d);
  cudaDeviceSynchronize();
  float2 forward_odata_h[8];
  cudaMemcpy(forward_odata_h, forward_odata_d, 2 * sizeof(float2) * (7/2+1), cudaMemcpyDeviceToHost);

  float2 forward_odata_ref[8];
  forward_odata_ref[0] = float2{21,0};
  forward_odata_ref[1] = float2{-3.5,7.26783};
  forward_odata_ref[2] = float2{-3.5,2.79116};
  forward_odata_ref[3] = float2{-3.5,0.798852};
  forward_odata_ref[4] = float2{21,0};
  forward_odata_ref[5] = float2{-3.5,7.26783};
  forward_odata_ref[6] = float2{-3.5,2.79116};
  forward_odata_ref[7] = float2{-3.5,0.798852};

  cufftDestroy(plan_fwd);

  if (!compare(forward_odata_ref, forward_odata_h, 8)) {
    std::cout << "forward_odata_h:" << std::endl;
    print_values(forward_odata_h, 8);
    std::cout << "forward_odata_ref:" << std::endl;
    print_values(forward_odata_ref, 8);

    cudaFree(forward_idata_d);
    cudaFree(forward_odata_d);
    cudaFree(backward_odata_d);

    return false;
  }

  cufftHandle plan_bwd;
  cufftCreate(&plan_bwd);
  cufftMakePlan1d(plan_bwd, 7, CUFFT_C2R, 2, &workSize);
  cufftExecC2R(plan_bwd, forward_odata_d, backward_odata_d);
  cudaDeviceSynchronize();
  float backward_odata_h[14];
  cudaMemcpy(backward_odata_h, backward_odata_d, 2 * sizeof(float) * 7, cudaMemcpyDeviceToHost);

  float backward_odata_ref[14];
  backward_odata_ref[0] = 0;
  backward_odata_ref[1] = 7;
  backward_odata_ref[2] = 14;
  backward_odata_ref[3] = 21;
  backward_odata_ref[4] = 28;
  backward_odata_ref[5] = 35;
  backward_odata_ref[6] = 42;
  backward_odata_ref[7] = 0;
  backward_odata_ref[8] = 7;
  backward_odata_ref[9] = 14;
  backward_odata_ref[10] = 21;
  backward_odata_ref[11] = 28;
  backward_odata_ref[12] = 35;
  backward_odata_ref[13] = 42;

  cudaFree(forward_idata_d);
  cudaFree(forward_odata_d);
  cudaFree(backward_odata_d);

  cufftDestroy(plan_bwd);

  if (!compare(backward_odata_ref, backward_odata_h, 14)) {
    std::cout << "backward_odata_h:" << std::endl;
    print_values(backward_odata_h, 14);
    std::cout << "backward_odata_ref:" << std::endl;
    print_values(backward_odata_ref, 14);
    return false;
  }
  return true;
}


#ifdef DEBUG_FFT
int main() {
#define FUNC r2cc2r_1d_outofplace_make_plan
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

