// ===--- r2cc2r_1d_inplace.cu -------------------------------*- CUDA -*---===//
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



bool r2cc2r_1d_inplace() {
  cufftHandle plan_fwd;
  float forward_idata_h[16];
  set_value(forward_idata_h, 7);
  set_value(forward_idata_h + 8, 7);

  float* data_d;
  cudaMalloc(&data_d, sizeof(float) * 16);
  cudaMemcpy(data_d, forward_idata_h, sizeof(float) * 16, cudaMemcpyHostToDevice);

  cufftPlan1d(&plan_fwd, 7, CUFFT_R2C, 2);
  cufftExecR2C(plan_fwd, data_d, (float2*)data_d);
  cudaDeviceSynchronize();
  float2 forward_odata_h[8];
  cudaMemcpy(forward_odata_h, data_d, sizeof(float) * 16, cudaMemcpyDeviceToHost);

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

    cudaFree(data_d);

    return false;
  }

  cufftHandle plan_bwd;
  cufftPlan1d(&plan_bwd, 7, CUFFT_C2R, 2);
  cufftExecC2R(plan_bwd, (float2*)data_d, data_d);
  cudaDeviceSynchronize();
  float backward_odata_h[16];
  cudaMemcpy(backward_odata_h, data_d, sizeof(float) * 16, cudaMemcpyDeviceToHost);

  float backward_odata_ref[16];
  backward_odata_ref[0] = 0;
  backward_odata_ref[1] = 7;
  backward_odata_ref[2] = 14;
  backward_odata_ref[3] = 21;
  backward_odata_ref[4] = 28;
  backward_odata_ref[5] = 35;
  backward_odata_ref[6] = 42;
  backward_odata_ref[7] = 0.798852;
  backward_odata_ref[8] = 0;
  backward_odata_ref[9] = 7;
  backward_odata_ref[10] = 14;
  backward_odata_ref[11] = 21;
  backward_odata_ref[12] = 28;
  backward_odata_ref[13] = 35;
  backward_odata_ref[14] = 42;
  backward_odata_ref[15] = 0.798852;

  cudaFree(data_d);
  cufftDestroy(plan_bwd);

  std::vector<int> indices = {0, 1, 2,  3,  4,  5,  6,
                              8, 9, 10, 11, 12, 13, 14};
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
#define FUNC r2cc2r_1d_inplace
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

