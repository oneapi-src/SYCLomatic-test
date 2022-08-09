// ===--- c2c_1d_inplace.cu ----------------------------------*- CUDA -*---===//
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


bool c2c_1d_inplace() {
  cufftHandle plan_fwd;
  float2 forward_idata_h[14];
  set_value((float*)forward_idata_h, 14);
  set_value((float*)forward_idata_h + 14, 14);

  float2* data_d;
  cudaMalloc(&data_d, 2 * sizeof(float2) * 7);
  cudaMemcpy(data_d, forward_idata_h, 2 * sizeof(float2) * 7, cudaMemcpyHostToDevice);

  cufftPlan1d(&plan_fwd, 7, CUFFT_C2C, 2);
  cufftExecC2C(plan_fwd, data_d, data_d, CUFFT_FORWARD);
  cudaDeviceSynchronize();
  float2 forward_odata_h[14];
  cudaMemcpy(forward_odata_h, data_d, 2 * sizeof(float2) * 7, cudaMemcpyDeviceToHost);

  float2 forward_odata_ref[14];
  forward_odata_ref[0] =  float2{42,49};
  forward_odata_ref[1] =  float2{-21.5356,7.53565};
  forward_odata_ref[2] =  float2{-12.5823,-1.41769};
  forward_odata_ref[3] =  float2{-8.5977,-5.4023};
  forward_odata_ref[4] =  float2{-5.4023,-8.5977};
  forward_odata_ref[5] =  float2{-1.41769,-12.5823};
  forward_odata_ref[6] =  float2{7.53565,-21.5356};
  forward_odata_ref[7] =  float2{42,49};
  forward_odata_ref[8] =  float2{-21.5356,7.53565};
  forward_odata_ref[9] =  float2{-12.5823,-1.41769};
  forward_odata_ref[10] = float2{-8.5977,-5.4023};
  forward_odata_ref[11] = float2{-5.4023,-8.5977};
  forward_odata_ref[12] = float2{-1.41769,-12.5823};
  forward_odata_ref[13] = float2{7.53565,-21.5356};

  cufftDestroy(plan_fwd);

  if (!compare(forward_odata_ref, forward_odata_h, 14)) {
    std::cout << "forward_odata_h:" << std::endl;
    print_values(forward_odata_h, 14);
    std::cout << "forward_odata_ref:" << std::endl;
    print_values(forward_odata_ref, 14);

    cudaFree(data_d);

    return false;
  }

  cufftHandle plan_bwd;
  cufftPlan1d(&plan_bwd, 7, CUFFT_C2C, 2);
  cufftExecC2C(plan_bwd, data_d, data_d, CUFFT_INVERSE);
  cudaDeviceSynchronize();
  float2 backward_odata_h[14];
  cudaMemcpy(backward_odata_h, data_d, 2 * sizeof(float2) * 7, cudaMemcpyDeviceToHost);

  float2 backward_odata_ref[14];
  backward_odata_ref[0] =  float2{0,7};
  backward_odata_ref[1] =  float2{14,21};
  backward_odata_ref[2] =  float2{28,35};
  backward_odata_ref[3] =  float2{42,49};
  backward_odata_ref[4] =  float2{56,63};
  backward_odata_ref[5] =  float2{70,77};
  backward_odata_ref[6] =  float2{84,91};
  backward_odata_ref[7] =  float2{0,7};
  backward_odata_ref[8] =  float2{14,21};
  backward_odata_ref[9] =  float2{28,35};
  backward_odata_ref[10] = float2{42,49};
  backward_odata_ref[11] = float2{56,63};
  backward_odata_ref[12] = float2{70,77};
  backward_odata_ref[13] = float2{84,91};

  cudaFree(data_d);
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
#define FUNC c2c_1d_inplace
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

