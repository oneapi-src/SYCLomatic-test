// ===--- c2c_2d_inplace_make_plan.cu ------------------------*- CUDA -*---===//
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

bool c2c_2d_inplace_make_plan() {
  cufftHandle plan_fwd;
  cufftCreate(&plan_fwd);
  float2 forward_idata_h[2][5];
  set_value((float*)forward_idata_h, 20);

  float2* data_d;
  cudaMalloc(&data_d,sizeof(float2) * 10);
  cudaMemcpy(data_d, forward_idata_h, sizeof(float2) * 10, cudaMemcpyHostToDevice);

  size_t workSize;
  cufftMakePlan2d(plan_fwd, 2, 5, CUFFT_C2C, &workSize);
  cufftExecC2C(plan_fwd, data_d, data_d, CUFFT_FORWARD);
  cudaDeviceSynchronize();
  float2 forward_odata_h[10];
  cudaMemcpy(forward_odata_h, data_d, sizeof(float2) * 10, cudaMemcpyDeviceToHost);

  float2 forward_odata_ref[10];
  forward_odata_ref[0] =  float2{90,100};
  forward_odata_ref[1] =  float2{-23.7638,3.76382};
  forward_odata_ref[2] =  float2{-13.2492,-6.7508};
  forward_odata_ref[3] =  float2{-6.7508,-13.2492};
  forward_odata_ref[4] =  float2{3.76382,-23.7638};
  forward_odata_ref[5] =  float2{-50,-50};
  forward_odata_ref[6] =  float2{0,0};
  forward_odata_ref[7] =  float2{0,0};
  forward_odata_ref[8] =  float2{0,0};
  forward_odata_ref[9] =  float2{0,0};

  cufftDestroy(plan_fwd);

  if (!compare(forward_odata_ref, forward_odata_h, 10)) {
    std::cout << "forward_odata_h:" << std::endl;
    print_values(forward_odata_h, 10);
    std::cout << "forward_odata_ref:" << std::endl;
    print_values(forward_odata_ref, 10);

    cudaFree(data_d);

    return false;
  }

  cufftHandle plan_bwd;
  cufftCreate(&plan_bwd);
  cufftMakePlan2d(plan_bwd, 2, 5, CUFFT_C2C, &workSize);
  cufftExecC2C(plan_bwd, data_d, data_d, CUFFT_INVERSE);
  cudaDeviceSynchronize();
  float2 backward_odata_h[10];
  cudaMemcpy(backward_odata_h, data_d, sizeof(float2) * 10, cudaMemcpyDeviceToHost);

  float2 backward_odata_ref[10];
  backward_odata_ref[0] =  float2{0,10};
  backward_odata_ref[1] =  float2{20,30};
  backward_odata_ref[2] =  float2{40,50};
  backward_odata_ref[3] =  float2{60,70};
  backward_odata_ref[4] =  float2{80,90};
  backward_odata_ref[5] =  float2{100,110};
  backward_odata_ref[6] =  float2{120,130};
  backward_odata_ref[7] =  float2{140,150};
  backward_odata_ref[8] =  float2{160,170};
  backward_odata_ref[9] =  float2{180,190};

  cudaFree(data_d);
  cufftDestroy(plan_bwd);

  if (!compare(backward_odata_ref, backward_odata_h, 10)) {
    std::cout << "backward_odata_h:" << std::endl;
    print_values(backward_odata_h, 10);
    std::cout << "backward_odata_ref:" << std::endl;
    print_values(backward_odata_ref, 10);
    return false;
  }
  return true;
}


#ifdef DEBUG_FFT
int main() {
#define FUNC c2c_2d_inplace_make_plan 
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

