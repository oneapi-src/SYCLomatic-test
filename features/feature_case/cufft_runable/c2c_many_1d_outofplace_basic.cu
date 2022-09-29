// ===--- c2c_many_1d_outofplace_basic.cu --------------------*- CUDA -*---===//
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

bool c2c_many_1d_outofplace_basic() {
  cufftHandle plan_fwd;
  float2 forward_idata_h[10];
  set_value((float*)forward_idata_h, 10);
  set_value((float*)forward_idata_h + 10, 10);

  float2* forward_idata_d;
  float2* forward_odata_d;
  float2* backward_odata_d;
  cudaMalloc(&forward_idata_d, sizeof(float2) * 10);
  cudaMalloc(&forward_odata_d, sizeof(float2) * 10);
  cudaMalloc(&backward_odata_d, sizeof(float2) * 10);
  cudaMemcpy(forward_idata_d, forward_idata_h, sizeof(float2) * 10, cudaMemcpyHostToDevice);

  int n[1] = {5};
  cufftPlanMany(&plan_fwd, 1, n, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, 2);
  cufftExecC2C(plan_fwd, forward_idata_d, forward_odata_d, CUFFT_FORWARD);
  cudaDeviceSynchronize();
  float2 forward_odata_h[10];
  cudaMemcpy(forward_odata_h, forward_odata_d, sizeof(float2) * 10, cudaMemcpyDeviceToHost);

  float2 forward_odata_ref[10];
  forward_odata_ref[0] =  float2{20,25};
  forward_odata_ref[1] =  float2{-11.8819,1.88191};
  forward_odata_ref[2] =  float2{-6.6246,-3.3754};
  forward_odata_ref[3] =  float2{-3.3754,-6.6246};
  forward_odata_ref[4] =  float2{1.88191,-11.8819};
  forward_odata_ref[5] =  float2{20,25};
  forward_odata_ref[6] =  float2{-11.8819,1.88191};
  forward_odata_ref[7] =  float2{-6.6246,-3.3754};
  forward_odata_ref[8] =  float2{-3.3754,-6.6246};
  forward_odata_ref[9] =  float2{1.88191,-11.8819};

  cufftDestroy(plan_fwd);

  if (!compare(forward_odata_ref, forward_odata_h, 10)) {
    std::cout << "forward_odata_h:" << std::endl;
    print_values(forward_odata_h, 10);
    std::cout << "forward_odata_ref:" << std::endl;
    print_values(forward_odata_ref, 10);

    cudaFree(forward_idata_d);
    cudaFree(forward_odata_d);
    cudaFree(backward_odata_d);

    return false;
  }

  cufftHandle plan_bwd;
  cufftPlanMany(&plan_bwd, 1, n, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, 2);
  cufftExecC2C(plan_bwd, forward_odata_d, backward_odata_d, CUFFT_INVERSE);
  cudaDeviceSynchronize();
  float2 backward_odata_h[10];
  cudaMemcpy(backward_odata_h, backward_odata_d, sizeof(float2) * 10, cudaMemcpyDeviceToHost);

  float2 backward_odata_ref[10];
  backward_odata_ref[0] =  float2{0,5};
  backward_odata_ref[1] =  float2{10,15};
  backward_odata_ref[2] =  float2{20,25};
  backward_odata_ref[3] =  float2{30,35};
  backward_odata_ref[4] =  float2{40,45};
  backward_odata_ref[5] =  float2{0,5};
  backward_odata_ref[6] =  float2{10,15};
  backward_odata_ref[7] =  float2{20,25};
  backward_odata_ref[8] =  float2{30,35};
  backward_odata_ref[9] =  float2{40,45};

  cudaFree(forward_idata_d);
  cudaFree(forward_odata_d);
  cudaFree(backward_odata_d);

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
#define FUNC c2c_many_1d_outofplace_basic
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

