// ===--- c2c_many_2d_inplace_basic.cu -----------------------*- CUDA -*---===//
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

bool c2c_many_2d_inplace_basic() {
  cufftHandle plan_fwd;
  cufftCreate(&plan_fwd);
  float2 forward_idata_h[2/*n0*/ * 3/*n1*/ * 2/*batch*/];
  set_value((float*)forward_idata_h, 12);
  set_value((float*)forward_idata_h + 12, 12);

  float2* data_d;
  cudaMalloc(&data_d, sizeof(float2) * 12);
  cudaMemcpy(data_d, forward_idata_h, sizeof(float2) * 12, cudaMemcpyHostToDevice);

  int n[2] = {2, 3};
  size_t workSize;
  cufftMakePlanMany(plan_fwd, 2, n, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, 2, &workSize);
  cufftExecC2C(plan_fwd, data_d, data_d, CUFFT_FORWARD);
  cudaDeviceSynchronize();
  float2 forward_odata_h[12];
  cudaMemcpy(forward_odata_h, data_d, sizeof(float2) * 12, cudaMemcpyDeviceToHost);

  float2 forward_odata_ref[12];
  forward_odata_ref[0] =  float2{30,36};
  forward_odata_ref[1] =  float2{-9.4641,-2.5359};
  forward_odata_ref[2] =  float2{-2.5359,-9.4641};
  forward_odata_ref[3] =  float2{-18,-18};
  forward_odata_ref[4] =  float2{0,0};
  forward_odata_ref[5] =  float2{0,0};
  forward_odata_ref[6] =  float2{30,36};
  forward_odata_ref[7] =  float2{-9.4641,-2.5359};
  forward_odata_ref[8] =  float2{-2.5359,-9.4641};
  forward_odata_ref[9] =  float2{-18,-18};
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
  cufftMakePlanMany(plan_bwd, 2, n, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, 2, &workSize);
  cufftExecC2C(plan_bwd, data_d, data_d, CUFFT_INVERSE);
  cudaDeviceSynchronize();
  float2 backward_odata_h[12];
  cudaMemcpy(backward_odata_h, data_d, sizeof(float2) * 12, cudaMemcpyDeviceToHost);

  float2 backward_odata_ref[12];
  backward_odata_ref[0] =  float2{0,6};
  backward_odata_ref[1] =  float2{12,18};
  backward_odata_ref[2] =  float2{24,30};
  backward_odata_ref[3] =  float2{36,42};
  backward_odata_ref[4] =  float2{48,54};
  backward_odata_ref[5] =  float2{60,66};
  backward_odata_ref[6] =  float2{0,6};
  backward_odata_ref[7] =  float2{12,18};
  backward_odata_ref[8] =  float2{24,30};
  backward_odata_ref[9] =  float2{36,42};
  backward_odata_ref[10] = float2{48,54};
  backward_odata_ref[11] = float2{60,66};

  cudaFree(data_d);
  cufftDestroy(plan_bwd);

  if (!compare(backward_odata_ref, backward_odata_h, 12)) {
    std::cout << "backward_odata_h:" << std::endl;
    print_values(backward_odata_h, 12);
    std::cout << "backward_odata_ref:" << std::endl;
    print_values(backward_odata_ref, 12);
    return false;
  }
  return true;
}


#ifdef DEBUG_FFT
int main() {
#define FUNC c2c_many_2d_inplace_basic
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

