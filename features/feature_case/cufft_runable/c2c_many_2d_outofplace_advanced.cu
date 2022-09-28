// ===--- c2c_many_2d_outofplace_advanced.cu -----------------*- CUDA -*---===//
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

// forward
// input
// +---+---+---+---+---+---+      -+
// |   c   |   c   |   c   |       |
// +---+---+---+---+---+---+       |
// |   c   |   c   |   c   |       batch0
// +---+---+---+---+---+---+      -+
// |   c   |   c   |   c   |       |
// +---+---+---+---+---+---+       |
// |   c   |   c   |   c   |       batch1
// +---+---+---+---+---+---+      -+
// |___________n2__________|
// |________nembed2________|
// output
// +---+---+---+---+---+---+ -+
// |   c   |   c   |   c   |  |
// +---+---+---+---+---+---+  batch0
// |   c   |   c   |   c   |  |
// +---+---+---+---+---+---+ -+
// |   c   |   c   |   c   |  |
// +---+---+---+---+---+---+  batch1
// |   c   |   c   |   c   |  |
// +---+---+---+---+---+---+ -+
// |__________n2___________|
// |________nembed2________|
bool c2c_many_2d_outofplace_advanced() {
  cufftHandle plan_fwd;
  cufftCreate(&plan_fwd);
  float2 forward_idata_h[12];
  std::memset(forward_idata_h, 0, sizeof(float2) * 12);
  forward_idata_h[0] = float2{0, 1};
  forward_idata_h[1] = float2{2, 3};
  forward_idata_h[2] = float2{4, 5};
  forward_idata_h[3] = float2{6, 7};
  forward_idata_h[4] = float2{8, 9};
  forward_idata_h[5] = float2{10, 11};
  forward_idata_h[6] = float2{0, 1};
  forward_idata_h[7] = float2{2, 3};
  forward_idata_h[8] = float2{4, 5};
  forward_idata_h[9] = float2{6, 7};
  forward_idata_h[10] = float2{8, 9};
  forward_idata_h[11] = float2{10, 11};

  float2* forward_idata_d;
  float2* forward_odata_d;
  float2* backward_odata_d;
  cudaMalloc(&forward_idata_d, sizeof(float2) * 12);
  cudaMalloc(&forward_odata_d, sizeof(float2) * 12);
  cudaMalloc(&backward_odata_d, sizeof(float2) * 12);
  cudaMemcpy(forward_idata_d, forward_idata_h, sizeof(float2) * 12, cudaMemcpyHostToDevice);

  size_t workSize;
  long long int n[2] = {2, 3};
  long long int inembed[2] = {2, 3};
  long long int onembed[2] = {2, 3};
  cufftXtMakePlanMany(plan_fwd, 2, n, inembed, 1, 6, CUDA_C_32F, onembed, 1, 6, CUDA_C_32F, 2, &workSize, CUDA_C_32F);
  cufftXtExec(plan_fwd, forward_idata_d, forward_odata_d, CUFFT_FORWARD);
  cudaDeviceSynchronize();
  float2 forward_odata_h[12];
  cudaMemcpy(forward_odata_h, forward_odata_d, sizeof(float2) * 12, cudaMemcpyDeviceToHost);

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

    cudaFree(forward_idata_d);
    cudaFree(forward_odata_d);
    cudaFree(backward_odata_d);

    return false;
  }

  cufftHandle plan_bwd;
  cufftCreate(&plan_bwd);
  cufftXtMakePlanMany(plan_bwd, 2, n, onembed, 1, 6, CUDA_C_32F, inembed, 1, 6, CUDA_C_32F, 2, &workSize, CUDA_C_32F);
  cufftXtExec(plan_bwd, forward_odata_d, backward_odata_d, CUFFT_INVERSE);
  cudaDeviceSynchronize();
  float2 backward_odata_h[12];
  cudaMemcpy(backward_odata_h, backward_odata_d, sizeof(float2) * 12, cudaMemcpyDeviceToHost);

  float2 backward_odata_ref[12];
  backward_odata_ref[0] = float2{0, 6};
  backward_odata_ref[1] = float2{12, 18};
  backward_odata_ref[2] = float2{24, 30};
  backward_odata_ref[3] = float2{36, 42};
  backward_odata_ref[4] = float2{48, 54};
  backward_odata_ref[5] = float2{60, 66};
  backward_odata_ref[6] = float2{0, 6};
  backward_odata_ref[7] = float2{12, 18};
  backward_odata_ref[8] = float2{24, 30};
  backward_odata_ref[9] = float2{36, 42};
  backward_odata_ref[10] = float2{48, 54};
  backward_odata_ref[11] = float2{60, 66};

  cudaFree(forward_idata_d);
  cudaFree(forward_odata_d);
  cudaFree(backward_odata_d);

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
#define FUNC c2c_many_2d_outofplace_advanced
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

