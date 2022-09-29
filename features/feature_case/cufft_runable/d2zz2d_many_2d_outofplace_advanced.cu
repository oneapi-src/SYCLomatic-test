// ===--- d2zz2d_many_2d_outofplace_advanced.cu --------------*- CUDA -*---===//
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
// +---+---+---+---+---+---+     -+
// | r | 0 | r | 0 | r | 0 |      |
// +---+---+---+---+---+---+      |
// | r | 0 | r | 0 | r | 0 |      batch0
// +---+---+---+---+---+---+     -+
// | r | 0 | r | 0 | r | 0 |      |
// +---+---+---+---+---+---+      |
// | r | 0 | r | 0 | r | 0 |      batch1
// +---+---+---+---+---+---+     -+
// |__________n2___________|
// |________nembed2________|
// output
// +---+---+---+---+ -+
// |   c   |   c   |  |
// +---+---+---+---+  batch0
// |   c   |   c   |  |
// +---+---+---+---+ -+
// |   c   |   c   |  |
// +---+---+---+---+  batch1
// |   c   |   c   |  |
// +---+---+---+---+ -+
// |______n2_______|
// |____nembed2____|
bool d2zz2d_many_2d_outofplace_advanced() {
  cufftHandle plan_fwd;
  cufftCreate(&plan_fwd);
  double forward_idata_h[24];
  std::memset(forward_idata_h, 0, sizeof(double) * 24);
  forward_idata_h[0]  = 0;
  forward_idata_h[2]  = 1;
  forward_idata_h[4]  = 2;
  forward_idata_h[6]  = 3;
  forward_idata_h[8]  = 4;
  forward_idata_h[10] = 5;
  forward_idata_h[12] = 0;
  forward_idata_h[14] = 1;
  forward_idata_h[16] = 2;
  forward_idata_h[18] = 3;
  forward_idata_h[20] = 4;
  forward_idata_h[22] = 5;

  double* forward_idata_d;
  double2* forward_odata_d;
  double* backward_odata_d;
  cudaMalloc(&forward_idata_d, sizeof(double) * 24);
  cudaMalloc(&forward_odata_d, sizeof(double2) * 8);
  cudaMalloc(&backward_odata_d, sizeof(double) * 24);
  cudaMemcpy(forward_idata_d, forward_idata_h, sizeof(double) * 24, cudaMemcpyHostToDevice);

  long long int n[2] = {2, 3};
  long long int inembed[2] = {2, 3};
  long long int onembed[2] = {2, 2};
  size_t workSize;
  cufftXtMakePlanMany(plan_fwd, 2, n, inembed, 2, 12, CUDA_R_64F, onembed, 1, 4, CUDA_C_64F, 2, &workSize, CUDA_C_64F);
  cufftXtExec(plan_fwd, forward_idata_d, forward_odata_d, CUFFT_FORWARD);
  cudaDeviceSynchronize();
  double2 forward_odata_h[8];
  cudaMemcpy(forward_odata_h, forward_odata_d, sizeof(double2) * 8, cudaMemcpyDeviceToHost);

  double2 forward_odata_ref[8];
  forward_odata_ref[0] =  double2{15,0};
  forward_odata_ref[1] =  double2{-3,1.73205};
  forward_odata_ref[2] =  double2{-9,0};
  forward_odata_ref[3] =  double2{0,0};
  forward_odata_ref[4] =  double2{15,0};
  forward_odata_ref[5] =  double2{-3,1.73205};
  forward_odata_ref[6] =  double2{-9,0};
  forward_odata_ref[7] =  double2{0,0};

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
  cufftXtMakePlanMany(plan_bwd, 2, n, onembed, 1, 4, CUDA_C_64F, inembed, 2, 12, CUDA_R_64F, 2, &workSize, CUDA_C_64F);
  cufftXtExec(plan_bwd, forward_odata_d, backward_odata_d, CUFFT_INVERSE);
  cudaDeviceSynchronize();
  double backward_odata_h[24];
  cudaMemcpy(backward_odata_h, backward_odata_d, sizeof(double) * 24, cudaMemcpyDeviceToHost);

  double backward_odata_ref[24];
  backward_odata_ref[0]  = 0;
  backward_odata_ref[2]  = 6;
  backward_odata_ref[4]  = 12;
  backward_odata_ref[6]  = 18;
  backward_odata_ref[8]  = 24;
  backward_odata_ref[10] = 30;
  backward_odata_ref[12] = 0;
  backward_odata_ref[14] = 6;
  backward_odata_ref[16] = 12;
  backward_odata_ref[18] = 18;
  backward_odata_ref[20] = 24;
  backward_odata_ref[22] = 30;

  cudaFree(forward_idata_d);
  cudaFree(forward_odata_d);
  cudaFree(backward_odata_d);

  cufftDestroy(plan_bwd);

  std::vector<int> indices = {0, 2, 4,
                              6, 8, 10,
                              12, 14, 16,
                              18, 20, 22};
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
#define FUNC d2zz2d_many_2d_outofplace_advanced
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

