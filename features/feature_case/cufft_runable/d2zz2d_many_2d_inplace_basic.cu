// ===--- d2zz2d_many_2d_inplace_basic.cu --------------------*- CUDA -*---===//
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



bool d2zz2d_many_2d_inplace_basic() {
  cufftHandle plan_fwd;
  cufftCreate(&plan_fwd);
  double forward_idata_h[16];
  set_value(forward_idata_h, 2, 3, 4);
  set_value(forward_idata_h + 8, 2, 3, 4);

  double* data_d;
  cudaMalloc(&data_d, sizeof(double) * 16);
  cudaMemcpy(data_d, forward_idata_h, sizeof(double) * 16, cudaMemcpyHostToDevice);

  int n[2] = {2, 3};
  size_t workSize;
  cufftMakePlanMany(plan_fwd, 2, n, nullptr, 0, 0, nullptr, 0, 0, CUFFT_D2Z, 2, &workSize);
  cufftExecD2Z(plan_fwd, data_d, (double2*)data_d);
  cudaDeviceSynchronize();
  double2 forward_odata_h[8];
  cudaMemcpy(forward_odata_h, data_d, sizeof(double) * 16, cudaMemcpyDeviceToHost);

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

    cudaFree(data_d);

    return false;
  }

  cufftHandle plan_bwd;
  cufftCreate(&plan_bwd);
  cufftMakePlanMany(plan_bwd, 2, n, nullptr, 0, 0, nullptr, 0, 0, CUFFT_Z2D, 2, &workSize);
  cufftExecZ2D(plan_bwd, (double2*)data_d, data_d);
  cudaDeviceSynchronize();
  double backward_odata_h[16];
  cudaMemcpy(backward_odata_h, data_d, sizeof(double) * 16, cudaMemcpyDeviceToHost);

  double backward_odata_ref[16];
  backward_odata_ref[0] =  0;
  backward_odata_ref[1] =  6;
  backward_odata_ref[2] =  12;
  backward_odata_ref[3] =  1.73205;
  backward_odata_ref[4] =  18;
  backward_odata_ref[5] =  24;
  backward_odata_ref[6] =  30;
  backward_odata_ref[7] =  1.73205;
  backward_odata_ref[8] =  0;
  backward_odata_ref[9] =  6;
  backward_odata_ref[10] = 12;
  backward_odata_ref[11] = 1.73205;
  backward_odata_ref[12] = 18;
  backward_odata_ref[13] = 24;
  backward_odata_ref[14] = 30;
  backward_odata_ref[15] = 1.73205;

  cudaFree(data_d);
  cufftDestroy(plan_bwd);

  std::vector<int> indices = {0, 1, 2,
                              4, 5, 6,
                              8, 9, 10,
                              12, 13, 14};
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
#define FUNC d2zz2d_many_2d_inplace_basic
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

