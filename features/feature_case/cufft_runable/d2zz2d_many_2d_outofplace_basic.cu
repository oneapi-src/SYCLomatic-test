// ===--- d2zz2d_many_2d_outofplace_basic.cu -----------------*- CUDA -*---===//
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

bool d2zz2d_many_2d_outofplace_basic() {
  cufftHandle plan_fwd;
  double forward_idata_h[2/*n0*/ * 3/*n1*/ * 2/*batch*/];
  set_value(forward_idata_h, 6);
  set_value(forward_idata_h + 6, 6);

  double* forward_idata_d;
  double2* forward_odata_d;
  double* backward_odata_d;
  cudaMalloc(&forward_idata_d, sizeof(double) * 2 * 3 * 2);
  cudaMalloc(&forward_odata_d, 2 * 2 * sizeof(double2) * (3/2+1));
  cudaMalloc(&backward_odata_d, sizeof(double) * 2 * 3 * 2);
  cudaMemcpy(forward_idata_d, forward_idata_h, sizeof(double) * 2 * 3 * 2, cudaMemcpyHostToDevice);

  int n[2] = {2, 3};
  cufftPlanMany(&plan_fwd, 2, n, nullptr, 0, 0, nullptr, 0, 0, CUFFT_D2Z, 2);
  cufftExecD2Z(plan_fwd, forward_idata_d, forward_odata_d);
  cudaDeviceSynchronize();
  double2 forward_odata_h[8];
  cudaMemcpy(forward_odata_h, forward_odata_d, 2 * 2 * sizeof(double2) * (3/2+1), cudaMemcpyDeviceToHost);

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
  cufftPlanMany(&plan_bwd, 2, n, nullptr, 0, 0, nullptr, 0, 0, CUFFT_Z2D, 2);
  cufftExecZ2D(plan_bwd, forward_odata_d, backward_odata_d);
  cudaDeviceSynchronize();
  double backward_odata_h[12];
  cudaMemcpy(backward_odata_h, backward_odata_d, sizeof(double) * 12, cudaMemcpyDeviceToHost);

  double backward_odata_ref[12];
  backward_odata_ref[0] =  0;
  backward_odata_ref[1] =  6;
  backward_odata_ref[2] =  12;
  backward_odata_ref[3] =  18;
  backward_odata_ref[4] =  24;
  backward_odata_ref[5] =  30;
  backward_odata_ref[6] =  0;
  backward_odata_ref[7] =  6;
  backward_odata_ref[8] =  12;
  backward_odata_ref[9] =  18;
  backward_odata_ref[10] = 24;
  backward_odata_ref[11] = 30;

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
#define FUNC d2zz2d_many_2d_outofplace_basic
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

