// ===--- d2zz2d_many_3d_outofplace_basic.cu -----------------*- CUDA -*---===//
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

bool d2zz2d_many_3d_outofplace_basic() {
  cufftHandle plan_fwd;
  double forward_idata_h[4/*n0*/ * 2/*n1*/ * 3/*n2*/ * 2/*batch*/];
  set_value(forward_idata_h, 24);
  set_value(forward_idata_h + 24, 24);

  double* forward_idata_d;
  double2* forward_odata_d;
  double* backward_odata_d;
  cudaMalloc(&forward_idata_d, 2 * sizeof(double) * 4 * 2 * 3);
  cudaMalloc(&forward_odata_d, 2 * sizeof(double2) * 4 * 2 * (3/2+1));
  cudaMalloc(&backward_odata_d, 2 * sizeof(double) * 4 * 2 * 3);
  cudaMemcpy(forward_idata_d, forward_idata_h, 2 * sizeof(double) * 4 * 2 * 3, cudaMemcpyHostToDevice);

  int n[3] = {4, 2, 3};
  cufftPlanMany(&plan_fwd, 3, n, nullptr, 0, 0, nullptr, 0, 0, CUFFT_D2Z, 2);
  cufftExecD2Z(plan_fwd, forward_idata_d, forward_odata_d);
  cudaDeviceSynchronize();
  double2 forward_odata_h[32];
  cudaMemcpy(forward_odata_h, forward_odata_d, 2 * sizeof(double2) * 4 * 2 * (3/2+1), cudaMemcpyDeviceToHost);

  double2 forward_odata_ref[32];
  forward_odata_ref[0] =  double2{276,0};
  forward_odata_ref[1] =  double2{-12,6.9282};
  forward_odata_ref[2] =  double2{-36,0};
  forward_odata_ref[3] =  double2{0,0};
  forward_odata_ref[4] =  double2{-72,72};
  forward_odata_ref[5] =  double2{0,0};
  forward_odata_ref[6] =  double2{0,0};
  forward_odata_ref[7] =  double2{0,0};
  forward_odata_ref[8] =  double2{-72,0};
  forward_odata_ref[9] =  double2{0,0};
  forward_odata_ref[10] = double2{0,0};
  forward_odata_ref[11] = double2{0,0};
  forward_odata_ref[12] = double2{-72,-72};
  forward_odata_ref[13] = double2{0,0};
  forward_odata_ref[14] = double2{0,0};
  forward_odata_ref[15] = double2{0,0};
  forward_odata_ref[16] = double2{276,0};
  forward_odata_ref[17] = double2{-12,6.9282};
  forward_odata_ref[18] = double2{-36,0};
  forward_odata_ref[19] = double2{0,0};
  forward_odata_ref[20] = double2{-72,72};
  forward_odata_ref[21] = double2{0,0};
  forward_odata_ref[22] = double2{0,0};
  forward_odata_ref[23] = double2{0,0};
  forward_odata_ref[24] = double2{-72,0};
  forward_odata_ref[25] = double2{0,0};
  forward_odata_ref[26] = double2{0,0};
  forward_odata_ref[27] = double2{0,0};
  forward_odata_ref[28] = double2{-72,-72};
  forward_odata_ref[29] = double2{0,0};
  forward_odata_ref[30] = double2{0,0};
  forward_odata_ref[31] = double2{0,0};

  cufftDestroy(plan_fwd);

  if (!compare(forward_odata_ref, forward_odata_h, 32)) {
    std::cout << "forward_odata_h:" << std::endl;
    print_values(forward_odata_h, 32);
    std::cout << "forward_odata_ref:" << std::endl;
    print_values(forward_odata_ref, 32);

    cudaFree(forward_idata_d);
    cudaFree(forward_odata_d);
    cudaFree(backward_odata_d);

    return false;
  }

  cufftHandle plan_bwd;
  cufftPlanMany(&plan_bwd, 3, n, nullptr, 0, 0, nullptr, 0, 0, CUFFT_Z2D, 2);
  cufftExecZ2D(plan_bwd, forward_odata_d, backward_odata_d);
  cudaDeviceSynchronize();
  double backward_odata_h[48];
  cudaMemcpy(backward_odata_h, backward_odata_d, sizeof(double) * 48, cudaMemcpyDeviceToHost);

  double backward_odata_ref[48];
  backward_odata_ref[0] =  0;
  backward_odata_ref[1] =  24;
  backward_odata_ref[2] =  48;
  backward_odata_ref[3] =  72;
  backward_odata_ref[4] =  96;
  backward_odata_ref[5] =  120;
  backward_odata_ref[6] =  144;
  backward_odata_ref[7] =  168;
  backward_odata_ref[8] =  192;
  backward_odata_ref[9] =  216;
  backward_odata_ref[10] = 240;
  backward_odata_ref[11] = 264;
  backward_odata_ref[12] = 288;
  backward_odata_ref[13] = 312;
  backward_odata_ref[14] = 336;
  backward_odata_ref[15] = 360;
  backward_odata_ref[16] = 384;
  backward_odata_ref[17] = 408;
  backward_odata_ref[18] = 432;
  backward_odata_ref[19] = 456;
  backward_odata_ref[20] = 480;
  backward_odata_ref[21] = 504;
  backward_odata_ref[22] = 528;
  backward_odata_ref[23] = 552;
  backward_odata_ref[24] = 0;
  backward_odata_ref[25] = 24;
  backward_odata_ref[26] = 48;
  backward_odata_ref[27] = 72;
  backward_odata_ref[28] = 96;
  backward_odata_ref[29] = 120;
  backward_odata_ref[30] = 144;
  backward_odata_ref[31] = 168;
  backward_odata_ref[32] = 192;
  backward_odata_ref[33] = 216;
  backward_odata_ref[34] = 240;
  backward_odata_ref[35] = 264;
  backward_odata_ref[36] = 288;
  backward_odata_ref[37] = 312;
  backward_odata_ref[38] = 336;
  backward_odata_ref[39] = 360;
  backward_odata_ref[40] = 384;
  backward_odata_ref[41] = 408;
  backward_odata_ref[42] = 432;
  backward_odata_ref[43] = 456;
  backward_odata_ref[44] = 480;
  backward_odata_ref[45] = 504;
  backward_odata_ref[46] = 528;
  backward_odata_ref[47] = 552;

  cudaFree(forward_idata_d);
  cudaFree(forward_odata_d);
  cudaFree(backward_odata_d);

  cufftDestroy(plan_bwd);

  if (!compare(backward_odata_ref, backward_odata_h, 48)) {
    std::cout << "backward_odata_h:" << std::endl;
    print_values(backward_odata_h, 48);
    std::cout << "backward_odata_ref:" << std::endl;
    print_values(backward_odata_ref, 48);
    return false;
  }
  return true;
}


#ifdef DEBUG_FFT
int main() {
#define FUNC d2zz2d_many_3d_outofplace_basic
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

