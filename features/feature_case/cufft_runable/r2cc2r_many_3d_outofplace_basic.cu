// ===--- r2cc2r_many_3d_outofplace_basic.cu -----------------*- CUDA -*---===//
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

bool r2cc2r_many_3d_outofplace_basic() {
  cufftHandle plan_fwd;
  float forward_idata_h[4/*n0*/ * 2/*n1*/ * 3/*n2*/ * 2/*batch*/];
  set_value(forward_idata_h, 24);
  set_value(forward_idata_h + 24, 24);

  float* forward_idata_d;
  float2* forward_odata_d;
  float* backward_odata_d;
  cudaMalloc(&forward_idata_d, 2 * sizeof(float) * 4 * 2 * 3);
  cudaMalloc(&forward_odata_d, 2 * sizeof(float2) * 4 * 2 * (3/2+1));
  cudaMalloc(&backward_odata_d, 2 * sizeof(float) * 4 * 2 * 3);
  cudaMemcpy(forward_idata_d, forward_idata_h, 2 * sizeof(float) * 4 * 2 * 3, cudaMemcpyHostToDevice);

  int n[3] = {4, 2, 3};
  cufftPlanMany(&plan_fwd, 3, n, nullptr, 0, 0, nullptr, 0, 0, CUFFT_R2C, 2);
  cufftExecR2C(plan_fwd, forward_idata_d, forward_odata_d);
  cudaDeviceSynchronize();
  float2 forward_odata_h[32];
  cudaMemcpy(forward_odata_h, forward_odata_d, 2 * sizeof(float2) * 4 * 2 * (3/2+1), cudaMemcpyDeviceToHost);

  float2 forward_odata_ref[32];
  forward_odata_ref[0] =  float2{276,0};
  forward_odata_ref[1] =  float2{-12,6.9282};
  forward_odata_ref[2] =  float2{-36,0};
  forward_odata_ref[3] =  float2{0,0};
  forward_odata_ref[4] =  float2{-72,72};
  forward_odata_ref[5] =  float2{0,0};
  forward_odata_ref[6] =  float2{0,0};
  forward_odata_ref[7] =  float2{0,0};
  forward_odata_ref[8] =  float2{-72,0};
  forward_odata_ref[9] =  float2{0,0};
  forward_odata_ref[10] = float2{0,0};
  forward_odata_ref[11] = float2{0,0};
  forward_odata_ref[12] = float2{-72,-72};
  forward_odata_ref[13] = float2{0,0};
  forward_odata_ref[14] = float2{0,0};
  forward_odata_ref[15] = float2{0,0};
  forward_odata_ref[16] = float2{276,0};
  forward_odata_ref[17] = float2{-12,6.9282};
  forward_odata_ref[18] = float2{-36,0};
  forward_odata_ref[19] = float2{0,0};
  forward_odata_ref[20] = float2{-72,72};
  forward_odata_ref[21] = float2{0,0};
  forward_odata_ref[22] = float2{0,0};
  forward_odata_ref[23] = float2{0,0};
  forward_odata_ref[24] = float2{-72,0};
  forward_odata_ref[25] = float2{0,0};
  forward_odata_ref[26] = float2{0,0};
  forward_odata_ref[27] = float2{0,0};
  forward_odata_ref[28] = float2{-72,-72};
  forward_odata_ref[29] = float2{0,0};
  forward_odata_ref[30] = float2{0,0};
  forward_odata_ref[31] = float2{0,0};

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
  cufftPlanMany(&plan_bwd, 3, n, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2R, 2);
  cufftExecC2R(plan_bwd, forward_odata_d, backward_odata_d);
  cudaDeviceSynchronize();
  float backward_odata_h[48];
  cudaMemcpy(backward_odata_h, backward_odata_d, sizeof(float) * 48, cudaMemcpyDeviceToHost);

  float backward_odata_ref[48];
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
#define FUNC r2cc2r_many_3d_outofplace_basic
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

