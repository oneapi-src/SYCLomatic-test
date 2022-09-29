// ===--- r2cc2r_many_3d_outofplace_advanced.cu --------------*- CUDA -*---===//
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
// +---+---+---+---+---+---+         -+ -+          ---+-------+     -----------+
// | r |   | r |   | r |   |          |  |             |       |                |
// +---+---+---+---+---+---+          n2 |             |       |                |
// | r |   | r |   | r |   |          |  nembed2       |       |                |
// +---+---+---+---+---+---+         ----+             |       |                |
// | r |   | r |   | r |   |          |  |             |       |                |
// +---+---+---+---+---+---+          n2 |             |       |                |
// | r |   | r |   | r |   |          |  nembed2       |       |                |
// +---+---+---+---+---+---+         ----+             n1      |                a batch
// | r |   | r |   | r |   |          |  |             |       |                |
// +---+---+---+---+---+---+          n2 |             |    nembed1             |
// | r |   | r |   | r |   |          |  nembed2       |       |                |
// +---+---+---+---+---+---+         ----+             |       |                |
// | r |   | r |   | r |   |          |  |             |       |                |
// +---+---+---+---+---+---+          n2 |             |       |                |
// | r |   | r |   | r |   |          |  nembed2       |       |                |
// +---+---+---+---+---+---+         -+--+          ---+-------+     -----------+
// |__________n3___________|
// |________nembed3________|
// output
// +---+---+---+---+ -+          ---+
// |   c   |   c   |  |             |
// +---+---+---+---+  n2/nembed2    |
// |   c   |   c   |  |             |
// +---+---+---+---+ -+             |
// |   c   |   c   |  |             |
// +---+---+---+---+  n2/nembed2    |
// |   c   |   c   |  |             |
// +---+---+---+---+ -+        n1/nembed1/a batch
// |   c   |   c   |  |             |
// +---+---+---+---+  n2/nembed2    |
// |   c   |   c   |  |             |
// +---+---+---+---+ -+             |
// |   c   |   c   |  |             |
// +---+---+---+---+  n2/nembed2    |
// |   c   |   c   |  |             |
// +---+---+---+---+ -+          ---+
// |______n3_______|
// |____nembed3____|
bool r2cc2r_many_3d_outofplace_advanced() {
  cufftHandle plan_fwd;
  cufftCreate(&plan_fwd);
  float forward_idata_h[96];
  std::memset(forward_idata_h, 0, sizeof(float) * 96);
  forward_idata_h[0]  = 0;
  forward_idata_h[2]  = 1;
  forward_idata_h[4]  = 2;
  forward_idata_h[6]  = 3;
  forward_idata_h[8]  = 4;
  forward_idata_h[10] = 5;
  forward_idata_h[12] = 6;
  forward_idata_h[14] = 7;
  forward_idata_h[16] = 8;
  forward_idata_h[18] = 9;
  forward_idata_h[20] = 10;
  forward_idata_h[22] = 11;
  forward_idata_h[24] = 12;
  forward_idata_h[26] = 13;
  forward_idata_h[28] = 14;
  forward_idata_h[30] = 15;
  forward_idata_h[32] = 16;
  forward_idata_h[34] = 17;
  forward_idata_h[36] = 18;
  forward_idata_h[38] = 19;
  forward_idata_h[40] = 20;
  forward_idata_h[42] = 21;
  forward_idata_h[44] = 22;
  forward_idata_h[46] = 23;
  std::memcpy(forward_idata_h + 48, forward_idata_h, sizeof(float) * 48);

  float* forward_idata_d;
  float2* forward_odata_d;
  float* backward_odata_d;
  cudaMalloc(&forward_idata_d, sizeof(float) * 96);
  cudaMalloc(&forward_odata_d, sizeof(float2) * 32);
  cudaMalloc(&backward_odata_d, sizeof(float) * 96);
  cudaMemcpy(forward_idata_d, forward_idata_h, sizeof(float) * 96, cudaMemcpyHostToDevice);

  long long int n[3] = {4, 2, 3};
  long long int inembed[3] = {4, 2, 3};
  long long int onembed[3] = {4, 2, 2};
  size_t workSize;
  cufftXtMakePlanMany(plan_fwd, 3, n, inembed, 2, 48, CUDA_R_32F, onembed, 1, 16, CUDA_C_32F, 2, &workSize, CUDA_C_32F);
  cufftXtExec(plan_fwd, forward_idata_d, forward_odata_d, CUFFT_FORWARD);
  cudaDeviceSynchronize();
  float2 forward_odata_h[32];
  cudaMemcpy(forward_odata_h, forward_odata_d, sizeof(float2) * 32, cudaMemcpyDeviceToHost);

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
  cufftCreate(&plan_bwd);
  cufftXtMakePlanMany(plan_bwd, 3, n, onembed, 1, 16, CUDA_C_32F, inembed, 2, 48, CUDA_R_32F, 2, &workSize, CUDA_C_32F);
  cufftXtExec(plan_bwd, forward_odata_d, backward_odata_d, CUFFT_INVERSE);
  cudaDeviceSynchronize();
  float backward_odata_h[96];
  cudaMemcpy(backward_odata_h, backward_odata_d, sizeof(float) * 96, cudaMemcpyDeviceToHost);

  float backward_odata_ref[96];
  backward_odata_ref[0] = 0;
  backward_odata_ref[2] = 24;
  backward_odata_ref[4] = 48;
  backward_odata_ref[6] = 72;
  backward_odata_ref[8] = 96;
  backward_odata_ref[10] = 120;
  backward_odata_ref[12] = 144;
  backward_odata_ref[14] = 168;
  backward_odata_ref[16] = 192;
  backward_odata_ref[18] = 216;
  backward_odata_ref[20] = 240;
  backward_odata_ref[22] = 264;
  backward_odata_ref[24] = 288;
  backward_odata_ref[26] = 312;
  backward_odata_ref[28] = 336;
  backward_odata_ref[30] = 360;
  backward_odata_ref[32] = 384;
  backward_odata_ref[34] = 408;
  backward_odata_ref[36] = 432;
  backward_odata_ref[38] = 456;
  backward_odata_ref[40] = 480;
  backward_odata_ref[42] = 504;
  backward_odata_ref[44] = 528;
  backward_odata_ref[46] = 552;
  std::memcpy(backward_odata_ref + 48, backward_odata_ref, sizeof(float) * 48);

  cudaFree(forward_idata_d);
  cudaFree(forward_odata_d);
  cudaFree(backward_odata_d);

  cufftDestroy(plan_bwd);

  std::vector<int> indices = {0, 2, 4, 6, 8,
                              10, 12, 14, 16, 18,
                              20, 12, 14, 16, 18,
                              30, 22, 24, 26, 28,
                              40, 32, 34, 36, 38,
                              50, 42, 44, 46, 48,
                              60, 52, 54, 56, 58,
                              70, 62, 64, 66, 68,
                              80, 72, 74, 76, 78,
                              90, 82, 84, 86, 88,
                              10, 92, 94};
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
#define FUNC r2cc2r_many_3d_outofplace_advanced
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

