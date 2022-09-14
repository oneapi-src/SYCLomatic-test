// ===--- r2cc2r_many_1d_inplace_advanced.cu -----------------*- CUDA -*---===//
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
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
// | r | 0 | r | 0 | r | 0 | r | 0 | r | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | r | 0 | r | 0 | r | 0 | r | 0 | r | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
// |___________________n___________________|                               |___________________n___________________|                               |
// |_________________nembed________________|                               |_________________nembed________________|                               |
// |___________________________________batch0______________________________|___________________________________batch1______________________________|
// output
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
// |   c   |   0   |   c   |   0   |   c   |   0   |   0   |   0   |   0   |   c   |   0   |   c   |   0   |   c   |   0   |   0   |   0   |   0   |
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
// |________________________n______________________|               |       |________________________n______________________|               |       |
// |_____________________________nembed____________________________|       |_____________________________nembed____________________________|       |
// |___________________________________batch0______________________________|___________________________________batch1______________________________|
bool r2cc2r_many_1d_inplace_advanced() {
  cufftHandle plan_fwd;
  cufftCreate(&plan_fwd);
  float forward_idata_h[36];
  std::memset(forward_idata_h, 0, sizeof(float) * 36);
  forward_idata_h[0] = 0;
  forward_idata_h[2] = 1;
  forward_idata_h[4] = 2;
  forward_idata_h[6] = 3;
  forward_idata_h[8] = 4;
  forward_idata_h[18] = 0;
  forward_idata_h[20] = 1;
  forward_idata_h[22] = 2;
  forward_idata_h[24] = 3;
  forward_idata_h[26] = 4;

  float* data_d;
  cudaMalloc(&data_d, sizeof(float) * 36);
  cudaMemcpy(data_d, forward_idata_h, sizeof(float) * 36, cudaMemcpyHostToDevice);

  size_t workSize;
  long long int n[1] = {5};
  long long int inembed[1] = {5};
  long long int onembed[1] = {4};
  cufftMakePlanMany64(plan_fwd, 1, n, inembed, 2, 18, onembed, 2, 9, CUFFT_R2C, 2, &workSize);
  cufftExecR2C(plan_fwd, data_d, (float2*)data_d);
  cudaDeviceSynchronize();
  float2 forward_odata_h[18];
  cudaMemcpy(forward_odata_h, data_d, sizeof(float) * 36, cudaMemcpyDeviceToHost);

  float2 forward_odata_ref[18];
  forward_odata_ref[0] =  float2{10,0};
  forward_odata_ref[1] =  float2{2,3};
  forward_odata_ref[2] =  float2{-2.5,3.44095};
  forward_odata_ref[3] =  float2{1,2};
  forward_odata_ref[4] =  float2{-2.5,0.812299};
  forward_odata_ref[5] =  float2{0,0};
  forward_odata_ref[6] =  float2{0,0};
  forward_odata_ref[7] =  float2{0,0};
  forward_odata_ref[8] =  float2{0,0};
  forward_odata_ref[9] =  float2{10,0};
  forward_odata_ref[10] = float2{0,0};
  forward_odata_ref[11] = float2{-2.5,3.44095};
  forward_odata_ref[12] = float2{0,0};
  forward_odata_ref[13] = float2{-2.5,0.812299};
  forward_odata_ref[14] = float2{0,0};
  forward_odata_ref[15] = float2{0,0};
  forward_odata_ref[16] = float2{0,0};
  forward_odata_ref[17] = float2{0,0};

  cufftDestroy(plan_fwd);

  std::vector<int> indices = {0, 2, 4,
                              9, 11, 13};
  if (!compare(forward_odata_ref, forward_odata_h, indices)) {
    std::cout << "forward_odata_h:" << std::endl;
    print_values(forward_odata_h, indices);
    std::cout << "forward_odata_ref:" << std::endl;
    print_values(forward_odata_ref, indices);

    cudaFree(data_d);

    return false;
  }

  cufftHandle plan_bwd;
  cufftCreate(&plan_bwd);
  cufftMakePlanMany64(plan_bwd, 1, n, onembed, 2, 9, inembed, 2, 18, CUFFT_C2R, 2, &workSize);
  cufftExecC2R(plan_bwd, (float2*)data_d, data_d);
  cudaDeviceSynchronize();
  float backward_odata_h[36];
  cudaMemcpy(backward_odata_h, data_d, sizeof(float) * 36, cudaMemcpyDeviceToHost);

  float backward_odata_ref[36];
  backward_odata_ref[0] = 0;
  backward_odata_ref[2] = 5;
  backward_odata_ref[4] = 10;
  backward_odata_ref[6] = 15;
  backward_odata_ref[8] = 20;
  backward_odata_ref[18] = 0;
  backward_odata_ref[20] = 5;
  backward_odata_ref[22] = 10;
  backward_odata_ref[24] = 15;
  backward_odata_ref[26] = 20;

  cudaFree(data_d);
  cufftDestroy(plan_bwd);

  std::vector<int> indices_bwd = {0, 2, 4, 6, 8,
                                  18, 20, 22, 24, 26};
  if (!compare(backward_odata_ref, backward_odata_h, indices_bwd)) {
    std::cout << "backward_odata_h:" << std::endl;
    print_values(backward_odata_h, indices_bwd);
    std::cout << "backward_odata_ref:" << std::endl;
    print_values(backward_odata_ref, indices_bwd);
    return false;
  }
  return true;
}


#ifdef DEBUG_FFT
int main() {
#define FUNC r2cc2r_many_1d_inplace_advanced
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

