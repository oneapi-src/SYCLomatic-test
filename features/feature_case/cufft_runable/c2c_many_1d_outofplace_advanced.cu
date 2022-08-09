// ===--- c2c_many_1d_outofplace_advanced.cu -----------------*- CUDA -*---===//
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
// |   c   |   0   |   c   |   0   |   c   |   0   |   0   |   0   |   0   |   c   |   0   |   c   |   0   |   c   |   0   |   0   |   0   |   0   |
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
// |________________________n______________________|               |       |________________________n______________________|               |       |
// |_____________________________nembed____________________________|       |_____________________________nembed____________________________|       |
// |___________________________________batch0______________________________|___________________________________batch1______________________________|
// output
// +---+---+---+---+---+---+---+---+---+---+---+---+
// |   c   |   c   |   c   |   c   |   c   |   c   |
// +---+---+---+---+---+---+---+---+---+---+---+---+
// |___________n___________|___________n___________|
// |_________nembed________|_________nembed________|
// |_________batch0________|_________batch1________|
bool c2c_many_1d_outofplace_advanced() {
  cufftHandle plan_fwd;
  cufftCreate(&plan_fwd);
  float2 forward_idata_h[18];
  std::memset(forward_idata_h, 0, sizeof(float2) * 18);
  set_value_with_stride(forward_idata_h, 3, 2);
  set_value_with_stride(forward_idata_h + 9, 3, 2);

  float2* forward_idata_d;
  float2* forward_odata_d;
  float2* backward_odata_d;
  cudaMalloc(&forward_idata_d, sizeof(float2) * 18);
  cudaMalloc(&forward_odata_d, sizeof(float2) * 6);
  cudaMalloc(&backward_odata_d, sizeof(float2) * 18);
  cudaMemcpy(forward_idata_d, forward_idata_h, sizeof(float2) * 18, cudaMemcpyHostToDevice);

  size_t workSize;
  long long int n[1] = {3};
  long long int inembed[1] = {4};
  long long int onembed[1] = {3};
  cufftXtMakePlanMany(plan_fwd, 1, n, inembed, 2, 9, CUDA_C_32F, onembed, 1, 3, CUDA_C_32F, 2, &workSize, CUDA_C_32F);
  cufftXtExec(plan_fwd, forward_idata_d, forward_odata_d, CUFFT_FORWARD);
  cudaDeviceSynchronize();
  float2 forward_odata_h[6];
  cudaMemcpy(forward_odata_h, forward_odata_d, sizeof(float2) * 6, cudaMemcpyDeviceToHost);

  float2 forward_odata_ref[6];
  forward_odata_ref[0] =  float2{6,9};
  forward_odata_ref[1] =  float2{-4.73205,-1.26795};
  forward_odata_ref[2] =  float2{-1.26795,-4.73205};
  forward_odata_ref[3] =  float2{6,9};
  forward_odata_ref[4] =  float2{-4.73205,-1.26795};
  forward_odata_ref[5] =  float2{-1.26795,-4.73205};

  cufftDestroy(plan_fwd);

  if (!compare(forward_odata_ref, forward_odata_h, 6)) {
    std::cout << "forward_odata_h:" << std::endl;
    print_values(forward_odata_h, 6);
    std::cout << "forward_odata_ref:" << std::endl;
    print_values(forward_odata_ref, 6);

    cudaFree(forward_idata_d);
    cudaFree(forward_odata_d);
    cudaFree(backward_odata_d);

    return false;
  }

  cufftHandle plan_bwd;
  cufftCreate(&plan_bwd);
  cufftXtMakePlanMany(plan_bwd, 1, n, onembed, 1, 3, CUDA_C_32F, inembed, 2, 9, CUDA_C_32F, 2, &workSize, CUDA_C_32F);
  cufftXtExec(plan_bwd, forward_odata_d, backward_odata_d, CUFFT_INVERSE);
  cudaDeviceSynchronize();
  float2 backward_odata_h[18];
  cudaMemcpy(backward_odata_h, backward_odata_d, sizeof(float2) * 18, cudaMemcpyDeviceToHost);

  float2 backward_odata_ref[18];
  backward_odata_ref[0] =  float2{0,3};
  backward_odata_ref[1] =  float2{0,0};
  backward_odata_ref[2] =  float2{6,9};
  backward_odata_ref[3] =  float2{0,0};
  backward_odata_ref[4] =  float2{12,15};
  backward_odata_ref[5] =  float2{0,0};
  backward_odata_ref[6] =  float2{0,0};
  backward_odata_ref[7] =  float2{0,0};
  backward_odata_ref[8] =  float2{0,0};
  backward_odata_ref[9] =  float2{0,3};
  backward_odata_ref[10] = float2{0,0};
  backward_odata_ref[11] = float2{6,9};
  backward_odata_ref[12] = float2{0,0};
  backward_odata_ref[13] = float2{12,15};
  backward_odata_ref[14] = float2{0,0};
  backward_odata_ref[15] = float2{0,0};
  backward_odata_ref[16] = float2{0,0};
  backward_odata_ref[17] = float2{0,0};

  cudaFree(forward_idata_d);
  cudaFree(forward_odata_d);
  cudaFree(backward_odata_d);

  cufftDestroy(plan_bwd);

  std::vector<int> indices = {0, 2, 4,
                              9, 11, 13};
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
#define FUNC c2c_many_1d_outofplace_advanced
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

