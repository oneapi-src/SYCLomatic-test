// ===--- z2z_many_1d_outofplace_advanced.cu -----------------*- CUDA -*---===//
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
bool z2z_many_1d_outofplace_advanced() {
  cufftHandle plan_fwd;
  cufftCreate(&plan_fwd);
  double2 forward_idata_h[18];
  std::memset(forward_idata_h, 0, sizeof(double2) * 18);
  set_value_with_stride(forward_idata_h, 3, 2);
  set_value_with_stride(forward_idata_h + 9, 3, 2);

  double2* forward_idata_d;
  double2* forward_odata_d;
  double2* backward_odata_d;
  cudaMalloc(&forward_idata_d, sizeof(double2) * 18);
  cudaMalloc(&forward_odata_d, sizeof(double2) * 6);
  cudaMalloc(&backward_odata_d, sizeof(double2) * 18);
  cudaMemcpy(forward_idata_d, forward_idata_h, sizeof(double2) * 18, cudaMemcpyHostToDevice);

  size_t workSize;
  long long int n[1] = {3};
  long long int inembed[1] = {4};
  long long int onembed[1] = {3};
  cufftXtMakePlanMany(plan_fwd, 1, n, inembed, 2, 9, CUDA_C_64F, onembed, 1, 3, CUDA_C_64F, 2, &workSize, CUDA_C_64F);
  cufftXtExec(plan_fwd, forward_idata_d, forward_odata_d, CUFFT_FORWARD);
  cudaDeviceSynchronize();
  double2 forward_odata_h[6];
  cudaMemcpy(forward_odata_h, forward_odata_d, sizeof(double2) * 6, cudaMemcpyDeviceToHost);

  double2 forward_odata_ref[6];
  forward_odata_ref[0] =  double2{6,9};
  forward_odata_ref[1] =  double2{-4.73205,-1.26795};
  forward_odata_ref[2] =  double2{-1.26795,-4.73205};
  forward_odata_ref[3] =  double2{6,9};
  forward_odata_ref[4] =  double2{-4.73205,-1.26795};
  forward_odata_ref[5] =  double2{-1.26795,-4.73205};

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
  cufftXtMakePlanMany(plan_bwd, 1, n, onembed, 1, 3, CUDA_C_64F, inembed, 2, 9, CUDA_C_64F, 2, &workSize, CUDA_C_64F);
  cufftXtExec(plan_bwd, forward_odata_d, backward_odata_d, CUFFT_INVERSE);
  cudaDeviceSynchronize();
  double2 backward_odata_h[18];
  cudaMemcpy(backward_odata_h, backward_odata_d, sizeof(double2) * 18, cudaMemcpyDeviceToHost);

  double2 backward_odata_ref[18];
  backward_odata_ref[0] =  double2{0,3};
  backward_odata_ref[1] =  double2{0,0};
  backward_odata_ref[2] =  double2{6,9};
  backward_odata_ref[3] =  double2{0,0};
  backward_odata_ref[4] =  double2{12,15};
  backward_odata_ref[5] =  double2{0,0};
  backward_odata_ref[6] =  double2{0,0};
  backward_odata_ref[7] =  double2{0,0};
  backward_odata_ref[8] =  double2{0,0};
  backward_odata_ref[9] =  double2{0,3};
  backward_odata_ref[10] = double2{0,0};
  backward_odata_ref[11] = double2{6,9};
  backward_odata_ref[12] = double2{0,0};
  backward_odata_ref[13] = double2{12,15};
  backward_odata_ref[14] = double2{0,0};
  backward_odata_ref[15] = double2{0,0};
  backward_odata_ref[16] = double2{0,0};
  backward_odata_ref[17] = double2{0,0};

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
#define FUNC z2z_many_1d_outofplace_advanced
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

