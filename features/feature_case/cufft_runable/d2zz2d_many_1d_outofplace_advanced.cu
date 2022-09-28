// ===--- d2zz2d_many_1d_outofplace_advanced.cu --------------*- CUDA -*---===//
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
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
// | r | 0 | r | 0 | r | 0 | r | 0 | r | 0 | 0 | 0 | 0 | r | 0 | r | 0 | r | 0 | r | 0 | r | 0 | 0 | 0 | 0 |
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
// |___________________n___________________|       |   |___________________n___________________|       |   |
// |___________________nembed______________________|   |___________________nembed______________________|   |
// |_______________________batch0______________________|________________________batch1_____________________|
// output
// +---+---+---+---+---+---+---+---+---+---+---+---+
// |   c   |   c   |   c   |   c   |   c   |   c   |
// +---+---+---+---+---+---+---+---+---+---+---+---+
// |___________n___________|___________n___________|
// |_________nembed________|_________nembed________|
// |_________batch0________|_________batch1________|
bool d2zz2d_many_1d_outofplace_advanced() {
  cufftHandle plan_fwd;
  cufftCreate(&plan_fwd);
  double forward_idata_h[26];
  std::memset(forward_idata_h, 0, sizeof(double) * 26);
  set_value_with_stride(forward_idata_h, 5, 2);
  set_value_with_stride(forward_idata_h + 13, 5, 2);

  double* forward_idata_d;
  double2* forward_odata_d;
  double* backward_odata_d;
  cudaMalloc(&forward_idata_d, sizeof(double) * 26);
  cudaMalloc(&forward_odata_d, sizeof(double2) * 6);
  cudaMalloc(&backward_odata_d, sizeof(double) * 26);
  cudaMemcpy(forward_idata_d, forward_idata_h, sizeof(double) * 26, cudaMemcpyHostToDevice);

  long long int n[1] = {5};
  long long int inembed[1] = {12};
  long long int onembed[1] = {3};
  size_t workSize;
  cufftXtMakePlanMany(plan_fwd, 1, n, inembed, 2, 13, CUDA_R_64F, onembed, 1, 3, CUDA_C_64F, 2, &workSize, CUDA_C_64F);
  cufftXtExec(plan_fwd, forward_idata_d, forward_odata_d, CUFFT_FORWARD);
  cudaDeviceSynchronize();
  double2 forward_odata_h[6];
  cudaMemcpy(forward_odata_h, forward_odata_d, sizeof(double2) * 6, cudaMemcpyDeviceToHost);

  double2 forward_odata_ref[6];
  forward_odata_ref[0] =  double2{10,0};
  forward_odata_ref[1] =  double2{-2.5,3.44095};
  forward_odata_ref[2] =  double2{-2.5,0.812299};
  forward_odata_ref[3] =  double2{10,0};
  forward_odata_ref[4] =  double2{-2.5,3.44095};
  forward_odata_ref[5] =  double2{-2.5,0.812299};

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
  cufftXtMakePlanMany(plan_bwd, 1, n, onembed, 1, 3, CUDA_C_64F, inembed, 2, 13, CUDA_R_64F, 2, &workSize, CUDA_C_64F);
  cufftXtExec(plan_bwd, forward_odata_d, backward_odata_d, CUFFT_INVERSE);
  cudaDeviceSynchronize();
  double backward_odata_h[26];
  cudaMemcpy(backward_odata_h, backward_odata_d, sizeof(double) * 26, cudaMemcpyDeviceToHost);

  double backward_odata_ref[26];
  backward_odata_ref[0]  = 0;
  backward_odata_ref[1]  = 0;
  backward_odata_ref[2]  = 5;
  backward_odata_ref[3]  = 0;
  backward_odata_ref[4]  = 10;
  backward_odata_ref[5]  = 0;
  backward_odata_ref[6]  = 15;
  backward_odata_ref[7]  = 0;
  backward_odata_ref[8]  = 20;
  backward_odata_ref[9]  = 0;
  backward_odata_ref[10] = 0;
  backward_odata_ref[11] = 0;
  backward_odata_ref[12] = 0;
  backward_odata_ref[13] = 0;
  backward_odata_ref[14] = 0;
  backward_odata_ref[15] = 5;
  backward_odata_ref[16] = 0;
  backward_odata_ref[17] = 10;
  backward_odata_ref[18] = 0;
  backward_odata_ref[19] = 15;
  backward_odata_ref[20] = 0;
  backward_odata_ref[21] = 20;
  backward_odata_ref[22] = 0;
  backward_odata_ref[23] = 0;
  backward_odata_ref[24] = 0;
  backward_odata_ref[25] = 0;

  cudaFree(forward_idata_d);
  cudaFree(forward_odata_d);
  cudaFree(backward_odata_d);

  cufftDestroy(plan_bwd);

  std::vector<int> indices = {0, 2, 4, 6, 8,
                              13, 15, 17, 19, 21};
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
#define FUNC d2zz2d_many_1d_outofplace_advanced
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

