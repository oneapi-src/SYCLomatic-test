// ===--- d2zz2d_many_2d_inplace_advanced.cu -----------------*- CUDA -*---===//
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
// +---+---+---+---+         -+
// | r | r | r | 0 |          |
// +---+---+---+---+          |
// | r | r | r | 0 |          batch0
// +---+---+---+---+         -+
// | r | r | r | 0 |          |
// +---+---+---+---+          |
// | r | r | r | 0 |          batch1
// +---+---+---+---+         -+
// |______n2_______|
// |___nembed2_____|
// output
// +---+---+---+---+         -+
// |   c   |   c   |          |
// +---+---+---+---+          |
// |   c   |   c   |          batch0
// +---+---+---+---+         -+
// |   c   |   c   |          |
// +---+---+---+---+          |
// |   c   |   c   |          batch1
// +---+---+---+---+         -+
// |______n2_______|
// |___nembed2_____|
bool d2zz2d_many_2d_inplace_advanced() {
  cufftHandle plan_fwd;
  cufftCreate(&plan_fwd);
  double forward_idata_h[16];
  std::memset(forward_idata_h, 0, sizeof(double) * 16);
  forward_idata_h[0]  = 0;
  forward_idata_h[1]  = 1;
  forward_idata_h[2]  = 2;
  forward_idata_h[4]  = 3;
  forward_idata_h[5]  = 4;
  forward_idata_h[6]  = 5;
  forward_idata_h[8]  = 0;
  forward_idata_h[9]  = 1;
  forward_idata_h[10] = 2;
  forward_idata_h[12] = 3;
  forward_idata_h[13] = 4;
  forward_idata_h[14] = 5;

  double* data_d;
  cudaMalloc(&data_d, sizeof(double) * 16);
  cudaMemcpy(data_d, forward_idata_h, sizeof(double) * 16, cudaMemcpyHostToDevice);

  size_t workSize;
  long long int n[2] = {2, 3};
  long long int inembed[2] = {2, 4};
  long long int onembed[2] = {2, 2};
  cufftMakePlanMany64(plan_fwd, 2, n, inembed, 1, 8, onembed, 1, 4, CUFFT_D2Z, 2, &workSize);
  cufftExecD2Z(plan_fwd, data_d, (double2*)data_d);
  cudaDeviceSynchronize();
  double2 forward_odata_h[8];
  cudaMemcpy(forward_odata_h, data_d, sizeof(double) * 16, cudaMemcpyDeviceToHost);

  double2 forward_odata_ref[8];
  forward_odata_ref[0] = double2{15, 0};
  forward_odata_ref[1] = double2{-3, 1.73205};
  forward_odata_ref[2] = double2{-9, 0};
  forward_odata_ref[3] = double2{0, 0};
  forward_odata_ref[4] = double2{15, 0};
  forward_odata_ref[5] = double2{-3, 1.73205};
  forward_odata_ref[6] = double2{-9, 0};
  forward_odata_ref[7] = double2{0, 0};

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
  cufftMakePlanMany64(plan_bwd, 2, n, onembed, 1, 4, inembed, 1, 8, CUFFT_Z2D, 2, &workSize);
  cufftExecZ2D(plan_bwd, (double2*)data_d, data_d);
  cudaDeviceSynchronize();
  double backward_odata_h[16];
  cudaMemcpy(backward_odata_h, data_d, sizeof(double) * 16, cudaMemcpyDeviceToHost);

  double backward_odata_ref[16];
  backward_odata_ref[0]  = 0;
  backward_odata_ref[1]  = 6;
  backward_odata_ref[2]  = 12;
  backward_odata_ref[4]  = 18;
  backward_odata_ref[5]  = 24;
  backward_odata_ref[6]  = 30;
  backward_odata_ref[8]  = 0;
  backward_odata_ref[9]  = 6;
  backward_odata_ref[10] = 12;
  backward_odata_ref[12] = 18;
  backward_odata_ref[13] = 24;
  backward_odata_ref[14] = 30;

  cudaFree(data_d);
  cufftDestroy(plan_bwd);

  std::vector<int> indices_bwd = {0, 1, 2, 4, 5, 6,
                                  8, 9, 10, 12, 13, 14};
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
#define FUNC d2zz2d_many_2d_inplace_advanced
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

