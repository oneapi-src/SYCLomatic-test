// ===--- c2c_many_2d_inplace_advanced.cu --------------------*- CUDA -*---===//
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
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+         -+
// |   c   |   0   |   c   |   0   |   c   |   0   |   0   |   0   |          |
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+          |
// |   c   |   0   |   c   |   0   |   c   |   0   |   0   |   0   |          batch0
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+  |  
// |   0   |   0   |   0   |   0   |   0   |   0   |   0   |   0   |   0   |  |
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+ -+
// |   c   |   0   |   c   |   0   |   c   |   0   |   0   |   0   |          |
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+          |
// |   c   |   0   |   c   |   0   |   c   |   0   |   0   |   0   |          batch1
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+  |  
// |   0   |   0   |   0   |   0   |   0   |   0   |   0   |   0   |   0   |  |
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+ -+
// |___________n2__________|
// |________nembed2________|
// output
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+         -+
// |   c   |   0   |   c   |   0   |   c   |   0   |   0   |   0   |          |
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+          |
// |   c   |   0   |   c   |   0   |   c   |   0   |   0   |   0   |          batch0
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+  |  
// |   0   |   0   |   0   |   0   |   0   |   0   |   0   |   0   |   0   |  |
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+ -+
// |   c   |   0   |   c   |   0   |   c   |   0   |   0   |   0   |          |
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+          |
// |   c   |   0   |   c   |   0   |   c   |   0   |   0   |   0   |          batch1
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+  |  
// |   0   |   0   |   0   |   0   |   0   |   0   |   0   |   0   |   0   |  |
// +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+ -+
// |_______________________n2______________________|               |
// |____________________________nembed2____________________________|
bool c2c_many_2d_inplace_advanced() {
  cufftHandle plan_fwd;
  cufftCreate(&plan_fwd);
  float2 forward_idata_h[50];
  std::memset(forward_idata_h, 0, sizeof(float2) * 50);
  forward_idata_h[0] = float2{0, 1};
  forward_idata_h[2] = float2{2, 3};
  forward_idata_h[4] = float2{4, 5};
  forward_idata_h[8] = float2{6, 7};
  forward_idata_h[10] = float2{8, 9};
  forward_idata_h[12] = float2{10, 11};
  forward_idata_h[25] = float2{0, 1};
  forward_idata_h[27] = float2{2, 3};
  forward_idata_h[29] = float2{4, 5};
  forward_idata_h[33] = float2{6, 7};
  forward_idata_h[35] = float2{8, 9};
  forward_idata_h[37] = float2{10, 11};

  float2* data_d;
  cudaMalloc(&data_d, sizeof(float2) * 50);
  cudaMemcpy(data_d, forward_idata_h, sizeof(float2) * 50, cudaMemcpyHostToDevice);

  size_t workSize;
  long long int n[2] = {2, 3};
  long long int inembed[2] = {3, 4};
  long long int onembed[2] = {3, 4};
  cufftMakePlanMany64(plan_fwd, 2, n, inembed, 2, 25, onembed, 2, 25, CUFFT_C2C, 2, &workSize);
  cufftExecC2C(plan_fwd, data_d, data_d, CUFFT_FORWARD);
  cudaDeviceSynchronize();
  float2 forward_odata_h[50];
  cudaMemcpy(forward_odata_h, data_d, sizeof(float2) * 50, cudaMemcpyDeviceToHost);

  float2 forward_odata_ref[50];
  forward_odata_ref[0] =  float2{30,36};
  forward_odata_ref[1] =  float2{2,3};
  forward_odata_ref[2] =  float2{-9.4641,-2.5359};
  forward_odata_ref[3] =  float2{6,7};
  forward_odata_ref[4] =  float2{-2.5359,-9.4641};
  forward_odata_ref[5] =  float2{10,11};
  forward_odata_ref[6] =  float2{0,0};
  forward_odata_ref[7] =  float2{0,0};
  forward_odata_ref[8] =  float2{-18,-18};
  forward_odata_ref[9] =  float2{0,0};
  forward_odata_ref[10] = float2{0,0};
  forward_odata_ref[11] = float2{0,0};
  forward_odata_ref[12] = float2{0,0};
  forward_odata_ref[13] = float2{0,0};
  forward_odata_ref[14] = float2{0,0};
  forward_odata_ref[15] = float2{0,0};
  forward_odata_ref[16] = float2{0,0};
  forward_odata_ref[17] = float2{0,0};
  forward_odata_ref[18] = float2{0,0};
  forward_odata_ref[19] = float2{0,0};
  forward_odata_ref[20] = float2{0,0};
  forward_odata_ref[21] = float2{0,0};
  forward_odata_ref[22] = float2{0,0};
  forward_odata_ref[23] = float2{0,0};
  forward_odata_ref[24] = float2{0,0};
  forward_odata_ref[25] = float2{30,36};
  forward_odata_ref[26] = float2{2,3};
  forward_odata_ref[27] = float2{-9.4641,-2.5359};
  forward_odata_ref[28] = float2{6,7};
  forward_odata_ref[29] = float2{-2.5359,-9.4641};
  forward_odata_ref[30] = float2{10,11};
  forward_odata_ref[31] = float2{0,0};
  forward_odata_ref[32] = float2{0,0};
  forward_odata_ref[33] = float2{-18,-18};
  forward_odata_ref[34] = float2{0,0};
  forward_odata_ref[35] = float2{0,0};
  forward_odata_ref[36] = float2{0,0};
  forward_odata_ref[37] = float2{0,0};

  cufftDestroy(plan_fwd);

  std::vector<int> indices = {0, 2, 4,
                              8, 10, 12,
                              25, 27, 29,
                              33, 35, 37};
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
  cufftMakePlanMany64(plan_bwd, 2, n, onembed, 2, 25, inembed, 2, 25, CUFFT_C2C, 2, &workSize);
  cufftExecC2C(plan_bwd, data_d, data_d, CUFFT_INVERSE);
  cudaDeviceSynchronize();
  float2 backward_odata_h[50];
  cudaMemcpy(backward_odata_h, data_d, sizeof(float2) * 50, cudaMemcpyDeviceToHost);

  float2 backward_odata_ref[50];
  backward_odata_ref[0] = float2{0, 6};
  backward_odata_ref[2] = float2{12, 18};
  backward_odata_ref[4] = float2{24, 30};
  backward_odata_ref[8] = float2{36, 42};
  backward_odata_ref[10] = float2{48, 54};
  backward_odata_ref[12] = float2{60, 66};
  backward_odata_ref[25] = float2{0, 6};
  backward_odata_ref[27] = float2{12, 18};
  backward_odata_ref[29] = float2{24, 30};
  backward_odata_ref[33] = float2{36, 42};
  backward_odata_ref[35] = float2{48, 54};
  backward_odata_ref[37] = float2{60, 66};

  cudaFree(data_d);
  cufftDestroy(plan_bwd);

  std::vector<int> indices_bwd = {0, 2, 4, 8, 10, 12,
                                  25, 27, 29, 33, 35, 37};
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
#define FUNC c2c_many_2d_inplace_advanced
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

