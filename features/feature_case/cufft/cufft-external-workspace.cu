// ====------ cufft-external-workspace.cu ------------------ *- CUDA -* ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include "cufft.h"
#include <cstring>
#include <iostream>

void print_values(float2* ptr, int ele_num) {
  for (int i = 0 ; i < ele_num; i++) {
    std::cout << "(" << ptr[i].x << "," << ptr[i].y << "), " << std::endl;
  }
}

bool compare(float2* ptr1, float2* ptr2, int ele_num) {
  for (int i = 0 ; i < ele_num; i++) {
    if (std::abs(ptr1[i].x - ptr2[i].x) > 0.01 || std::abs(ptr1[i].y - ptr2[i].y) > 0.01) {
      return false;
    }
  }
  return true;
}

bool test() {
  cufftHandle plan_fwd;
  cufftCreate(&plan_fwd);
  float2 forward_idata_h[14];
  forward_idata_h[0] =  float2{0,1};
  forward_idata_h[1] =  float2{2,3};
  forward_idata_h[2] =  float2{4,5};
  forward_idata_h[3] =  float2{6,7};
  forward_idata_h[4] =  float2{8,9};
  forward_idata_h[5] =  float2{10,11};
  forward_idata_h[6] =  float2{12,13};
  forward_idata_h[7] =  float2{0,1};
  forward_idata_h[8] =  float2{2,3};
  forward_idata_h[9] =  float2{4,5};
  forward_idata_h[10] = float2{6,7};
  forward_idata_h[11] = float2{8,9};
  forward_idata_h[12] = float2{10,11};
  forward_idata_h[13] = float2{12,13};

  float2* data_d;
  cudaMalloc(&data_d, 2 * sizeof(float2) * 7);
  cudaMemcpy(data_d, forward_idata_h, 2 * sizeof(float2) * 7, cudaMemcpyHostToDevice);

  size_t workSize;
  void *fwd_workspace;
  cufftSetAutoAllocation(plan_fwd, 0);
  cufftMakePlan1d(plan_fwd, 7, CUFFT_C2C, 2, &workSize);
  cudaMalloc(&fwd_workspace, workSize);
  cufftSetWorkArea(plan_fwd, fwd_workspace);
  cufftExecC2C(plan_fwd, data_d, data_d, CUFFT_FORWARD);
  cudaDeviceSynchronize();
  float2 forward_odata_h[14];
  cudaMemcpy(forward_odata_h, data_d, 2 * sizeof(float2) * 7, cudaMemcpyDeviceToHost);

  float2 forward_odata_ref[14];
  forward_odata_ref[0] =  float2{42,49};
  forward_odata_ref[1] =  float2{-21.5356,7.53565};
  forward_odata_ref[2] =  float2{-12.5823,-1.41769};
  forward_odata_ref[3] =  float2{-8.5977,-5.4023};
  forward_odata_ref[4] =  float2{-5.4023,-8.5977};
  forward_odata_ref[5] =  float2{-1.41769,-12.5823};
  forward_odata_ref[6] =  float2{7.53565,-21.5356};
  forward_odata_ref[7] =  float2{42,49};
  forward_odata_ref[8] =  float2{-21.5356,7.53565};
  forward_odata_ref[9] =  float2{-12.5823,-1.41769};
  forward_odata_ref[10] = float2{-8.5977,-5.4023};
  forward_odata_ref[11] = float2{-5.4023,-8.5977};
  forward_odata_ref[12] = float2{-1.41769,-12.5823};
  forward_odata_ref[13] = float2{7.53565,-21.5356};

  cufftDestroy(plan_fwd);
  cudaFree(fwd_workspace);

  if (!compare(forward_odata_ref, forward_odata_h, 14)) {
    std::cout << "forward_odata_h:" << std::endl;
    print_values(forward_odata_h, 14);
    std::cout << "forward_odata_ref:" << std::endl;
    print_values(forward_odata_ref, 14);

    cudaFree(data_d);

    return false;
  }

  cufftHandle plan_bwd;
  void *bwd_workspace;
  cufftCreate(&plan_bwd);
  cufftSetAutoAllocation(plan_bwd, 0);
  cufftMakePlan1d(plan_bwd, 7, CUFFT_C2C, 2, &workSize);
  cudaMalloc(&bwd_workspace, workSize);
  cufftSetWorkArea(plan_bwd, bwd_workspace);
  cufftExecC2C(plan_bwd, data_d, data_d, CUFFT_INVERSE);
  cudaDeviceSynchronize();
  float2 backward_odata_h[14];
  cudaMemcpy(backward_odata_h, data_d, 2 * sizeof(float2) * 7, cudaMemcpyDeviceToHost);

  float2 backward_odata_ref[14];
  backward_odata_ref[0] =  float2{0,7};
  backward_odata_ref[1] =  float2{14,21};
  backward_odata_ref[2] =  float2{28,35};
  backward_odata_ref[3] =  float2{42,49};
  backward_odata_ref[4] =  float2{56,63};
  backward_odata_ref[5] =  float2{70,77};
  backward_odata_ref[6] =  float2{84,91};
  backward_odata_ref[7] =  float2{0,7};
  backward_odata_ref[8] =  float2{14,21};
  backward_odata_ref[9] =  float2{28,35};
  backward_odata_ref[10] = float2{42,49};
  backward_odata_ref[11] = float2{56,63};
  backward_odata_ref[12] = float2{70,77};
  backward_odata_ref[13] = float2{84,91};

  cudaFree(data_d);
  cufftDestroy(plan_bwd);
  cudaFree(bwd_workspace);

  if (!compare(backward_odata_ref, backward_odata_h, 14)) {
    std::cout << "backward_odata_h:" << std::endl;
    print_values(backward_odata_h, 14);
    std::cout << "backward_odata_ref:" << std::endl;
    print_values(backward_odata_ref, 14);
    return false;
  }
  return true;
}

int main() {
  bool res = test();
  cudaDeviceSynchronize();
  if (!res) {
    std::cout << "Fail" << std::endl;
    return -1;
  }
  std::cout << "Pass" << std::endl;
  return 0;
}
