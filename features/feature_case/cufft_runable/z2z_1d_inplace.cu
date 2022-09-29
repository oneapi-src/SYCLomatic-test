// ===--- z2z_1d_inplace.cu ----------------------------------*- CUDA -*---===//
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


bool z2z_1d_inplace() {
  cufftHandle plan_fwd;
  double2 forward_idata_h[14];
  set_value((double*)forward_idata_h, 14);
  set_value((double*)forward_idata_h + 14, 14);

  double2* data_d;
  cudaMalloc(&data_d, 2 * sizeof(double2) * 7);
  cudaMemcpy(data_d, forward_idata_h, 2 * sizeof(double2) * 7, cudaMemcpyHostToDevice);

  cufftPlan1d(&plan_fwd, 7, CUFFT_Z2Z, 2);
  cufftExecZ2Z(plan_fwd, data_d, data_d, CUFFT_FORWARD);
  cudaDeviceSynchronize();
  double2 forward_odata_h[14];
  cudaMemcpy(forward_odata_h, data_d, 2 * sizeof(double2) * 7, cudaMemcpyDeviceToHost);

  double2 forward_odata_ref[14];
  forward_odata_ref[0] =  double2{42,49};
  forward_odata_ref[1] =  double2{-21.5356,7.53565};
  forward_odata_ref[2] =  double2{-12.5823,-1.41769};
  forward_odata_ref[3] =  double2{-8.5977,-5.4023};
  forward_odata_ref[4] =  double2{-5.4023,-8.5977};
  forward_odata_ref[5] =  double2{-1.41769,-12.5823};
  forward_odata_ref[6] =  double2{7.53565,-21.5356};
  forward_odata_ref[7] =  double2{42,49};
  forward_odata_ref[8] =  double2{-21.5356,7.53565};
  forward_odata_ref[9] =  double2{-12.5823,-1.41769};
  forward_odata_ref[10] = double2{-8.5977,-5.4023};
  forward_odata_ref[11] = double2{-5.4023,-8.5977};
  forward_odata_ref[12] = double2{-1.41769,-12.5823};
  forward_odata_ref[13] = double2{7.53565,-21.5356};

  cufftDestroy(plan_fwd);

  if (!compare(forward_odata_ref, forward_odata_h, 14)) {
    std::cout << "forward_odata_h:" << std::endl;
    print_values(forward_odata_h, 14);
    std::cout << "forward_odata_ref:" << std::endl;
    print_values(forward_odata_ref, 14);

    cudaFree(data_d);

    return false;
  }

  cufftHandle plan_bwd;
  cufftPlan1d(&plan_bwd, 7, CUFFT_Z2Z, 2);
  cufftExecZ2Z(plan_bwd, data_d, data_d, CUFFT_INVERSE);
  cudaDeviceSynchronize();
  double2 backward_odata_h[14];
  cudaMemcpy(backward_odata_h, data_d, 2 * sizeof(double2) * 7, cudaMemcpyDeviceToHost);

  double2 backward_odata_ref[14];
  backward_odata_ref[0] =  double2{0,7};
  backward_odata_ref[1] =  double2{14,21};
  backward_odata_ref[2] =  double2{28,35};
  backward_odata_ref[3] =  double2{42,49};
  backward_odata_ref[4] =  double2{56,63};
  backward_odata_ref[5] =  double2{70,77};
  backward_odata_ref[6] =  double2{84,91};
  backward_odata_ref[7] =  double2{0,7};
  backward_odata_ref[8] =  double2{14,21};
  backward_odata_ref[9] =  double2{28,35};
  backward_odata_ref[10] = double2{42,49};
  backward_odata_ref[11] = double2{56,63};
  backward_odata_ref[12] = double2{70,77};
  backward_odata_ref[13] = double2{84,91};

  cudaFree(data_d);
  cufftDestroy(plan_bwd);

  if (!compare(backward_odata_ref, backward_odata_h, 14)) {
    std::cout << "backward_odata_h:" << std::endl;
    print_values(backward_odata_h, 14);
    std::cout << "backward_odata_ref:" << std::endl;
    print_values(backward_odata_ref, 14);
    return false;
  }
  return true;
}



#ifdef DEBUG_FFT
int main() {
#define FUNC z2z_1d_inplace
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

