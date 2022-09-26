// ===--- z2z_3d_inplace.cu ----------------------------------*- CUDA -*---===//
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


bool z2z_3d_inplace() {
  cufftHandle plan_fwd;
  double2 forward_idata_h[2][3][5];
  set_value((double*)forward_idata_h, 60);

  double2* data_d;
  cudaMalloc(&data_d,sizeof(double2) * 30);
  cudaMemcpy(data_d, forward_idata_h, sizeof(double2) * 30, cudaMemcpyHostToDevice);

  cufftPlan3d(&plan_fwd, 2, 3, 5, CUFFT_Z2Z);
  cufftExecZ2Z(plan_fwd, data_d, data_d, CUFFT_FORWARD);
  cudaDeviceSynchronize();
  double2 forward_odata_h[30];
  cudaMemcpy(forward_odata_h, data_d, sizeof(double2) * 30, cudaMemcpyDeviceToHost);

  double2 forward_odata_ref[30];
  forward_odata_ref[0]  =  double2{870,900};
  forward_odata_ref[1]  =  double2{-71.2915,11.2914};
  forward_odata_ref[2]  =  double2{-39.7476,-20.2524};
  forward_odata_ref[3]  =  double2{-20.2524,-39.7476};
  forward_odata_ref[4]  =  double2{11.2915,-71.2915};
  forward_odata_ref[5]  =  double2{-236.603,-63.3975};
  forward_odata_ref[6]  =  double2{0,0};
  forward_odata_ref[7]  =  double2{0,0};
  forward_odata_ref[8]  =  double2{0,0};
  forward_odata_ref[9]  =  double2{0,0};
  forward_odata_ref[10] =  double2{-63.3975,-236.603};
  forward_odata_ref[11] =  double2{0,0};
  forward_odata_ref[12] =  double2{0,0};
  forward_odata_ref[13] =  double2{0,0};
  forward_odata_ref[14] =  double2{0,0};
  forward_odata_ref[15] =  double2{-450,-450};
  forward_odata_ref[16] =  double2{0,0};
  forward_odata_ref[17] =  double2{0,0};
  forward_odata_ref[18] =  double2{0,0};
  forward_odata_ref[19] =  double2{0,0};
  forward_odata_ref[20] =  double2{0,0};
  forward_odata_ref[21] =  double2{0,0};
  forward_odata_ref[22] =  double2{0,0};
  forward_odata_ref[23] =  double2{0,0};
  forward_odata_ref[24] =  double2{0,0};
  forward_odata_ref[25] =  double2{0,0};
  forward_odata_ref[26] =  double2{0,0};
  forward_odata_ref[27] =  double2{0,0};
  forward_odata_ref[28] =  double2{0,0};
  forward_odata_ref[29] =  double2{0,0};

  cufftDestroy(plan_fwd);

  if (!compare(forward_odata_ref, forward_odata_h, 30)) {
    std::cout << "forward_odata_h:" << std::endl;
    print_values(forward_odata_h, 30);
    std::cout << "forward_odata_ref:" << std::endl;
    print_values(forward_odata_ref, 30);

    cudaFree(data_d);

    return false;
  }

  cufftHandle plan_bwd;
  cufftPlan3d(&plan_bwd, 2, 3, 5, CUFFT_Z2Z);
  cufftExecZ2Z(plan_bwd, data_d, data_d, CUFFT_INVERSE);
  cudaDeviceSynchronize();
  double2 backward_odata_h[30];
  cudaMemcpy(backward_odata_h, data_d, sizeof(double2) * 30, cudaMemcpyDeviceToHost);

  double2 backward_odata_ref[30];
  backward_odata_ref[0]  =  double2{0,30};
  backward_odata_ref[1]  =  double2{60,90};
  backward_odata_ref[2]  =  double2{120,150};
  backward_odata_ref[3]  =  double2{180,210};
  backward_odata_ref[4]  =  double2{240,270};
  backward_odata_ref[5]  =  double2{300,330};
  backward_odata_ref[6]  =  double2{360,390};
  backward_odata_ref[7]  =  double2{420,450};
  backward_odata_ref[8]  =  double2{480,510};
  backward_odata_ref[9]  =  double2{540,570};
  backward_odata_ref[10] =  double2{600,630};
  backward_odata_ref[11] =  double2{660,690};
  backward_odata_ref[12] =  double2{720,750};
  backward_odata_ref[13] =  double2{780,810};
  backward_odata_ref[14] =  double2{840,870};
  backward_odata_ref[15] =  double2{900,930};
  backward_odata_ref[16] =  double2{960,990};
  backward_odata_ref[17] =  double2{1020,1050};
  backward_odata_ref[18] =  double2{1080,1110};
  backward_odata_ref[19] =  double2{1140,1170};
  backward_odata_ref[20] =  double2{1200,1230};
  backward_odata_ref[21] =  double2{1260,1290};
  backward_odata_ref[22] =  double2{1320,1350};
  backward_odata_ref[23] =  double2{1380,1410};
  backward_odata_ref[24] =  double2{1440,1470};
  backward_odata_ref[25] =  double2{1500,1530};
  backward_odata_ref[26] =  double2{1560,1590};
  backward_odata_ref[27] =  double2{1620,1650};
  backward_odata_ref[28] =  double2{1680,1710};
  backward_odata_ref[29] =  double2{1740,1770};

  cudaFree(data_d);
  cufftDestroy(plan_bwd);

  if (!compare(backward_odata_ref, backward_odata_h, 30)) {
    std::cout << "backward_odata_h:" << std::endl;
    print_values(backward_odata_h, 30);
    std::cout << "backward_odata_ref:" << std::endl;
    print_values(backward_odata_ref, 30);
    return false;
  }
  return true;
}


#ifdef DEBUG_FFT
int main() {
#define FUNC z2z_3d_inplace
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

