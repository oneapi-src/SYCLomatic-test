// ===--- c2c_3d_inplace.cu ----------------------------------*- CUDA -*---===//
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

bool c2c_3d_inplace() {
  cufftHandle plan_fwd;
  float2 forward_idata_h[2][3][5];
  set_value((float*)forward_idata_h, 60);

  float2* data_d;
  cudaMalloc(&data_d,sizeof(float2) * 30);
  cudaMemcpy(data_d, forward_idata_h, sizeof(float2) * 30, cudaMemcpyHostToDevice);

  cufftPlan3d(&plan_fwd, 2, 3, 5, CUFFT_C2C);
  cufftExecC2C(plan_fwd, data_d, data_d, CUFFT_FORWARD);
  cudaDeviceSynchronize();
  float2 forward_odata_h[30];
  cudaMemcpy(forward_odata_h, data_d, sizeof(float2) * 30, cudaMemcpyDeviceToHost);

  float2 forward_odata_ref[30];
  forward_odata_ref[0]  =  float2{870,900};
  forward_odata_ref[1]  =  float2{-71.2915,11.2914};
  forward_odata_ref[2]  =  float2{-39.7476,-20.2524};
  forward_odata_ref[3]  =  float2{-20.2524,-39.7476};
  forward_odata_ref[4]  =  float2{11.2915,-71.2915};
  forward_odata_ref[5]  =  float2{-236.603,-63.3975};
  forward_odata_ref[6]  =  float2{0,0};
  forward_odata_ref[7]  =  float2{0,0};
  forward_odata_ref[8]  =  float2{0,0};
  forward_odata_ref[9]  =  float2{0,0};
  forward_odata_ref[10] =  float2{-63.3975,-236.603};
  forward_odata_ref[11] =  float2{0,0};
  forward_odata_ref[12] =  float2{0,0};
  forward_odata_ref[13] =  float2{0,0};
  forward_odata_ref[14] =  float2{0,0};
  forward_odata_ref[15] =  float2{-450,-450};
  forward_odata_ref[16] =  float2{0,0};
  forward_odata_ref[17] =  float2{0,0};
  forward_odata_ref[18] =  float2{0,0};
  forward_odata_ref[19] =  float2{0,0};
  forward_odata_ref[20] =  float2{0,0};
  forward_odata_ref[21] =  float2{0,0};
  forward_odata_ref[22] =  float2{0,0};
  forward_odata_ref[23] =  float2{0,0};
  forward_odata_ref[24] =  float2{0,0};
  forward_odata_ref[25] =  float2{0,0};
  forward_odata_ref[26] =  float2{0,0};
  forward_odata_ref[27] =  float2{0,0};
  forward_odata_ref[28] =  float2{0,0};
  forward_odata_ref[29] =  float2{0,0};

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
  cufftPlan3d(&plan_bwd, 2, 3, 5, CUFFT_C2C);
  cufftExecC2C(plan_bwd, data_d, data_d, CUFFT_INVERSE);
  cudaDeviceSynchronize();
  float2 backward_odata_h[30];
  cudaMemcpy(backward_odata_h, data_d, sizeof(float2) * 30, cudaMemcpyDeviceToHost);

  float2 backward_odata_ref[30];
  backward_odata_ref[0]  =  float2{0,30};
  backward_odata_ref[1]  =  float2{60,90};
  backward_odata_ref[2]  =  float2{120,150};
  backward_odata_ref[3]  =  float2{180,210};
  backward_odata_ref[4]  =  float2{240,270};
  backward_odata_ref[5]  =  float2{300,330};
  backward_odata_ref[6]  =  float2{360,390};
  backward_odata_ref[7]  =  float2{420,450};
  backward_odata_ref[8]  =  float2{480,510};
  backward_odata_ref[9]  =  float2{540,570};
  backward_odata_ref[10] =  float2{600,630};
  backward_odata_ref[11] =  float2{660,690};
  backward_odata_ref[12] =  float2{720,750};
  backward_odata_ref[13] =  float2{780,810};
  backward_odata_ref[14] =  float2{840,870};
  backward_odata_ref[15] =  float2{900,930};
  backward_odata_ref[16] =  float2{960,990};
  backward_odata_ref[17] =  float2{1020,1050};
  backward_odata_ref[18] =  float2{1080,1110};
  backward_odata_ref[19] =  float2{1140,1170};
  backward_odata_ref[20] =  float2{1200,1230};
  backward_odata_ref[21] =  float2{1260,1290};
  backward_odata_ref[22] =  float2{1320,1350};
  backward_odata_ref[23] =  float2{1380,1410};
  backward_odata_ref[24] =  float2{1440,1470};
  backward_odata_ref[25] =  float2{1500,1530};
  backward_odata_ref[26] =  float2{1560,1590};
  backward_odata_ref[27] =  float2{1620,1650};
  backward_odata_ref[28] =  float2{1680,1710};
  backward_odata_ref[29] =  float2{1740,1770};

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
#define FUNC c2c_3d_inplace
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

