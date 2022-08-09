// ===--- c2c_many_3d_outofplace_advanced.cu -----------------*- CUDA -*---===//
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
// +---+---+---+---+---+---+---+---+  ---+              ----+       
// |   c   |   c   |   c   |   c   |     |                  |       
// +---+---+---+---+---+---+---+---+     n2/nembed2         |     
// |   c   |   c   |   c   |   c   |     |                  |                    
// +---+---+---+---+---+---+---+---+     |                  |            
// |   c   |   c   |   c   |   c   |     |                  |            
// +---+---+---+---+---+---+---+---+  ---+          n1/nembed1/a batch
// |   c   |   c   |   c   |   c   |     |                  |            
// +---+---+---+---+---+---+---+---+     n2/nembed2         |          
// |   c   |   c   |   c   |   c   |     |                  |        
// +---+---+---+---+---+---+---+---+     |                  |        
// |   c   |   c   |   c   |   c   |     |                  |        
// +---+---+---+---+---+---+---+---+  ---+              ----+     
// output
// +---+---+---+---+---+---+---+---+  ---+              ----+       
// |   c   |   c   |   c   |   c   |     |                  |       
// +---+---+---+---+---+---+---+---+     n2/nembed2         |     
// |   c   |   c   |   c   |   c   |     |                  |                    
// +---+---+---+---+---+---+---+---+     |                  |            
// |   c   |   c   |   c   |   c   |     |                  |            
// +---+---+---+---+---+---+---+---+  ---+          n1/nembed1/a batch
// |   c   |   c   |   c   |   c   |     |                  |            
// +---+---+---+---+---+---+---+---+     n2/nembed2         |          
// |   c   |   c   |   c   |   c   |     |                  |        
// +---+---+---+---+---+---+---+---+     |                  |        
// |   c   |   c   |   c   |   c   |     |                  |        
// +---+---+---+---+---+---+---+---+  ---+              ----+     
// |______________n3_______________|
// |____________nembed3____________|
bool c2c_many_3d_outofplace_advanced() {
  cufftHandle plan_fwd;
  cufftCreate(&plan_fwd);
  float2 forward_idata_h[48];
  std::memset(forward_idata_h, 0, sizeof(float2) * 48);
  set_value((float*)forward_idata_h, 48);
  set_value((float*)forward_idata_h + 48, 48);

  float2* forward_idata_d;
  float2* forward_odata_d;
  float2* backward_odata_d;
  cudaMalloc(&forward_idata_d, sizeof(float2) * 48);
  cudaMalloc(&forward_odata_d, sizeof(float2) * 48);
  cudaMalloc(&backward_odata_d, sizeof(float2) * 48);
  cudaMemcpy(forward_idata_d, forward_idata_h, sizeof(float2) * 48, cudaMemcpyHostToDevice);

  size_t workSize;
  long long int n[3] = {2, 3, 4};
  long long int inembed[3] = {2, 3, 4};
  long long int onembed[3] = {2, 3, 4};
  cufftXtMakePlanMany(plan_fwd, 3, n, inembed, 1, 24, CUDA_C_32F, onembed, 1, 24, CUDA_C_32F, 2, &workSize, CUDA_C_32F);
  cufftXtExec(plan_fwd, forward_idata_d, forward_odata_d, CUFFT_FORWARD);
  cudaDeviceSynchronize();
  float2 forward_odata_h[48];
  cudaMemcpy(forward_odata_h, forward_odata_d, sizeof(float2) * 48, cudaMemcpyDeviceToHost);

  float2 forward_odata_ref[48];
  forward_odata_ref[0] =  float2{552,576};
  forward_odata_ref[1] =  float2{-48,0};
  forward_odata_ref[2] =  float2{-24,-24};
  forward_odata_ref[3] =  float2{0,-48};
  forward_odata_ref[4] =  float2{-151.426,-40.5744};
  forward_odata_ref[5] =  float2{0,0};
  forward_odata_ref[6] =  float2{0,0};
  forward_odata_ref[7] =  float2{0,0};
  forward_odata_ref[8] =  float2{-40.5744,-151.426};
  forward_odata_ref[9] =  float2{0,0};
  forward_odata_ref[10] = float2{0,0};
  forward_odata_ref[11] = float2{0,0};
  forward_odata_ref[12] = float2{-288,-288};
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
  forward_odata_ref[24] = float2{552,576};
  forward_odata_ref[25] = float2{-48,0};
  forward_odata_ref[26] = float2{-24,-24};
  forward_odata_ref[27] = float2{0,-48};
  forward_odata_ref[28] = float2{-151.426,-40.5744};
  forward_odata_ref[29] = float2{0,0};
  forward_odata_ref[30] = float2{0,0};
  forward_odata_ref[31] = float2{0,0};
  forward_odata_ref[32] = float2{-40.5744,-151.426};
  forward_odata_ref[33] = float2{0,0};
  forward_odata_ref[34] = float2{0,0};
  forward_odata_ref[35] = float2{0,0};
  forward_odata_ref[36] = float2{-288,-288};
  forward_odata_ref[37] = float2{0,0};
  forward_odata_ref[38] = float2{0,0};
  forward_odata_ref[39] = float2{0,0};
  forward_odata_ref[40] = float2{0,0};
  forward_odata_ref[41] = float2{0,0};
  forward_odata_ref[42] = float2{0,0};
  forward_odata_ref[43] = float2{0,0};
  forward_odata_ref[44] = float2{0,0};
  forward_odata_ref[45] = float2{0,0};
  forward_odata_ref[46] = float2{0,0};
  forward_odata_ref[47] = float2{0,0};

  cufftDestroy(plan_fwd);

  if (!compare(forward_odata_ref, forward_odata_h, 48)) {
    std::cout << "forward_odata_h:" << std::endl;
    print_values(forward_odata_h, 48);
    std::cout << "forward_odata_ref:" << std::endl;
    print_values(forward_odata_ref, 48);

    cudaFree(forward_idata_d);
    cudaFree(forward_odata_d);
    cudaFree(backward_odata_d);

    return false;
  }

  cufftHandle plan_bwd;
  cufftCreate(&plan_bwd);
  cufftXtMakePlanMany(plan_bwd, 3, n, onembed, 1, 24, CUDA_C_32F, inembed, 1, 24, CUDA_C_32F, 2, &workSize, CUDA_C_32F);
  cufftXtExec(plan_bwd, forward_odata_d, backward_odata_d, CUFFT_INVERSE);
  cudaDeviceSynchronize();
  float2 backward_odata_h[48];
  cudaMemcpy(backward_odata_h, backward_odata_d, sizeof(float2) * 48, cudaMemcpyDeviceToHost);

  float2 backward_odata_ref[48] = {
    float2{0, 24},
    float2{48, 72},
    float2{96, 120},
    float2{144, 168},
    float2{192, 216},
    float2{240, 264},
    float2{288, 312},
    float2{336, 360},
    float2{384, 408},
    float2{432, 456},
    float2{480, 504},
    float2{528, 552},
    float2{576, 600},
    float2{624, 648},
    float2{672, 696},
    float2{720, 744},
    float2{768, 792},
    float2{816, 840},
    float2{864, 888},
    float2{912, 936},
    float2{960, 984},
    float2{1008, 1032},
    float2{1056, 1080},
    float2{1104, 1128},
    float2{0, 24},
    float2{48, 72},
    float2{96, 120},
    float2{144, 168},
    float2{192, 216},
    float2{240, 264},
    float2{288, 312},
    float2{336, 360},
    float2{384, 408},
    float2{432, 456},
    float2{480, 504},
    float2{528, 552},
    float2{576, 600},
    float2{624, 648},
    float2{672, 696},
    float2{720, 744},
    float2{768, 792},
    float2{816, 840},
    float2{864, 888},
    float2{912, 936},
    float2{960, 984},
    float2{1008, 1032},
    float2{1056, 1080},
    float2{1104, 1128}
  };

  cudaFree(forward_idata_d);
  cudaFree(forward_odata_d);
  cudaFree(backward_odata_d);

  cufftDestroy(plan_bwd);

  if (!compare(backward_odata_ref, backward_odata_h, 48)) {
    std::cout << "backward_odata_h:" << std::endl;
    print_values(backward_odata_h, 48);
    std::cout << "backward_odata_ref:" << std::endl;
    print_values(backward_odata_ref, 48);
    return false;
  }
  return true;
}


#ifdef DEBUG_FFT
int main() {
#define FUNC c2c_many_3d_outofplace_advanced
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

