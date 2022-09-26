// ===--- z2z_many_3d_outofplace_advanced.cu -----------------*- CUDA -*---===//
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
bool z2z_many_3d_outofplace_advanced() {
  cufftHandle plan_fwd;
  cufftCreate(&plan_fwd);
  double2 forward_idata_h[48];
  std::memset(forward_idata_h, 0, sizeof(double2) * 48);
  set_value((double*)forward_idata_h, 48);
  set_value((double*)forward_idata_h + 48, 48);

  double2* forward_idata_d;
  double2* forward_odata_d;
  double2* backward_odata_d;
  cudaMalloc(&forward_idata_d, sizeof(double2) * 48);
  cudaMalloc(&forward_odata_d, sizeof(double2) * 48);
  cudaMalloc(&backward_odata_d, sizeof(double2) * 48);
  cudaMemcpy(forward_idata_d, forward_idata_h, sizeof(double2) * 48, cudaMemcpyHostToDevice);

  size_t workSize;
  long long int n[3] = {2, 3, 4};
  long long int inembed[3] = {2, 3, 4};
  long long int onembed[3] = {2, 3, 4};
  cufftXtMakePlanMany(plan_fwd, 3, n, inembed, 1, 24, CUDA_C_64F, onembed, 1, 24, CUDA_C_64F, 2, &workSize, CUDA_C_64F);
  cufftXtExec(plan_fwd, forward_idata_d, forward_odata_d, CUFFT_FORWARD);
  cudaDeviceSynchronize();
  double2 forward_odata_h[48];
  cudaMemcpy(forward_odata_h, forward_odata_d, sizeof(double2) * 48, cudaMemcpyDeviceToHost);

  double2 forward_odata_ref[48];
  forward_odata_ref[0] =  double2{552,576};
  forward_odata_ref[1] =  double2{-48,0};
  forward_odata_ref[2] =  double2{-24,-24};
  forward_odata_ref[3] =  double2{0,-48};
  forward_odata_ref[4] =  double2{-151.426,-40.5744};
  forward_odata_ref[5] =  double2{0,0};
  forward_odata_ref[6] =  double2{0,0};
  forward_odata_ref[7] =  double2{0,0};
  forward_odata_ref[8] =  double2{-40.5744,-151.426};
  forward_odata_ref[9] =  double2{0,0};
  forward_odata_ref[10] = double2{0,0};
  forward_odata_ref[11] = double2{0,0};
  forward_odata_ref[12] = double2{-288,-288};
  forward_odata_ref[13] = double2{0,0};
  forward_odata_ref[14] = double2{0,0};
  forward_odata_ref[15] = double2{0,0};
  forward_odata_ref[16] = double2{0,0};
  forward_odata_ref[17] = double2{0,0};
  forward_odata_ref[18] = double2{0,0};
  forward_odata_ref[19] = double2{0,0};
  forward_odata_ref[20] = double2{0,0};
  forward_odata_ref[21] = double2{0,0};
  forward_odata_ref[22] = double2{0,0};
  forward_odata_ref[23] = double2{0,0};
  forward_odata_ref[24] = double2{552,576};
  forward_odata_ref[25] = double2{-48,0};
  forward_odata_ref[26] = double2{-24,-24};
  forward_odata_ref[27] = double2{0,-48};
  forward_odata_ref[28] = double2{-151.426,-40.5744};
  forward_odata_ref[29] = double2{0,0};
  forward_odata_ref[30] = double2{0,0};
  forward_odata_ref[31] = double2{0,0};
  forward_odata_ref[32] = double2{-40.5744,-151.426};
  forward_odata_ref[33] = double2{0,0};
  forward_odata_ref[34] = double2{0,0};
  forward_odata_ref[35] = double2{0,0};
  forward_odata_ref[36] = double2{-288,-288};
  forward_odata_ref[37] = double2{0,0};
  forward_odata_ref[38] = double2{0,0};
  forward_odata_ref[39] = double2{0,0};
  forward_odata_ref[40] = double2{0,0};
  forward_odata_ref[41] = double2{0,0};
  forward_odata_ref[42] = double2{0,0};
  forward_odata_ref[43] = double2{0,0};
  forward_odata_ref[44] = double2{0,0};
  forward_odata_ref[45] = double2{0,0};
  forward_odata_ref[46] = double2{0,0};
  forward_odata_ref[47] = double2{0,0};

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
  cufftXtMakePlanMany(plan_bwd, 3, n, onembed, 1, 24, CUDA_C_64F, inembed, 1, 24, CUDA_C_64F, 2, &workSize, CUDA_C_64F);
  cufftXtExec(plan_bwd, forward_odata_d, backward_odata_d, CUFFT_INVERSE);
  cudaDeviceSynchronize();
  double2 backward_odata_h[48];
  cudaMemcpy(backward_odata_h, backward_odata_d, sizeof(double2) * 48, cudaMemcpyDeviceToHost);

  double2 backward_odata_ref[48] = {
    double2{0, 24},
    double2{48, 72},
    double2{96, 120},
    double2{144, 168},
    double2{192, 216},
    double2{240, 264},
    double2{288, 312},
    double2{336, 360},
    double2{384, 408},
    double2{432, 456},
    double2{480, 504},
    double2{528, 552},
    double2{576, 600},
    double2{624, 648},
    double2{672, 696},
    double2{720, 744},
    double2{768, 792},
    double2{816, 840},
    double2{864, 888},
    double2{912, 936},
    double2{960, 984},
    double2{1008, 1032},
    double2{1056, 1080},
    double2{1104, 1128},
    double2{0, 24},
    double2{48, 72},
    double2{96, 120},
    double2{144, 168},
    double2{192, 216},
    double2{240, 264},
    double2{288, 312},
    double2{336, 360},
    double2{384, 408},
    double2{432, 456},
    double2{480, 504},
    double2{528, 552},
    double2{576, 600},
    double2{624, 648},
    double2{672, 696},
    double2{720, 744},
    double2{768, 792},
    double2{816, 840},
    double2{864, 888},
    double2{912, 936},
    double2{960, 984},
    double2{1008, 1032},
    double2{1056, 1080},
    double2{1104, 1128}
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
#define FUNC z2z_many_3d_outofplace_advanced
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

