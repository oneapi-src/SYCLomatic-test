// ===------ FftUtils_api_test3.cu------------------------- *- CUDA -* ---=== //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

// TEST_FEATURE: FftUtils_fft_solver

#include "cufft.h"

int main() {
  cufftHandle plan;
  float2* odata;
  float2* idata;
  cufftPlan1d(&plan, 10, CUFFT_C2C, 3);
  cufftExecC2C(plan, idata, odata, CUFFT_FORWARD);
  return 0;
}
