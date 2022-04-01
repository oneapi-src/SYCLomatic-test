// ====------ cufft-deduce.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <cstdio>
#include <cufft.h>
#include <cuda_runtime.h>


size_t* work_size;
int odist;
int ostride;
int * onembed;
int idist;
int istride;
int* inembed;
int * n;
constexpr int rank = 3;

void foo1(cufftHandle plan) {
  double* odata;
  double2* idata;
  cufftExecZ2D(plan, idata, odata);
}

void foo2(cufftHandle plan) {
  double* odata;
  double2* idata;
  cufftExecZ2D(plan, idata, odata);
}

int main() {
  constexpr cufftType_t type = CUFFT_Z2D;

  cufftHandle plan1;
  cufftMakePlanMany(plan1, rank, n, inembed, istride, idist, onembed, ostride, odist, type, 12, work_size);

  cufftHandle plan2;
  cufftMakePlanMany(plan2, rank, n, inembed, istride, idist, onembed, ostride, odist, type, 12, work_size);

  foo1(plan1);
  foo2(plan2);

  return 0;
}

