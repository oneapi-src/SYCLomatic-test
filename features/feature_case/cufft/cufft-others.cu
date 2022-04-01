// ====------ cufft-others.cu---------- *- CUDA -* ----===////
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


int main() {
  cufftHandle plan;
  float2* iodata;

  cufftPlan1d(&plan, 10 + 2, CUFFT_R2C, 3);

  cufftExecR2C(plan, (float*)iodata, iodata);

  return 0;
}

int foo2() {
  cufftHandle plan_mmany64_Z2Z;
  size_t* work_size_mmany64_Z2Z;
  long long int odist_mmany64_Z2Z;
  long long int ostride_mmany64_Z2Z;
  long long int * onembed_mmany64_Z2Z;
  long long int idist_mmany64_Z2Z;
  long long int istride_mmany64_Z2Z;
  long long int* inembed_mmany64_Z2Z;
  long long int * n_mmany64_Z2Z;
  double2* odata_mmany64_Z2Z;
  double2* idata_mmany64_Z2Z;

  cufftMakePlanMany64(plan_mmany64_Z2Z, 3, n_mmany64_Z2Z, inembed_mmany64_Z2Z, istride_mmany64_Z2Z, idist_mmany64_Z2Z, onembed_mmany64_Z2Z, ostride_mmany64_Z2Z, odist_mmany64_Z2Z, CUFFT_Z2Z, 12, work_size_mmany64_Z2Z);

  cufftExecZ2Z(plan_mmany64_Z2Z, idata_mmany64_Z2Z, odata_mmany64_Z2Z, CUFFT_FORWARD);

  cufftExecZ2Z(plan_mmany64_Z2Z, idata_mmany64_Z2Z, odata_mmany64_Z2Z, CUFFT_INVERSE);

  return 0;
}

int foo3(cudaStream_t stream) {
  cufftHandle plan;
  float2* iodata;

  cufftSetStream(plan, stream);
  cufftPlan1d(&plan, 10 + 2, CUFFT_R2C, 3);

  cufftExecR2C(plan, (float*)iodata, iodata);

  return 0;
}
