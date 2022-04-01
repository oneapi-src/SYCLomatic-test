// ====------ cufft-func-ptr.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <cufft.h>

static cufftResult (*pt2CufftExec)(cufftHandle, cufftDoubleComplex *,
                                    double *) = &cufftExecZ2D;

int main() {
  cufftHandle plan1;
  cufftPlan1d(&plan1, 10, CUFFT_Z2D, 1);

  double* odata;
  double2* idata;

  pt2CufftExec(plan1, idata, odata);

  return 0;
}

int foo1() {
  typedef cufftResult (*Func_t)(cufftHandle, cufftDoubleComplex *, double *);

  static Func_t FuncPtr  = &cufftExecZ2D;

  cufftHandle plan1;
  cufftPlan1d(&plan1, 10, CUFFT_Z2D, 1);

  double* odata;
  double2* idata;

  FuncPtr(plan1, idata, odata);

  return 0;
}

int foo2() {
  using Func_t = cufftResult (*)(cufftHandle, cufftDoubleComplex *, double *);

  Func_t FuncPtr2  = &cufftExecZ2D;

  cufftHandle plan1;
  cufftPlan1d(&plan1, 10, CUFFT_Z2D, 1);

  double* odata;
  double2* idata;

  FuncPtr2(plan1, idata, odata);

  return 0;
}

int foo3() {
  using Func_t = cufftResult (*)(cufftHandle, cufftDoubleComplex *, double *);

  Func_t FuncPtr3;
  FuncPtr3 = &cufftExecZ2D;

  cufftHandle plan1;
  cufftPlan1d(&plan1, 10, CUFFT_Z2D, 1);

  double* odata;
  double2* idata;

  FuncPtr3(plan1, idata, odata);

  return 0;
}

int foo4() {
  cufftResult (*FuncPtr4)(cufftHandle, cufftDoubleComplex *, double *);

  FuncPtr4 = &cufftExecZ2D;

  cufftHandle plan1;
  cufftPlan1d(&plan1, 10, CUFFT_Z2D, 1);

  double* odata;
  double2* idata;

  FuncPtr4(plan1, idata, odata);

  return 0;
}
