// ====------ cufft-different-locations.cu---------- *- CUDA -* ----===////
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
double* odata;
double2* idata;

#define HANDLE_CUFFT_ERROR( err ) (CufftHandleError( err, __FILE__, __LINE__ ))
static void CufftHandleError( cufftResult err, const char *file, int line ) {
  if (err != CUFFT_SUCCESS) {
    fprintf(stderr, "Cufft error in file '%s' in line %i : %s.\n",
            __FILE__, __LINE__, "error" );
  }
}

int main() {
  cufftHandle plan1;
  cufftResult res1 = cufftMakePlanMany(plan1, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size);
  cufftResult res2 = cufftExecZ2D(plan1, idata, odata);

  cufftHandle plan2;
  res1 = cufftMakePlanMany(plan2, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size);
  res2 = cufftExecZ2D(plan2, idata, odata);

  cufftHandle plan3;
  HANDLE_CUFFT_ERROR(cufftMakePlanMany(plan3, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size));
  HANDLE_CUFFT_ERROR(cufftExecZ2D(plan3, idata, odata));

  cufftHandle plan4;
  cufftHandle plan5;
  if(cufftMakePlanMany(plan4, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size)) {
  } else if (cufftMakePlanMany(plan5, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size)) {
  }
  if (cufftExecZ2D(plan4, idata, odata)) {
  } else if(cufftExecZ2D(plan5, idata, odata)) {
  }

  cufftHandle plan6;
  if(int res = 0) {
  }
  if(cufftResult res = cufftMakePlanMany(plan6, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size)) {
  }
  if(cufftResult res = cufftExecZ2D(plan6, idata, odata)) {
  }

  cufftHandle plan7;
  for (0;;) {
  }
  for (cufftMakePlanMany(plan7, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size);;) {
  }
  for (cufftExecZ2D(plan7, idata, odata);;) {
  }

  cufftHandle plan8;
  for (;cufftMakePlanMany(plan8, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size);) {
  }
  for (;cufftExecZ2D(plan8, idata, odata);) {
  }

  cufftHandle plan9;
  while (cufftMakePlanMany(plan9, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size) != 0) {
  }
  while (cufftExecZ2D(plan9, idata, odata) != 0) {
  }

  cufftHandle plan10;
  do {
  } while (cufftMakePlanMany(plan10, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size));
  do {
  } while (cufftExecZ2D(plan10, idata, odata));

  cufftHandle plan11;
  switch (int stat = cufftMakePlanMany(plan11, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size)){
  }
  switch (int stat = cufftExecZ2D(plan11, idata, odata)){
  }
  return 0;
}

cufftResult foo1(cufftHandle plan) {
  return cufftMakePlanMany(plan, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size);
}

cufftResult foo2(cufftHandle plan) {
  return cufftExecZ2D(plan, idata, odata);
}

cufftResult foo3(cufftHandle plan) {
  cufftMakePlanMany(plan, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size);
}

cufftResult foo4(cufftHandle plan) {
  cufftExecZ2D(plan, idata, odata);
}

