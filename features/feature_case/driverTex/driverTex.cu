// ====------ driverTex.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
int main(){
  CUtexref tex;
  CUdeviceptr dptr;
  size_t s, b;
  unsigned int uflag;
  CUaddress_mode addr_mode;
  CUfilter_mode filter_mode;
  cuTexRefGetFlags(&uflag, tex);
  cuTexRefGetAddressMode(&addr_mode, tex, 0);
  cuTexRefGetFilterMode(&filter_mode, tex);
  cuTexRefSetAddress(&s, tex, dptr, b);
  CUDA_ARRAY_DESCRIPTOR desc;
  cuTexRefSetAddress2D(tex, &desc, dptr, b);


  CUarray_st **arr_ptr = new CUarray;
  CUDA_ARRAY3D_DESCRIPTOR p3DDesc;

  cuArray3DCreate(arr_ptr, &p3DDesc);
  cuArray3DGetDescriptor(&p3DDesc, *arr_ptr);

  CUDA_ARRAY_DESCRIPTOR halfDesc;
  cuArrayGetDescriptor(&halfDesc, *arr_ptr);

  cuTexRefCreate(&tex);
  cuTexRefDestroy(tex);
  CUarray arr;
  CUsurfref ref;
  cuSurfRefGetArray(&arr, ref);
  cuSurfRefSetArray(ref, arr, 0);
  return 0;
}

