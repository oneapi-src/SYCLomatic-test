// ====------ pointer_attributes_usmnone.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <cuda.h>

int main() {
  void *base_ptr;
  void *ptr;
  //CHECK:  if (DPCT_CHECK_ERROR(base_ptr = dpct::get_base_addr((dpct::device_ptr)ptr)) != 0);
  if (cuPointerGetAttribute(base_ptr, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, (CUdeviceptr)ptr) != CUDA_SUCCESS);
}