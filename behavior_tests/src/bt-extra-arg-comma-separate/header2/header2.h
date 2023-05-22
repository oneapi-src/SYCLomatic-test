// ====------ header2.h---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include "cuda_runtime.h"

// CHECK: inline void helloFromGPU2(const cl::sycl::nd_item<3> &item_ct1) {
// CHECK-NEXT:     int a = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2) + item_ct1.get_group(2) +
// CHECK-NEXT:     item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
// CHECK-NEXT: }
__global__ void helloFromGPU2() {
  int a = blockIdx.x * blockDim.x + threadIdx.x + blockIdx.x +
          blockDim.x + threadIdx.x;
}