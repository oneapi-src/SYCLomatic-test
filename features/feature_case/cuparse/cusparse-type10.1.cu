// ====------ cusparse-type10.1.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <cstdio>
#include <cusparse_v2.h>
#include <cuda_runtime.h>

// CUSPARSE_STATUS_NOT_SUPPORTED is available since v10.2.
int main(){
  cusparseStatus_t a6;
  a6 = CUSPARSE_STATUS_NOT_SUPPORTED;
}
