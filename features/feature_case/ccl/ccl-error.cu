// ====------ ccl-error.cu---------------------------------- *- CUDA -* ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include "nccl.h"
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,13,0)
#endif
int main(){
  int version;
  ncclResult_t res;
  int major = NCCL_MAJOR;
  int minor = NCCL_MINOR;
  res = ncclGetVersion(&version);
  ncclGetErrorString(res);
  ncclGetLastError(NULL);
  if (res == ncclSuccess) {
    return 0;
  }
}