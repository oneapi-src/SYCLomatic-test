// ===------ cusolver_range.cu ------------------------------*- CUDA -*-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include "cusolverDn.h"

void test0() {
  cusolverEigRange_t range = CUSOLVER_EIG_RANGE_ALL;
  range = CUSOLVER_EIG_RANGE_V;
  range = CUSOLVER_EIG_RANGE_I;
}
