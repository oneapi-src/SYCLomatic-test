// ====------ test.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <iostream>
#include "test.h"

void ewSource(double *s)
{
  double t;
  t = sindeg(s[1]);
  std::cout << t;
}

int main(){
  double a[5] = {1.0, 2.9, 30.0, 5.1};
  ewSource(a);
  return 0;
}
