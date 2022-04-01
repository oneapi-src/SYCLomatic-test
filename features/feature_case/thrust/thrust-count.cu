// ====------ thrust-count.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <thrust/device_vector.h>
#include <thrust/count.h>
#include <vector>
#include "report.h"

void checkCount() {
  std::vector<int> input{1, 2, 3, 2, 3, 3};
  thrust::device_vector<int> vecD(input.begin(), input.end());
  int c;
  c = thrust::count(vecD.begin(), vecD.end(), 0);
  Report::check("thrust::count - No policy - 0", c, 0);
  c = thrust::count(vecD.begin(), vecD.end(), 1);
  Report::check("thrust::count - No policy - 1", c, 1);
  c = thrust::count(vecD.begin(), vecD.end(), 2);
  Report::check("thrust::count - No policy - 2", c, 2);
  c = thrust::count(vecD.begin(), vecD.end(), 3);
  Report::check("thrust::count - No policy - 3", c, 3);
}

void checkCountIf() {

}

int main() {
  Report::start("thrust::count/count_if");
  checkCount();
  checkCountIf();
  return Report::finish();
}