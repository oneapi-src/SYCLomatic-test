// ====------ cub_costant_iterator.cu-------------------- *- CUDA -* ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <cub/cub.cuh>
#include <stdio.h>

#define DATA_NUM 100

bool test_constant_iterator() {
  cub::ConstantInputIterator<int> iter(1024);
  for (int i = 0; i < DATA_NUM; ++i)
    if (iter[i] != 1024)
      return false;
  return true;
}

int main() {
  if (test_constant_iterator()) {
    std::cout << "cub::ConstantInputIterator Pass\n";
    return 0;
  }
  return 1;
}
