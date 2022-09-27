// ====------ cub_counting_iterator.cu-------------------- *- CUDA -* ------===//
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

bool test_counting_iterator() {
  cub::CountingInputIterator<int> iter(0);
  for (int i = 0; i < DATA_NUM; ++i)
    if (iter[i] != i)
      return false;
  return true;
}

int main() {
  if (test_counting_iterator()) {
    std::cout << "cub::CountingInputIterator Pass\n";
    return 0;
  }
  return 1;
}
