// ====------ DplExtrasIterators_api_test3.cu ------------ *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// TEST_FEATURE: DplExtrasIterators_constant_iterator

#include <thrust/iterator/constant_iterator.h>

void foo(thrust::constant_iterator<int> i) {
}

int main() {
  return 0;
}
