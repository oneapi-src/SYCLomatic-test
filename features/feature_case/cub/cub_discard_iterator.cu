// ====------ cub_discard_iterator.cu -------------------- *- CUDA -* ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <cub/cub.cuh>

bool test() {
  cub::DiscardOutputIterator<> Iter, Begin = Iter;
  for (int i = 0; i < 10; ++i, Iter++) {
    *Iter = i;
  }

  return Iter - Begin == 10;
}

int main() {
  if (!test())
    return 1;
  return 0;
}
