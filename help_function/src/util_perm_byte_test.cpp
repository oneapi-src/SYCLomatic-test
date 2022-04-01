// ====------ util_perm_byte_test.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>

void byte_perm_ref(unsigned int *d_data) {

  unsigned int lo;
  unsigned int hi;

  lo = 0x33221100;
  hi = 0x77665544;

  lo = 0x33221100;
  hi = 0x77665544;

  for (int i = 0; i < 17; i++)
    d_data[i] = dpct::byte_level_permute(lo, hi, 0x1111 * i);

  d_data[17] = dpct::byte_level_permute(lo, 0, 0x0123);
  d_data[18] = dpct::byte_level_permute(lo, hi, 0x7531);
  d_data[19] = dpct::byte_level_permute(lo, hi, 0x6420);
}

void byte_level_permute_test() {
  const int N = 20;
  unsigned int refer[N] = {0x0,        0x11111111, 0x22222222, 0x33333333,
                           0x44444444, 0x55555555, 0x66666666, 0x77777777,
                           0x0,        0x11111111, 0x22222222, 0x33333333,
                           0x44444444, 0x55555555, 0x66666666, 0x77777777,
                           0x11111100, 0x112233,   0x77553311, 0x66442200};
  unsigned int data[N];

  byte_perm_ref(data);

  bool pass = true;
  for (int i = 0; i < N; i++) {
    if (refer[i] != data[i]) {
      printf("Index %d: 0x%x vs 0x%x\n", i, data[i], refer[i]);
      exit(-1);
    }
  }
  printf("byte_level_permute_test passed\n");
}

int main(void) {

  byte_level_permute_test();
  return 0;
}
