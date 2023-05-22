// ====--------------- half.cu---------- *- CUDA -* -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include "cuda_fp16.h"
#include <cuda.h>
#include <iostream>

using namespace std;

__device__ void device_fp() {
  __half a = 0;
  __hneg(a);
}
__global__ void test_fp() { device_fp(); }

int main() {
  try {
    test_fp<<<1, 1>>>();
  } catch (exception const &e) {
    cout << "Catch exception: " << e.what() << endl;
    char name[100];
    CUdevice device = 0;
    cuDeviceGetName(name, 90, device);
    string ExpectException =
        "'half' is not supported in '" + string(name) + "' device";
    cout << "Expect exception: " << ExpectException << endl;
    if (e.what() == ExpectException) {
      return 0;
    }
    return 1;
  }
  return 0;
}
