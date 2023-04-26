// ====-------------- double.cu---------- *- CUDA -* ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <cuda.h>
#include <iostream>
#include <vector>

using namespace std;

__global__ void _rcbrt(double *const DeviceResult, double Input1) {
  *DeviceResult = rcbrt(Input1);
}

void testErfcinv(double *const DeviceResult, double Input) {
  _rcbrt<<<1, 1>>>(DeviceResult, Input);
  cudaDeviceSynchronize();
}

int main() {
  double *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  try {
    testErfcinv(DeviceResult, 0);
  } catch (exception const &e) {
    cout << "Catch exception: " << e.what() << endl;
    char name[100];
    CUdevice device;
    cuDeviceGetName(name, 90, device);
    string ExpectException =
        "'double' is not supported in '" + string(name) + "' device";
    cout << "Expect exception: " << ExpectException << endl;
    if (e.what() == ExpectException) {
      return 0;
    }
    return 1;
  }
  return 0;
}
