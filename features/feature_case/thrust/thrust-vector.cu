// ====------ thrust-vector.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>

static uint32_t errorCount = 0;

void logError(const char *msg) {
  std::cout << "ERROR: " << msg << "\n";
  errorCount++;
}

template<typename VT>
void printVector(VT &v, const char *prefix = "") {
  std::cout << prefix;
  for (auto e : v)
    std::cout << e << " ";
  std::cout << "\n";
}

template<typename VT>
void initVector(VT &v) {
  for (auto i = 0; i < v.size(); ++i) {
    v[i] = i + 1;
  }
}

template<typename VT>
void checkVector(VT &v, const char *msg) {
  std::cout << msg << "...\n";
  for (auto i = 0; i < v.size(); ++i) {
    if (v[i] != i + 1) {
      logError(msg);
    }
  }  
}

int main() {

  thrust::host_vector<int> hv(4);
  thrust::device_vector<int> dv(4);

  initVector(hv);
  checkVector(hv, "Check host_vector initialization");

  initVector(dv);
  checkVector(dv, "Check device_vector initialization");

  thrust::device_vector<int> dv2 = hv;
  checkVector(dv2, "Check host_vector -> device_vector copy");

  thrust::host_vector<int> hv2 = dv;
  checkVector(hv2, "Check device_vector -> host_vector copy");

  thrust::host_vector<int> hv3 = hv;
  checkVector(hv3, "Check host_vector -> host_vector copy");

  thrust::host_vector<int> dv3 = dv;
  checkVector(hv3, "Check device_vector -> device_vector copy");

  thrust::device_vector<int> dv4(2, 99); // initializes 2-element vector with 99
  dv4[0] = hv[0];
  dv4[1] = hv[1];
  checkVector(dv4, "Check host_vector element -> device_vector element copy");

  thrust::host_vector<int> hv4(2, 99); // initializes 2-element vector with 99
  hv4[0] = dv[0];
  hv4[1] = dv[1];
  checkVector(hv4, "Check device_vector element -> host_vector element copy");

  hv2.resize(5);
  hv2[4] = 5;
  if (hv2.size() != 5)
    logError("Unexpected host_vector size");
  checkVector(hv2, "Check host_vector dynamic expansion");

  dv2.resize(5);
  dv2[4] = 5;
  if (dv2.size() != 5)
    logError("Unexpected device_vector size");
  checkVector(dv2, "Check device_vector dynamic expansion");

  if (errorCount == 0)
  {
    std::cout << "PASSED\n";
    exit(0);
  }
  else {
    std::cout << "FAILED: " << errorCount << " failures detected\n";
    exit(-1);
  }
}
