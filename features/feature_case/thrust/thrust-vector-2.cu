// ====------ thrust-vector-2.cu---------- *- CUDA -* ----===////
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
#include "report.h"

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
  uint32_t errorCount = 0;
  for (auto i = 0; i < v.size(); ++i) {
    if (v[i] != i + 1) {
      Report::fail(std::string(msg) + " - element " + std::to_string(i));
      errorCount++;
    }
  }
  if (errorCount == 0)
    Report::pass(msg);
}

int main() {
  Report::start("thrust:: vector operations");
  
  thrust::host_vector<int> hv(4);
  thrust::device_vector<int> dv(4);

  initVector(hv);
  checkVector(hv, "host_vector initialization");

  initVector(dv);
  checkVector(dv, "device_vector initialization");

  thrust::device_vector<int> dv2 = hv;
  checkVector(dv2, "host_vector -> device_vector copy");

  thrust::host_vector<int> hv2 = dv;
  checkVector(hv2, "device_vector -> host_vector copy");

  thrust::host_vector<int> hv3 = hv;
  checkVector(hv3, "host_vector -> host_vector copy");

  thrust::host_vector<int> dv3 = dv;
  checkVector(hv3, "device_vector -> device_vector copy");

  thrust::device_vector<int> dv4(2, 99); // initializes 2-element vector with 99
  dv4[0] = hv[0];
  dv4[1] = hv[1];
  checkVector(dv4, "host_vector element -> device_vector element copy");

  thrust::host_vector<int> hv4(2, 99); // initializes 2-element vector with 99
  hv4[0] = dv[0];
  hv4[1] = dv[1];
  checkVector(hv4, "device_vector element -> host_vector element copy");

  hv2.resize(5);
  hv2[4] = 5;
  Report::check("host_vector size", hv2.size(), 5);
  checkVector(hv2, "host_vector dynamic expansion");

  dv2.resize(5);
  dv2[4] = 5;
  Report::check("device_vector size", dv2.size(), 5);
  checkVector(dv2, "device_vector dynamic expansion");

  thrust::device_vector<int> dv5(2, -1);
  dv5[1] = 10;
  thrust::device_reference<int> r0 = dv5[0];
  thrust::device_reference<int> r1 = dv5[1];
  r0 = r1;
  Report::check("device_reference = ", dv5[0], 10);

  return Report::finish();
}
