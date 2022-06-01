// ====------ thrust-copy.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <cuda_runtime.h>
#include "report.h"

template<typename VT>
void initVector(VT &vec) {
  for (typename VT::size_type i = 0; i < vec.size(); i++) {
    vec[i] = i;
  }
}

template<typename VT>
void checkVector(const char *msg, const VT &vec) {
  uint32_t failCount = 0;
  for (typename VT::size_type i = 0; i < vec.size(); i++) {
    if (vec[i] != i)
      failCount++;
  }
  Report::check(msg, failCount, 0);
}

void checkCopy() {
  thrust::device_vector<int> dv1(100);
  thrust::device_vector<int> dv2(100);
  initVector(dv1);
  checkVector("device_vector initialization", dv1);
  thrust::copy(dv1.begin(), dv1.end(), dv2.begin());
  checkVector("thrust::copy - No policy", dv2);
  thrust::host_vector<int> hv1(100);
  thrust::host_vector<int> hv2(100);
  initVector(hv1);
  checkVector("host_vector initialization", hv1);
  thrust::copy(thrust::seq, hv1.begin(), hv1.end(), hv2.begin());
  checkVector("thrust::copy - With policy", hv2);
}

void checkCopyIf() {
  auto isEven = [] __host__ __device__ (int v) {return (v % 2) == 0;};
  auto isOdd = [](int v) {return (v % 2) != 0;};
  const int N = 4;
  int vecHIn[N] = {-1, 0, 1, 2};
  int vecHOut[N];
  thrust::copy_if(vecHIn, vecHIn+N, vecHOut, isEven);
  Report::check("thrust::copy_if(host memory, raw pointer) - No policy - 0", vecHOut[0], 0);
  Report::check("thrust::copy_if(host memory, raw pointer) - No policy - 1", vecHOut[1], 2);
  thrust::copy_if(thrust::seq, vecHIn, vecHIn+N, vecHOut, isOdd);
  Report::check("thrust::copy_if(host memory, raw pointer) - With policy - 0", vecHOut[0], -1);
  Report::check("thrust::copy_if(host memory, raw pointer) - With policy - 1", vecHOut[1], 1);

/*
** Below thrust::copy_if will fail, because host memory is assumed
** when raw pointers are used and no policy is specified.  Migrated code
** does it the other way around, so the migrated version of code below
** passes.
  int *vecDIn;
  int *vecDOut;
  cudaMalloc(&vecDIn, sizeof(int)*N);
  cudaMalloc(&vecDOut, sizeof(int)*N);
  cudaMemcpy(vecDIn, vecHIn, sizeof(int)*N, cudaMemcpyHostToDevice);
  thrust::copy_if(vecDIn, vecDIn+N, vecDOut, isEven);
  cudaMemcpy(vecHOut, vecDOut, sizeof(int)*N, cudaMemcpyDeviceToHost);
  Report::check("thrust::copy_if(device memory, raw pointer)- No policy - 0", vecHOut[0], 0);
  Report::check("thrust::copy_if(device memory, raw pointer) - No policy - 1", vecHOut[1], 2);
*/

  thrust::device_vector<int> vecIn(4);
  thrust::device_vector<int> vecOut(4);
  initVector(vecIn);
  thrust::copy_if(vecIn.begin(), vecIn.end(), vecOut.begin(), isEven);
  Report::check("thrust::copy_if(device memory) - No policy - 0", vecOut[0], 0);
  Report::check("thrust::copy_if(device memory) - No policy - 1", vecOut[1], 2);
}

void checkCopyN() {
  thrust::device_vector<int> vecIn(4);
  thrust::device_vector<int> vecOut(4);
  initVector(vecIn);
  initVector(vecOut);
  thrust::copy_n(vecIn.begin(), 2, vecOut.begin()+2);
  Report::check("thrust::copy_n(device memory) - No policy - 2", vecOut[2], 0);
  Report::check("thrust::copy_n(device memory) - No policy - 3", vecOut[3], 1);

  const int N = 4;
  int vecInH[N] = {0, 1, 2, 3};
  int vecOutH[N] = {4, 5, 6, 7};
  thrust::copy_n(thrust::seq, vecInH, 2, vecOutH+2);
  Report::check("thrust::copy_n(host memory, raw pointer) - With policy - 2", vecOutH[2], 0);
  Report::check("thrust::copy_n(host memory, raw pointer) - With policy - 3", vecOut[3], 1);
}

int main() {
  Report::start("thrust::copy/copy_if/copy_n");
  checkCopy();
  checkCopyIf();
  checkCopyN();
  return Report::finish();
}
