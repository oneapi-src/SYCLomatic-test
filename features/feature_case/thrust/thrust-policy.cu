// ====------ thrust-policy.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include "report.h"

struct IsEven {
  __host__ __device__ bool operator()(const int x) const {
    return x % 2 == 0;
  }
} isEven;

template<typename Iterator>
bool verify(Iterator begin, Iterator end, Iterator expected) {
  for (Iterator p = begin; p != end; p++, expected++) {
    if (*p != *expected)
      return false;
  }
  return true;
}

template<typename Iterator>
void print(Iterator begin, Iterator end) {
  for (Iterator p = begin; p != end; p++)
    std::cout << " " << *p;
  std::cout << std::endl;
}

void check() {
  const int N = 4;
  int vecHIn[N] = {-1, 0, 1, 2};
  int vecHOut[N];

  thrust::device_vector<int> dVecIn(vecHIn, vecHIn+N);
  thrust::device_vector<int> dVecOut(N);
 
  // No policy specified.  Assume host memory.
  int expected[2] = {0, 2};
  memset(vecHOut, 0, N*sizeof(int));
  thrust::copy_if(vecHIn, vecHIn+N, vecHOut, isEven);
  //print(vecHOut, vecHOut + 2);
  Report::check("No policy with raw pointers.  Assume host memory",
                verify(vecHOut, vecHOut + 2, expected), true);

  int *vecDIn;
  int *vecDOut;
  cudaMalloc(&vecDIn, sizeof(int)*N);
  cudaMalloc(&vecDOut, sizeof(int)*N);
  cudaMemcpy(vecDIn, vecHIn, sizeof(int)*N, cudaMemcpyHostToDevice);
 
  // Policy (thrust::device) specified. Derive memory source from policy. This works with nvcc!
  memset(vecHOut, 0, N*sizeof(int));
  thrust::copy_if(thrust::device, vecDIn, vecDIn+N, vecDOut, isEven);
  cudaMemcpy(vecHOut, vecDOut, sizeof(int)*N, cudaMemcpyDeviceToHost);
  //print(vecHOut, vecHOut + 2);
  Report::check("Device policy with raw pointers.  Derive memory from policy (device)",
                verify(vecHOut, vecHOut + 2, expected), true);

  // Policy (thrust::host) specified. Derive memory source from policy. This works with nvcc!
  memset(vecHOut, 0, N*sizeof(int));
  thrust::copy_if(thrust::host, vecHIn, vecHIn+N, vecHOut, isEven);
  //print(vecHOut, vecHOut + 2);
  Report::check("Host policy with raw pointers.  Derive memory from policy (host)",
                verify(vecHOut, vecHOut + 2, expected), true);

  // No policy specified. thrust::device_vector used for input/output use device policy
  thrust::device_vector<int> vecDvIn(vecHIn, vecHIn+N);
  thrust::device_vector<int> vecDvOut(N, 0);
  thrust::device_vector<int> expectedD(expected, expected + 2);
  thrust::copy_if(vecDvIn.begin(), vecDvIn.end(), vecDvOut.begin(), isEven);
  //print(vecDvOut.begin(), vecDvOut.begin() + 2);
  Report::check("No policy with device_vector.  Derive memory from input iterator",
                verify(vecDvOut.begin(), vecDvOut.begin() + 2, expectedD.begin()), true);
}

int main() {
  Report::start("Check correct use of raw pointers for host/device policies");
  check();
  return Report::finish();
}