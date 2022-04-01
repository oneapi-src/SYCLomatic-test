// ====------ test.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

struct is_odd {
  __host__ __device__ bool operator()(const int x) const {
    return x % 2;
  }
};

struct identity {
  __host__ __device__ bool operator()(const int x) const {
    return x;
  }
};

thrust::negate<int> neg;
thrust::plus<int> plus;

int main() {
  const int dataLen = 10;
  int inDataH[dataLen]  = {-5, 0, 2, -3, 2, 4, 0, -1, 2, 8};
  int outDataH[dataLen];
  int stencilH[dataLen] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};

  thrust::device_vector<int> inDataD(dataLen);
  thrust::device_vector<int> outDataD(dataLen);
  thrust::device_vector<int> stencilD(dataLen);

  // Policy
  thrust::transform_if(thrust::host, inDataH, inDataH + dataLen, outDataH, neg, is_odd());

  // Policy and stencil
  thrust::transform_if(thrust::host, inDataH, inDataH + dataLen, stencilH, outDataH, neg, identity());

  // Policy, second input, stencil and binary op
  thrust::transform_if(thrust::host, inDataH, inDataH + dataLen, inDataH, stencilH, outDataH, plus, identity());

  // No policy
  thrust::transform_if(inDataD.begin(), inDataD.end(), outDataD.begin(), neg, is_odd());

  // No policy and stencil
  thrust::transform_if(inDataD.begin(), inDataD.end(), stencilD.begin(), outDataD.begin(), neg, identity());

  // No policy, second input, stencil and binary op
  thrust::transform_if(inDataD.begin(), inDataD.end(), inDataD.begin(), stencilD.begin(), outDataD.begin(), plus, identity());
}
