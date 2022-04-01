// ====------ thrust-transform-if.cu---------- *- CUDA -* ----===////
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
#include "report.h"

struct is_odd {
  __host__ __device__ bool operator()(const int x) const {
    return x % 2 != 0;
  }
};

struct identity {
  __host__ __device__ bool operator()(const int x) const {
    return x;
  }
};

thrust::negate<int> neg;
thrust::plus<int> plus;

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
  const int dataLen = 10;
  int inDataH[dataLen]  = {-5, 0, 2, -3, 2, 4, 0, -1, 2, 8};
  int stencilH[dataLen] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};

  thrust::device_vector<int> inDataD(inDataH, inDataH + dataLen);
  thrust::device_vector<int> stencilD(stencilH, stencilH + dataLen);

  // Policy
  int outData1H[dataLen] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  int expectedData1[dataLen] = {5, 0, 0, 3, 0, 0, 0, 1, 0, 0};
  thrust::transform_if(thrust::host, inDataH, inDataH + dataLen, outData1H, neg, is_odd());
  //print(outData1H, outData1H + dataLen);
  Report::check("transform_if(policy, input, inputEnd, output, unaryOp, predicate)",
                verify(outData1H, outData1H + dataLen, expectedData1), true);

  // Policy and stencil
  int outData2H[dataLen] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  int expectedData2[dataLen] = { 5, 0, -2,  0, -2, 0, 0,  0, -2, 0};
  thrust::transform_if(thrust::host, inDataH, inDataH + dataLen, stencilH, outData2H, neg, identity());
  //print(outData2H, outData2H + dataLen);
  Report::check("transform_if(policy, input, inputEnd, stencil, output, unaryOp, predicate)",
                verify(outData2H, outData2H + dataLen, expectedData2), true);

  // Policy, second input, stencil and binary op
  int outData3H[dataLen] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  int expectedData3[dataLen] = {-10, 0, 4,  0, 4, 0,  0, 0, 4, 0};
  thrust::transform_if(thrust::host, inDataH, inDataH + dataLen, inDataH, stencilH, outData3H, plus, identity());
  //print(outData3H, outData3H + dataLen);
  Report::check("transform_if(policy, input, inputEnd, input2, stencil, output, binaryOp, predicate)",
                verify(outData3H, outData3H + dataLen, expectedData3), true);

  // No policy
  thrust::device_vector<int> outData4D(dataLen, 0);
  thrust::device_vector<int> expectedData4(expectedData1, expectedData1 + dataLen);
  thrust::transform_if(inDataD.begin(), inDataD.end(), outData4D.begin(), neg, is_odd());
  //print(outData4D.begin(), outData4D.end());
  Report::check("transform_if(input, inputEnd, output, unaryOp, predicate)",
                verify(outData4D.begin(), outData4D.end(), expectedData4.begin()), true);

  // No policy and stencil
  thrust::device_vector<int> outData5D(dataLen, 0);
  thrust::device_vector<int> expectedData5(expectedData2, expectedData2 + dataLen);
  thrust::transform_if(inDataD.begin(), inDataD.end(), stencilD.begin(), outData5D.begin(), neg, identity());
  //print(outData5D.begin(), outData5D.end());
  Report::check("transform_if(input, inputEnd, stencil, output, unaryOp, predicate)",
                verify(outData5D.begin(), outData5D.end(), expectedData5.begin()), true);

  // No policy, second input, stencil and binary op
  thrust::device_vector<int> outData6D(dataLen, 0);
  thrust::device_vector<int> expectedData6(expectedData3, expectedData3 + dataLen);
  thrust::transform_if(inDataD.begin(), inDataD.end(), inDataD.begin(), stencilD.begin(), outData6D.begin(), plus, identity());
  //print(outData6D.begin(), outData6D.end());
  Report::check("transform_if(input, inputEnd, input2, stencil, output, binaryOp, predicate)",
                verify(outData6D.begin(), outData6D.end(), expectedData6.begin()), true);
}

int main() {
  Report::start("thrust::transform_if");
  check();
  return Report::finish();
}