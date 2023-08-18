//====-------- math-experimental-bf16.cu --------- *- CUDA -* -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
//===----------------------------------------------------------------------===//

#include <iomanip>
#include <iostream>
#include <vector>

#include "cuda_bf16.h"

using namespace std;

typedef vector<__nv_bfloat16> bf16_vector;
typedef pair<__nv_bfloat16, int> bf16i_pair;

int passed = 0;
int failed = 0;

void check(bool IsPassed) {
  if (IsPassed) {
    cout << " ---- passed" << endl;
    passed++;
  } else {
    cout << " ---- failed" << endl;
    failed++;
  }
}

void checkResult(const string &FuncName, const vector<float> &Inputs,
                 const float &Expect, const float &Result,
                 const int precision) {
  cout << FuncName << "(" << Inputs[0];
  for (size_t i = 1; i < Inputs.size(); ++i) {
    cout << ", " << Inputs[i];
  }
  cout << ") = " << fixed << setprecision(precision) << Result << " (expect "
       << Expect - pow(10, -precision) << " ~ " << Expect + pow(10, -precision)
       << ")";
  cout.unsetf(ios::fixed);
  check(abs(Result - Expect) < pow(10, -precision));
}

void checkResult(const string &FuncName, const vector<__nv_bfloat16> &Inputs,
                 const __nv_bfloat16 &Expect, const float &Result,
                 const int precision) {
  vector<float> FInputs;
  for (const auto &it : Inputs) {
    FInputs.push_back(__bfloat162float(it));
  }
  float FExpect{__bfloat162float(Expect)};
  checkResult(FuncName, FInputs, FExpect, Result, precision);
}

// Bfloat16 Arithmetic Functions

__global__ void habs(float *const Result, __nv_bfloat16 Input1) {
  *Result = __habs(Input1);
}

void testHabsCases(const vector<pair<__nv_bfloat16, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    habs<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__habs", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void hfma(float *const Result, __nv_bfloat16 Input1,
                     __nv_bfloat16 Input2, __nv_bfloat16 Input3) {
  *Result = __hfma(Input1, Input2, Input3);
}

void testHfmaCases(const vector<pair<bf16_vector, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hfma<<<1, 1>>>(Result, TestCase.first[0], TestCase.first[1],
                   TestCase.first[2]);
    cudaDeviceSynchronize();
    checkResult("__hfma", TestCase.first, TestCase.second.first, *Result,
                TestCase.second.second);
    if (TestCase.first.size() != 3) {
      failed++;
      cout << " ---- failed" << endl;
      return;
    }
  }
}

__global__ void hfma_relu(float *const Result, __nv_bfloat16 Input1,
                          __nv_bfloat16 Input2, __nv_bfloat16 Input3) {
  *Result = __hfma_relu(Input1, Input2, Input3);
}

void testHfma_reluCases(
    const vector<pair<bf16_vector, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hfma_relu<<<1, 1>>>(Result, TestCase.first[0], TestCase.first[1],
                        TestCase.first[2]);
    cudaDeviceSynchronize();
    checkResult("__hfma_relu", TestCase.first, TestCase.second.first, *Result,
                TestCase.second.second);
    if (TestCase.first.size() != 3) {
      failed++;
      cout << " ---- failed" << endl;
      return;
    }
  }
}

__global__ void hfma_sat(float *const Result, __nv_bfloat16 Input1,
                         __nv_bfloat16 Input2, __nv_bfloat16 Input3) {
  *Result = __hfma_sat(Input1, Input2, Input3);
}

void testHfma_satCases(const vector<pair<bf16_vector, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hfma_sat<<<1, 1>>>(Result, TestCase.first[0], TestCase.first[1],
                       TestCase.first[2]);
    cudaDeviceSynchronize();
    checkResult("__hfma_sat", TestCase.first, TestCase.second.first, *Result,
                TestCase.second.second);
    if (TestCase.first.size() != 3) {
      failed++;
      cout << " ---- failed" << endl;
      return;
    }
  }
}

int main() {
  testHabsCases({
      {{-0.3}, {0.30078125, 16}},
      {{0.3}, {0.30078125, 16}},
      {{0.5}, {0.5, 16}},
      {{0.4}, {0.400390625, 16}},
      {{6}, {6, 15}},
  });
  testHfmaCases({
      {{-0.3, -0.4, -0.2}, {-0.07958984375, 17}},
      {{0.3, -0.4, -0.1}, {-0.220703125, 16}},
      {{0.3, 0.4, 0.1}, {0.220703125, 16}},
      {{0.3, 0.4, 0}, {0.12060546875, 17}},
      {{3, 4, 5}, {17, 14}},
  });
  testHfma_reluCases({
      {{-0.3, -0.4, -0.2}, {0, 37}},
      {{0.3, -0.4, -0.1}, {0, 37}},
      {{0.3, 0.4, 0.1}, {0.220703125, 16}},
      {{0.3, 0.4, 0}, {0.12060546875, 17}},
      {{3, 4, 5}, {17, 14}},
  });
  testHfma_satCases({
      {{-0.3, -0.4, -0.2}, {0, 37}},
      {{0.3, -0.4, -0.1}, {0, 37}},
      {{0.3, 0.4, 0.1}, {0.220703125, 16}},
      {{0.3, 0.4, 0}, {0.12060546875, 17}},
      {{3, 4, 5}, {1, 15}},
  });
  cout << "passed " << passed << "/" << passed + failed << " cases!" << endl;
  if (failed) {
    cout << "failed!" << endl;
  }
  return failed;
}
