//====---------- math-emu-bf162.cu- --------- *- CUDA -* ------------------===//
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

typedef pair<__nv_bfloat162, __nv_bfloat162> bf162_pair;
typedef pair<__nv_bfloat162, int> bf162i_pair;

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

void checkResult(const string &FuncName, const vector<float2> &Inputs,
                 const float2 &Expect, const float2 &Result,
                 const int precision) {
  cout << FuncName << "({" << Inputs[0].x << ", " << Inputs[0].y << "}";
  for (size_t i = 1; i < Inputs.size(); ++i) {
    cout << ", {" << Inputs[i].x << ", " << Inputs[i].y << "}";
  }
  cout << ") = " << fixed << setprecision(precision) << "{" << Result.x << ", "
       << Result.y << "} (expect {" << Expect.x - pow(10, -precision) << " ~ "
       << Expect.x + pow(10, -precision) << ", "
       << Expect.y - pow(10, -precision) << " ~ "
       << Expect.y + pow(10, -precision) << ")";
  cout.unsetf(ios::fixed);
  check(abs(Result.x - Expect.x) < pow(10, -precision) &&
        abs(Result.y - Expect.y) < pow(10, -precision));
}

void checkResult(const string &FuncName, const vector<__nv_bfloat162> &Inputs,
                 const __nv_bfloat162 &Expect, const float2 &Result,
                 const int precision) {
  vector<float2> FInputs;
  for (const auto &it : Inputs) {
    FInputs.push_back({__bfloat162float(it.x), __bfloat162float(it.y)});
  }
  float2 FExpect{__bfloat162float(Expect.x), __bfloat162float(Expect.y)};
  checkResult(FuncName, FInputs, FExpect, Result, precision);
}

// Bfloat162 Arithmetic Functions

__global__ void h2div(float *const Result, __nv_bfloat162 Input1,
                      __nv_bfloat162 Input2) {
  auto ret = __h2div(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testH2divCases(const vector<pair<bf162_pair, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    h2div<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__h2div", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

int main() {
  testH2divCases({
      {{{-0.3, -5}, {0.4, 6}}, {{-0.75, -0.83203125}, 16}},
      {{{0.3, 5}, {-4, 0.6}}, {{-0.0751953125, 8.3125}, 15}},
      {{{0.3, 0.5}, {0.4, 0.2}}, {{0.75, 2.5}, 15}},
      {{{0.3, 0.5}, {0.4, 0.6}}, {{0.75, 0.83203125}, 16}},
      {{{3, 5}, {4, 6}}, {{0.75, 0.83203125}, 16}},
  });
  cout << "passed " << passed << "/" << passed + failed << " cases!" << endl;
  if (failed) {
    cout << "failed!" << endl;
  }
  return failed;
}
