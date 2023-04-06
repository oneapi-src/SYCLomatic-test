// ====---------- math-bf16-conv.cu---------- *- CUDA -* ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <iomanip>
#include <iostream>
#include <vector>

#include "cuda_bf16.h"

using namespace std;

typedef pair<float2, int> f2i_pair;
typedef pair<float, int> fi_pair;
typedef pair<__nv_bfloat162, int> bf2i_pair;
typedef pair<__nv_bfloat16, int> bfi_pair;

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

void checkResult(const string &FuncName, const vector<__nv_bfloat162> &Inputs,
                 const float2 &Expect, const float2 &Result,
                 const int precision) {
  cout << FuncName << "({" << __bfloat162float(Inputs[0].x) << ", "
       << __bfloat162float(Inputs[0].y) << "}";
  for (size_t i = 1; i < Inputs.size(); ++i) {
    cout << ", {" << __bfloat162float(Inputs[i].x) << ", "
         << __bfloat162float(Inputs[i].y) << "}";
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

void checkResult(const string &FuncName, const vector<__nv_bfloat16> &Inputs,
                 const float &Expect, const float &Result,
                 const int precision) {
  cout << FuncName << "(" << __bfloat162float(Inputs[0]) << "";
  for (size_t i = 1; i < Inputs.size(); ++i) {
    cout << ", " << __bfloat162float(Inputs[i]);
  }
  cout << ") = " << fixed << setprecision(precision) << Result << " (expect "
       << Expect - pow(10, -precision) << " ~ " << Expect + pow(10, -precision)
       << ")";
  cout.unsetf(ios::fixed);
  check(abs(Result - Expect) < pow(10, -precision));
}

void checkResult(const string &FuncName, const vector<float2> &Inputs,
                 const __nv_bfloat162 &Expect, const __nv_bfloat162 &Result,
                 const int precision) {
  cout << FuncName << "({" << Inputs[0].x << ", " << Inputs[0].y << "}";
  for (size_t i = 1; i < Inputs.size(); ++i) {
    cout << ", {" << Inputs[i].x << ", " << Inputs[i].y << "}";
  }
  cout << ") = " << fixed << setprecision(precision) << "{"
       << __bfloat162float(Result.x) << ", " << __bfloat162float(Result.y)
       << "} (expect {" << __bfloat162float(Expect.x) - pow(10, -precision)
       << " ~ " << __bfloat162float(Expect.x) + pow(10, -precision) << ", "
       << __bfloat162float(Expect.y) - pow(10, -precision) << " ~ "
       << __bfloat162float(Expect.y) + pow(10, -precision) << ")";
  cout.unsetf(ios::fixed);
  check(abs(__bfloat162float(Result.x) - __bfloat162float(Expect.x)) <
            pow(10, -precision) &&
        abs(__bfloat162float(Result.y) - __bfloat162float(Expect.y)) <
            pow(10, -precision));
}

void checkResult(const string &FuncName, const vector<float> &Inputs,
                 const __nv_bfloat16 &Expect, const __nv_bfloat16 &Result,
                 const int precision) {
  cout << FuncName << "(" << Inputs[0] << "";
  for (size_t i = 1; i < Inputs.size(); ++i) {
    cout << ", " << Inputs[i];
  }
  cout << ") = " << fixed << setprecision(precision) << __bfloat162float(Result)
       << " (expect " << __bfloat162float(Expect) - pow(10, -precision) << " ~ "
       << __bfloat162float(Expect) + pow(10, -precision) << ")";
  cout.unsetf(ios::fixed);
  check(abs(__bfloat162float(Result) - __bfloat162float(Expect)) <
        pow(10, -precision));
}

__global__ void bFloat1622float2(float *const Result, __nv_bfloat162 Input1) {
  auto ret = __bfloat1622float2(Input1);
  Result[0] = ret.x;
  Result[1] = ret.y;
}

void testBFloat1622float2Cases(
    const vector<pair<__nv_bfloat162, f2i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    bFloat1622float2<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__bfloat1622float2", {TestCase.first}, TestCase.second.first,
                {Result[0], Result[1]}, TestCase.second.second);
    auto ret = __bfloat1622float2(TestCase.first);
    Result[0] = ret.x;
    Result[1] = ret.y;
    checkResult("(host)__bfloat1622float2", {TestCase.first},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void bFloat162float(float *const Result, __nv_bfloat16 Input1) {
  *Result = __bfloat162float(Input1);
}

void testBFloat162floatCases(
    const vector<pair<__nv_bfloat16, fi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    bFloat162float<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__bfloat162float", {TestCase.first}, TestCase.second.first,
                *Result, TestCase.second.second);
    *Result = __bfloat162float(TestCase.first);
    checkResult("(host)__bfloat162float", {TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void float22bFloat162_rn(__nv_bfloat16 *const Result,
                                    float2 Input1) {
  auto ret = __float22bfloat162_rn(Input1);
  Result[0] = ret.x;
  Result[1] = ret.y;
}

void testFloat22bFloat162_rnCases(
    const vector<pair<float2, bf2i_pair>> &TestCases) {
  __nv_bfloat16 *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    float22bFloat162_rn<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__float22bfloat162_rn", {TestCase.first},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
    auto ret = __float22bfloat162_rn(TestCase.first);
    Result[0] = ret.x;
    Result[1] = ret.y;
    checkResult("(host)__float22bfloat162_rn", {TestCase.first},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void float2bFloat16(__nv_bfloat16 *const Result, float Input1) {
  *Result = __float2bfloat16(Input1);
}

void testFloat2bFloat16Cases(const vector<pair<float, bfi_pair>> &TestCases) {
  __nv_bfloat16 *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    float2bFloat16<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__float2bfloat16", {TestCase.first}, TestCase.second.first,
                *Result, TestCase.second.second);
    *Result = __float2bfloat16(TestCase.first);
    checkResult("(host)__float2bfloat16", {TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

int main() {
  testBFloat1622float2Cases({
      {{-0.3, -0.5}, {{-0.30078125, -0.5}, 16}},
      {{0.3, 0.5}, {{0.30078125, 0.5}, 16}},
      {{30, 50}, {{30, 50}, 14}},
      {{0.432643, 0.23654}, {{0.43359375, 0.236328125}, 16}},
  });
  testBFloat162floatCases({
      {-0.3, {-0.30078125, 16}},
      {0.3, {0.30078125, 16}},
      {30, {30, 14}},
      {0.432643, {0.43359375, 16}},
  });
  testFloat22bFloat162_rnCases({
      {{-0.3, -0.5}, {{-0.30078125, -0.5}, 16}},
      {{0.3, 0.5}, {{0.30078125, 0.5}, 16}},
      {{30, 50}, {{30, 50}, 14}},
      {{0.432643, 0.23654}, {{0.43359375, 0.236328125}, 16}},
  });
  testFloat2bFloat16Cases({
      {-0.3, {-0.30078125, 16}},
      {0.3, {0.30078125, 16}},
      {30, {30, 14}},
      {0.432643, {0.43359375, 16}},
  });
  cout << "passed " << passed << "/" << passed + failed << " cases!" << endl;
  if (failed) {
    cout << "failed!" << endl;
  }
  return failed;
}
