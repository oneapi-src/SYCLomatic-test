//====---------- math-emu-bf162.cu ---------- *- CUDA -* ------------------===//
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
typedef vector<__nv_bfloat162> bf162_vector;
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

__global__ void habs2(float *const Result, __nv_bfloat162 Input1) {
  auto ret = __habs2(Input1);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHabs2Cases(
    const vector<pair<__nv_bfloat162, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    habs2<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__habs2", {TestCase.first}, TestCase.second.first,
                {Result[0], Result[1]}, TestCase.second.second);
  }
}

__global__ void hadd2(float *const Result, __nv_bfloat162 Input1,
                      __nv_bfloat162 Input2) {
  auto ret = __hadd2(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHadd2Cases(const vector<pair<bf162_pair, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hadd2<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hadd2", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hadd2_rn(float *const Result, __nv_bfloat162 Input1,
                         __nv_bfloat162 Input2) {
  auto ret = __hadd2_rn(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHadd2_rnCases(const vector<pair<bf162_pair, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hadd2_rn<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hadd2_rn", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hadd2_sat(float *const Result, __nv_bfloat162 Input1,
                          __nv_bfloat162 Input2) {
  auto ret = __hadd2_sat(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHadd2_satCases(
    const vector<pair<bf162_pair, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hadd2_sat<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hadd2_sat", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hcmadd(float *const Result, __nv_bfloat162 Input1,
                       __nv_bfloat162 Input2, __nv_bfloat162 Input3) {
  auto ret = __hcmadd(Input1, Input2, Input3);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHcmaddCases(const vector<pair<bf162_vector, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hcmadd<<<1, 1>>>(Result, TestCase.first[0], TestCase.first[1],
                     TestCase.first[2]);
    cudaDeviceSynchronize();
    checkResult("__hcmadd", TestCase.first, TestCase.second.first,
                {Result[0], Result[1]}, TestCase.second.second);
    if (TestCase.first.size() != 3) {
      failed++;
      cout << " ---- failed" << endl;
      return;
    }
  }
}

__global__ void hfma2(float *const Result, __nv_bfloat162 Input1,
                      __nv_bfloat162 Input2, __nv_bfloat162 Input3) {
  auto ret = __hfma2(Input1, Input2, Input3);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHfma2Cases(const vector<pair<bf162_vector, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hfma2<<<1, 1>>>(Result, TestCase.first[0], TestCase.first[1],
                    TestCase.first[2]);
    cudaDeviceSynchronize();
    checkResult("__hfma2", TestCase.first, TestCase.second.first,
                {Result[0], Result[1]}, TestCase.second.second);
    if (TestCase.first.size() != 3) {
      failed++;
      cout << " ---- failed" << endl;
      return;
    }
  }
}

__global__ void hfma2_sat(float *const Result, __nv_bfloat162 Input1,
                          __nv_bfloat162 Input2, __nv_bfloat162 Input3) {
  auto ret = __hfma2_sat(Input1, Input2, Input3);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHfma2_satCases(
    const vector<pair<bf162_vector, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hfma2_sat<<<1, 1>>>(Result, TestCase.first[0], TestCase.first[1],
                        TestCase.first[2]);
    cudaDeviceSynchronize();
    checkResult("__hfma2_sat", TestCase.first, TestCase.second.first,
                {Result[0], Result[1]}, TestCase.second.second);
    if (TestCase.first.size() != 3) {
      failed++;
      cout << " ---- failed" << endl;
      return;
    }
  }
}

__global__ void hmul2(float *const Result, __nv_bfloat162 Input1,
                      __nv_bfloat162 Input2) {
  auto ret = __hmul2(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHmul2Cases(const vector<pair<bf162_pair, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hmul2<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hmul2", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hmul2_rn(float *const Result, __nv_bfloat162 Input1,
                         __nv_bfloat162 Input2) {
  auto ret = __hmul2_rn(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHmul2_rnCases(const vector<pair<bf162_pair, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hmul2_rn<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hmul2_rn", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hmul2_sat(float *const Result, __nv_bfloat162 Input1,
                          __nv_bfloat162 Input2) {
  auto ret = __hmul2_sat(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHmul2_satCases(
    const vector<pair<bf162_pair, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hmul2_sat<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hmul2_sat", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hneg2(float *const Result, __nv_bfloat162 Input1) {
  auto ret = __hneg2(Input1);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHneg2Cases(
    const vector<pair<__nv_bfloat162, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hneg2<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__hneg2", {TestCase.first}, TestCase.second.first,
                {Result[0], Result[1]}, TestCase.second.second);
  }
}

__global__ void hsub2(float *const Result, __nv_bfloat162 Input1,
                      __nv_bfloat162 Input2) {
  auto ret = __hsub2(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHsub2Cases(const vector<pair<bf162_pair, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hsub2<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hsub2", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hsub2_rn(float *const Result, __nv_bfloat162 Input1,
                         __nv_bfloat162 Input2) {
  auto ret = __hsub2_rn(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHsub2_rnCases(const vector<pair<bf162_pair, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hsub2_rn<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hsub2_rn", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hsub2_sat(float *const Result, __nv_bfloat162 Input1,
                          __nv_bfloat162 Input2) {
  auto ret = __hsub2_sat(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHsub2_satCases(
    const vector<pair<bf162_pair, bf162i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hsub2_sat<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hsub2_sat", {TestCase.first.first, TestCase.first.second},
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
  testHabs2Cases({
      {{-0.3, -5}, {{0.30078125, 5}, 15}},
      {{0.3, 5}, {{0.30078125, 5}, 15}},
      {{0.4, 0.2}, {{0.400390625, 0.2001953125}, 16}},
      {{0.3, 0.5}, {{0.30078125, 0.5}, 16}},
      {{3, 5}, {{3, 5}, 15}},
  });
  testHadd2Cases({
      {{{-0.3, -0.5}, {-0.4, -0.6}}, {{-0.703125, -1.1015625}, 15}},
      {{{0.3, 0.5}, {-0.4, 0.6}}, {{-0.099609375, 1.1015625}, 15}},
      {{{0.3, 0.5}, {0.4, 0.2}}, {{0.703125, 0.69921875}, 16}},
      {{{0.3, 0.5}, {0.4, 0.6}}, {{0.703125, 1.1015625}, 15}},
      {{{3, 5}, {4, 6}}, {{7, 11}, 15}},
  });
  testHadd2_rnCases({
      {{{-0.3, -0.5}, {-0.4, -0.6}}, {{-0.703125, -1.1015625}, 15}},
      {{{0.3, 0.5}, {-0.4, 0.6}}, {{-0.099609375, 1.1015625}, 15}},
      {{{0.3, 0.5}, {0.4, 0.2}}, {{0.703125, 0.69921875}, 16}},
      {{{0.3, 0.5}, {0.4, 0.6}}, {{0.703125, 1.1015625}, 15}},
      {{{3, 5}, {4, 6}}, {{7, 11}, 15}},
  });
  testHadd2_satCases({
      {{{-0.3, -0.5}, {-0.4, -0.6}}, {{0, 0}, 37}},
      {{{0.3, 0.5}, {-0.4, 0.6}}, {{0, 1}, 15}},
      {{{0.3, 0.5}, {0.4, 0.2}}, {{0.703125, 0.69921875}, 16}},
      {{{0.3, 0.5}, {0.4, 0.6}}, {{0.703125, 1}, 15}},
      {{{3, 5}, {4, 6}}, {{1, 1}, 15}},
  });
  testHcmaddCases({
      {{{-0.3, -0.5}, {-0.4, -0.6}, {-0.2, -0.7}}, {{-0.38, -0.32}, 2}},
      {{{0.3, 0.5}, {-0.4, 0.6}, {-0.1, 0.2}},
       {{-0.5234375, 0.1806640625}, 16}},
      {{{0.3, 0.5}, {0.4, 0.2}, {0.1, 0.1}}, {{0.12060546875, 0.359375}, 16}},
      {{{0.3, 0.5}, {0.4, 0.6}, {0, 0.3}}, {{-0.18, 0.68}, 2}},
      {{{3, 5}, {4, 6}, {5, 8}}, {{-13, 46}, 14}},
  });
  testHfma2Cases({
      {{{-0.3, -0.5}, {-0.4, -0.6}, {-0.2, -0.7}},
       {{-0.07958984375, -0.3984375}, 16}},
      {{{0.3, 0.5}, {-0.4, 0.6}, {-0.1, 0.2}}, {{-0.220703125, 0.5}, 16}},
      {{{0.3, 0.5}, {0.4, 0.2}, {0.1, 0.1}}, {{0.220703125, 0.2001953125}, 16}},
      {{{0.3, 0.5}, {0.4, 0.6}, {0, 0.3}}, {{0.12060546875, 0.6015625}, 16}},
      {{{3, 5}, {4, 6}, {5, 8}}, {{17, 38}, 14}},
  });
  testHfma2_satCases({
      {{{-0.3, -0.5}, {-0.4, -0.6}, {-0.2, -0.7}}, {{0, 0}, 37}},
      {{{0.3, 0.5}, {-0.4, 0.6}, {-0.1, 0.2}}, {{0, 0.5}, 16}},
      {{{0.3, 0.5}, {0.4, 0.2}, {0.1, 0.1}}, {{0.220703125, 0.2001953125}, 16}},
      {{{0.3, 0.5}, {0.4, 0.6}, {0, 0.3}}, {{0.12060546875, 0.6015625}, 16}},
      {{{3, 5}, {4, 6}, {5, 8}}, {{1, 1}, 15}},
  });
  testHmul2Cases({
      {{{-0.3, -0.5}, {-0.4, -0.6}}, {{0.12060546875, 0.30078125}, 16}},
      {{{0.3, 0.5}, {-0.4, 0.6}}, {{-0.12060546875, 0.30078125}, 16}},
      {{{0.3, 0.5}, {0.4, 0.2}}, {{0.12060546875, 0.10009765625}, 17}},
      {{{0.3, 0.5}, {0.4, 0.6}}, {{0.12060546875, 0.30078125}, 16}},
      {{{3, 5}, {4, 6}}, {{12, 30}, 14}},
  });
  testHmul2_rnCases({
      {{{-0.3, -0.5}, {-0.4, -0.6}}, {{0.12060546875, 0.30078125}, 16}},
      {{{0.3, 0.5}, {-0.4, 0.6}}, {{-0.12060546875, 0.30078125}, 16}},
      {{{0.3, 0.5}, {0.4, 0.2}}, {{0.12060546875, 0.10009765625}, 17}},
      {{{0.3, 0.5}, {0.4, 0.6}}, {{0.12060546875, 0.30078125}, 16}},
      {{{3, 5}, {4, 6}}, {{12, 30}, 14}},
  });
  testHmul2_satCases({
      {{{-0.3, -0.5}, {-0.4, -0.6}}, {{0.12060546875, 0.30078125}, 16}},
      {{{0.3, 0.5}, {-0.4, 0.6}}, {{0, 0.30078125}, 16}},
      {{{0.3, 0.5}, {0.4, 0.2}}, {{0.12060546875, 0.10009765625}, 17}},
      {{{0.3, 0.5}, {0.4, 0.6}}, {{0.12060546875, 0.30078125}, 16}},
      {{{3, 5}, {4, 6}}, {{1, 1}, 15}},
  });
  testHneg2Cases({
      {{-0.3, -5}, {{0.30078125, 5}, 15}},
      {{0.3, 5}, {{-0.30078125, -5}, 15}},
      {{0.4, 0.2}, {{-0.400390625, -0.2001953125}, 16}},
      {{0.3, 0.5}, {{-0.30078125, -0.5}, 16}},
      {{3, 5}, {{-3, -5}, 15}},
  });
  testHsub2Cases({
      {{{-0.3, -0.5}, {-0.4, -0.6}}, {{0.099609375, 0.1015625}, 17}},
      {{{0.3, 0.5}, {-0.4, 0.6}}, {{0.703125, -0.1015625}, 16}},
      {{{0.3, 0.5}, {0.4, 0.2}}, {{-0.099609375, 0.30078125}, 16}},
      {{{0.3, 0.5}, {0.4, 0.6}}, {{-0.099609375, -0.1015625}, 17}},
      {{{3, 5}, {4, 6}}, {{-1, -1}, 15}},
  });
  testHsub2_rnCases({
      {{{-0.3, -0.5}, {-0.4, -0.6}}, {{0.099609375, 0.1015625}, 17}},
      {{{0.3, 0.5}, {-0.4, 0.6}}, {{0.703125, -0.1015625}, 16}},
      {{{0.3, 0.5}, {0.4, 0.2}}, {{-0.099609375, 0.30078125}, 16}},
      {{{0.3, 0.5}, {0.4, 0.6}}, {{-0.099609375, -0.1015625}, 17}},
      {{{3, 5}, {4, 6}}, {{-1, -1}, 15}},
  });
  testHsub2_satCases({
      {{{-0.3, -0.5}, {-0.4, -0.6}}, {{0.099609375, 0.1015625}, 17}},
      {{{0.3, 0.5}, {-0.4, 0.6}}, {{0.703125, 0}, 16}},
      {{{0.3, 0.5}, {0.4, 0.2}}, {{0, 0.30078125}, 16}},
      {{{0.3, 0.5}, {0.4, 0.6}}, {{0, 0}, 37}},
      {{{3, 5}, {-4, -6}}, {{1, 1}, 15}},
  });
  cout << "passed " << passed << "/" << passed + failed << " cases!" << endl;
  if (failed) {
    cout << "failed!" << endl;
  }
  return failed;
}
