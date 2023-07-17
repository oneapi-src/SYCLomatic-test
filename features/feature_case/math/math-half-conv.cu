// ===-------------- math-half-conv.cu -------=--------- *- CUDA -* -------===//
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

#include "cuda_fp16.h"

using namespace std;

typedef pair<__half, int> hi_pair;

int passed = 0;
int failed = 0;

void check(bool IsPassed) {
  if (IsPassed) {
    std::cout << " ---- passed" << std::endl;
    passed++;
  } else {
    std::cout << " ---- failed" << std::endl;
    failed++;
  }
}

void checkResult(const string &FuncName, const vector<float> &Inputs,
                 const float &Expect, const float &Result,
                 const int precision) {
  cout << FuncName << "(" << Inputs[0] << "";
  for (size_t i = 1; i < Inputs.size(); ++i) {
    cout << ", " << Inputs[i];
  }
  cout << ") = " << fixed << setprecision(precision) << Result << " (expect "
       << Expect - pow(10, -precision) << " ~ " << Expect + pow(10, -precision)
       << ")";
  cout.unsetf(ios::fixed);
  check(abs(Result - Expect) < pow(10, -precision));
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

void checkResult(const string &FuncName, const vector<__half> &Inputs,
                 const __half &Expect, const float &Result,
                 const int precision) {
  vector<float> FInputs;
  for (const auto &it : Inputs) {
    FInputs.push_back(__half2float(it));
  }
  float FExpect = __half2float(Expect);
  checkResult(FuncName, FInputs, FExpect, Result, precision);
}

void checkResult(const string &FuncName, const vector<__half2> &Inputs,
                 const __half2 &Expect, const float2 &Result,
                 const int precision) {
  vector<float2> FInputs;
  for (const auto &it : Inputs) {
    FInputs.push_back({__half2float(it.x), __half2float(it.y)});
  }
  float2 FExpect{__half2float(Expect.x), __half2float(Expect.y)};
  checkResult(FuncName, FInputs, FExpect, Result, precision);
}

__global__ void setValue(__half *Input1, const __half Input2) {
  *Input1 = Input2;
}

__global__ void setValue(__half2 *Input1, const __half2 Input2) {
  *Input1 = Input2;
}

__global__ void double2half(float *const Result, double Input1) {
  *Result = __double2half(Input1);
}

void testDouble2halfCases(const vector<pair<double, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    double2half<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__double2half", {TestCase.first}, TestCase.second.first,
                *Result, TestCase.second.second);
    *Result = __double2half(TestCase.first);
    checkResult("(host)__double2half", {TestCase.first}, TestCase.second.first,
                *Result, TestCase.second.second);
  }
}

__global__ void ldca(float *const Result, __half *Input1) {
  *Result = __ldca(Input1);
}

void testLdcaCases(const vector<pair<__half, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half *Input;
    cudaMallocManaged(&Input, sizeof(*Input));
    setValue<<<1, 1>>>(Input, TestCase.first);
    cudaDeviceSynchronize();
    ldca<<<1, 1>>>(Result, Input);
    cudaDeviceSynchronize();
    checkResult("__ldca", {TestCase.first}, TestCase.first, *Result,
                TestCase.second);
  }
}

__global__ void ldca(float *const Result, __half2 *Input1) {
  auto ret = __ldca(Input1);
  Result[0] = __half2float(ret.x);
  Result[1] = __half2float(ret.y);
}

void testLdcaCases2(const vector<pair<__half2, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half2 *Input;
    cudaMallocManaged(&Input, sizeof(*Input));
    setValue<<<1, 1>>>(Input, TestCase.first);
    cudaDeviceSynchronize();
    ldca<<<1, 1>>>(Result, Input);
    cudaDeviceSynchronize();
    checkResult("__ldca", {TestCase.first}, TestCase.first,
                {Result[0], Result[1]}, TestCase.second);
  }
}

__global__ void ldcg(float *const Result, __half *Input1) {
  *Result = __ldcg(Input1);
}

void testLdcgCases(const vector<pair<__half, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half *Input;
    cudaMallocManaged(&Input, sizeof(*Input));
    setValue<<<1, 1>>>(Input, TestCase.first);
    cudaDeviceSynchronize();
    ldcg<<<1, 1>>>(Result, Input);
    cudaDeviceSynchronize();
    checkResult("__ldcg", {TestCase.first}, TestCase.first, *Result,
                TestCase.second);
  }
}

__global__ void ldcg(float *const Result, __half2 *Input1) {
  auto ret = __ldcg(Input1);
  Result[0] = __half2float(ret.x);
  Result[1] = __half2float(ret.y);
}

void testLdcgCases2(const vector<pair<__half2, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half2 *Input;
    cudaMallocManaged(&Input, sizeof(*Input));
    setValue<<<1, 1>>>(Input, TestCase.first);
    cudaDeviceSynchronize();
    ldcg<<<1, 1>>>(Result, Input);
    cudaDeviceSynchronize();
    checkResult("__ldcg", {TestCase.first}, TestCase.first,
                {Result[0], Result[1]}, TestCase.second);
  }
}

__global__ void ldcs(float *const Result, __half *Input1) {
  *Result = __ldcs(Input1);
}

void testLdcsCases(const vector<pair<__half, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half *Input;
    cudaMallocManaged(&Input, sizeof(*Input));
    setValue<<<1, 1>>>(Input, TestCase.first);
    cudaDeviceSynchronize();
    ldcs<<<1, 1>>>(Result, Input);
    cudaDeviceSynchronize();
    checkResult("__ldcs", {TestCase.first}, TestCase.first, *Result,
                TestCase.second);
  }
}

__global__ void ldcs(float *const Result, __half2 *Input1) {
  auto ret = __ldcs(Input1);
  Result[0] = __half2float(ret.x);
  Result[1] = __half2float(ret.y);
}

void testLdcsCases2(const vector<pair<__half2, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half2 *Input;
    cudaMallocManaged(&Input, sizeof(*Input));
    setValue<<<1, 1>>>(Input, TestCase.first);
    cudaDeviceSynchronize();
    ldcs<<<1, 1>>>(Result, Input);
    cudaDeviceSynchronize();
    checkResult("__ldcs", {TestCase.first}, TestCase.first,
                {Result[0], Result[1]}, TestCase.second);
  }
}

__global__ void ldcv(float *const Result, __half *Input1) {
  *Result = __ldcv(Input1);
}

void testLdcvCases(const vector<pair<__half, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half *Input;
    cudaMallocManaged(&Input, sizeof(*Input));
    setValue<<<1, 1>>>(Input, TestCase.first);
    cudaDeviceSynchronize();
    ldcv<<<1, 1>>>(Result, Input);
    cudaDeviceSynchronize();
    checkResult("__ldcv", {TestCase.first}, TestCase.first, *Result,
                TestCase.second);
  }
}

__global__ void ldcv(float *const Result, __half2 *Input1) {
  auto ret = __ldcv(Input1);
  Result[0] = __half2float(ret.x);
  Result[1] = __half2float(ret.y);
}

void testLdcvCases2(const vector<pair<__half2, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half2 *Input;
    cudaMallocManaged(&Input, sizeof(*Input));
    setValue<<<1, 1>>>(Input, TestCase.first);
    cudaDeviceSynchronize();
    ldcv<<<1, 1>>>(Result, Input);
    cudaDeviceSynchronize();
    checkResult("__ldcv", {TestCase.first}, TestCase.first,
                {Result[0], Result[1]}, TestCase.second);
  }
}

__global__ void ldg(float *const Result, __half *Input1) {
  *Result = __ldg(Input1);
}

void testLdgCases(const vector<pair<__half, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half *Input;
    cudaMallocManaged(&Input, sizeof(*Input));
    setValue<<<1, 1>>>(Input, TestCase.first);
    cudaDeviceSynchronize();
    ldg<<<1, 1>>>(Result, Input);
    cudaDeviceSynchronize();
    checkResult("__ldg", {TestCase.first}, TestCase.first, *Result,
                TestCase.second);
  }
}

__global__ void ldg(float *const Result, __half2 *Input1) {
  auto ret = __ldg(Input1);
  Result[0] = __half2float(ret.x);
  Result[1] = __half2float(ret.y);
}

void testLdgCases2(const vector<pair<__half2, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half2 *Input;
    cudaMallocManaged(&Input, sizeof(*Input));
    setValue<<<1, 1>>>(Input, TestCase.first);
    cudaDeviceSynchronize();
    ldg<<<1, 1>>>(Result, Input);
    cudaDeviceSynchronize();
    checkResult("__ldg", {TestCase.first}, TestCase.first,
                {Result[0], Result[1]}, TestCase.second);
  }
}

__global__ void ldlu(float *const Result, __half *Input1) {
  *Result = __ldlu(Input1);
}

void testLdluCases(const vector<pair<__half, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half *Input;
    cudaMallocManaged(&Input, sizeof(*Input));
    setValue<<<1, 1>>>(Input, TestCase.first);
    cudaDeviceSynchronize();
    ldlu<<<1, 1>>>(Result, Input);
    cudaDeviceSynchronize();
    checkResult("__ldlu", {TestCase.first}, TestCase.first, *Result,
                TestCase.second);
  }
}

__global__ void ldlu(float *const Result, __half2 *Input1) {
  auto ret = __ldlu(Input1);
  Result[0] = __half2float(ret.x);
  Result[1] = __half2float(ret.y);
}

void testLdluCases2(const vector<pair<__half2, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    half2 *Input;
    cudaMallocManaged(&Input, sizeof(*Input));
    setValue<<<1, 1>>>(Input, TestCase.first);
    cudaDeviceSynchronize();
    ldlu<<<1, 1>>>(Result, Input);
    cudaDeviceSynchronize();
    checkResult("__ldlu", {TestCase.first}, TestCase.first,
                {Result[0], Result[1]}, TestCase.second);
  }
}

__global__ void stcg(float *const Result, __half Input1, __half *const Temp) {
  __stcg(Temp, Input1);
  *Result = __half2float(*Temp);
}

void testStcgCases(const vector<pair<__half, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  __half *Temp;
  cudaMallocManaged(&Temp, sizeof(*Temp));
  for (const auto &TestCase : TestCases) {
    stcg<<<1, 1>>>(Result, TestCase.first, Temp);
    cudaDeviceSynchronize();
    checkResult("__stcg", {TestCase.first}, TestCase.first, *Result,
                TestCase.second);
  }
}

__global__ void stcg(float *const Result, __half2 Input1, __half2 *const Temp) {
  __stcg(Temp, Input1);
  Result[0] = __half2float(Temp->x);
  Result[1] = __half2float(Temp->y);
}

void testStcgCases2(const vector<pair<__half2, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  __half2 *Temp;
  cudaMallocManaged(&Temp, sizeof(*Temp));
  for (const auto &TestCase : TestCases) {
    stcg<<<1, 1>>>(Result, TestCase.first, Temp);
    cudaDeviceSynchronize();
    checkResult("__stcg", {TestCase.first}, TestCase.first,
                {Result[0], Result[1]}, TestCase.second);
  }
}

__global__ void stcs(float *const Result, __half Input1, __half *const Temp) {
  __stcs(Temp, Input1);
  *Result = __half2float(*Temp);
}

void testStcsCases(const vector<pair<__half, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  __half *Temp;
  cudaMallocManaged(&Temp, sizeof(*Temp));
  for (const auto &TestCase : TestCases) {
    stcs<<<1, 1>>>(Result, TestCase.first, Temp);
    cudaDeviceSynchronize();
    checkResult("__stcs", {TestCase.first}, TestCase.first, *Result,
                TestCase.second);
  }
}

__global__ void stcs(float *const Result, __half2 Input1, __half2 *const Temp) {
  __stcs(Temp, Input1);
  Result[0] = __half2float(Temp->x);
  Result[1] = __half2float(Temp->y);
}

void testStcsCases2(const vector<pair<__half2, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  __half2 *Temp;
  cudaMallocManaged(&Temp, sizeof(*Temp));
  for (const auto &TestCase : TestCases) {
    stcs<<<1, 1>>>(Result, TestCase.first, Temp);
    cudaDeviceSynchronize();
    checkResult("__stcs", {TestCase.first}, TestCase.first,
                {Result[0], Result[1]}, TestCase.second);
  }
}

__global__ void stwb(float *const Result, __half Input1, __half *const Temp) {
  __stwb(Temp, Input1);
  *Result = __half2float(*Temp);
}

void testStwbCases(const vector<pair<__half, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  __half *Temp;
  cudaMallocManaged(&Temp, sizeof(*Temp));
  for (const auto &TestCase : TestCases) {
    stwb<<<1, 1>>>(Result, TestCase.first, Temp);
    cudaDeviceSynchronize();
    checkResult("__stwb", {TestCase.first}, TestCase.first, *Result,
                TestCase.second);
  }
}

__global__ void stwb(float *const Result, __half2 Input1, __half2 *const Temp) {
  __stwb(Temp, Input1);
  Result[0] = __half2float(Temp->x);
  Result[1] = __half2float(Temp->y);
}

void testStwbCases2(const vector<pair<__half2, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  __half2 *Temp;
  cudaMallocManaged(&Temp, sizeof(*Temp));
  for (const auto &TestCase : TestCases) {
    stwb<<<1, 1>>>(Result, TestCase.first, Temp);
    cudaDeviceSynchronize();
    checkResult("__stwb", {TestCase.first}, TestCase.first,
                {Result[0], Result[1]}, TestCase.second);
  }
}

__global__ void stwt(float *const Result, __half Input1, __half *const Temp) {
  __stwt(Temp, Input1);
  *Result = __half2float(*Temp);
}

void testStwtCases(const vector<pair<__half, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  __half *Temp;
  cudaMallocManaged(&Temp, sizeof(*Temp));
  for (const auto &TestCase : TestCases) {
    stwt<<<1, 1>>>(Result, TestCase.first, Temp);
    cudaDeviceSynchronize();
    checkResult("__stwt", {TestCase.first}, TestCase.first, *Result,
                TestCase.second);
  }
}

__global__ void stwt(float *const Result, __half2 Input1, __half2 *const Temp) {
  __stwt(Temp, Input1);
  Result[0] = __half2float(Temp->x);
  Result[1] = __half2float(Temp->y);
}

void testStwtCases2(const vector<pair<__half2, int>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  __half2 *Temp;
  cudaMallocManaged(&Temp, sizeof(*Temp));
  for (const auto &TestCase : TestCases) {
    stwt<<<1, 1>>>(Result, TestCase.first, Temp);
    cudaDeviceSynchronize();
    checkResult("__stwt", {TestCase.first}, TestCase.first,
                {Result[0], Result[1]}, TestCase.second);
  }
}

int main() {
  testDouble2halfCases({
      {-0.3, {-0.3, 4}},
      {0.3, {0.3, 4}},
      {0.5, {0.5, 16}},
      {23, {23, 14}},
  });
  testLdcaCases({
      {-0.3, 16},
      {-0.4, 16},
      {0, 37},
      {0.7, 16},
      {1, 15},
      {100.6, 14},
  });
  testLdcaCases2({
      {{-0.3, -0.4}, 16},
      {{0, 0.7}, 16},
      {{1, 100.6}, 14},
      {{100.6, 1}, 14},
  });
  testLdcgCases({
      {-0.3, 16},
      {-0.4, 16},
      {0, 37},
      {0.7, 16},
      {1, 15},
      {100.6, 14},
  });
  testLdcgCases2({
      {{-0.3, -0.4}, 16},
      {{0, 0.7}, 16},
      {{1, 100.6}, 14},
      {{100.6, 1}, 14},
  });
  testLdcsCases({
      {-0.3, 16},
      {-0.4, 16},
      {0, 37},
      {0.7, 16},
      {1, 15},
      {100.6, 14},
  });
  testLdcsCases2({
      {{-0.3, -0.4}, 16},
      {{0, 0.7}, 16},
      {{1, 100.6}, 14},
      {{100.6, 1}, 14},
  });
  testLdcvCases({
      {-0.3, 16},
      {-0.4, 16},
      {0, 37},
      {0.7, 16},
      {1, 15},
      {100.6, 14},
  });
  testLdcvCases2({
      {{-0.3, -0.4}, 16},
      {{0, 0.7}, 16},
      {{1, 100.6}, 14},
      {{100.6, 1}, 14},
  });
  testLdgCases({
      {-0.3, 16},
      {-0.4, 16},
      {0, 37},
      {0.7, 16},
      {1, 15},
      {100.6, 14},
  });
  testLdgCases2({
      {{-0.3, -0.4}, 16},
      {{0, 0.7}, 16},
      {{1, 100.6}, 14},
      {{100.6, 1}, 14},
  });
  testLdluCases({
      {-0.3, 16},
      {-0.4, 16},
      {0, 37},
      {0.7, 16},
      {1, 15},
      {100.6, 14},
  });
  testLdluCases2({
      {{-0.3, -0.4}, 16},
      {{0, 0.7}, 16},
      {{1, 100.6}, 14},
      {{100.6, 1}, 14},
  });
  testStcgCases({
      {-0.3, 16},
      {-0.4, 16},
      {0, 37},
      {0.7, 16},
      {1, 15},
      {100.6, 14},
  });
  testStcgCases2({
      {{-0.3, -0.4}, 16},
      {{0, 0.7}, 16},
      {{1, 100.6}, 14},
      {{100.6, 1}, 14},
  });
  testStcsCases({
      {-0.3, 16},
      {-0.4, 16},
      {0, 37},
      {0.7, 16},
      {1, 15},
      {100.6, 14},
  });
  testStcsCases2({
      {{-0.3, -0.4}, 16},
      {{0, 0.7}, 16},
      {{1, 100.6}, 14},
      {{100.6, 1}, 14},
  });
  testStwbCases({
      {-0.3, 16},
      {-0.4, 16},
      {0, 37},
      {0.7, 16},
      {1, 15},
      {100.6, 14},
  });
  testStwbCases2({
      {{-0.3, -0.4}, 16},
      {{0, 0.7}, 16},
      {{1, 100.6}, 14},
      {{100.6, 1}, 14},
  });
  testStwtCases({
      {-0.3, 16},
      {-0.4, 16},
      {0, 37},
      {0.7, 16},
      {1, 15},
      {100.6, 14},
  });
  testStwtCases2({
      {{-0.3, -0.4}, 16},
      {{0, 0.7}, 16},
      {{1, 100.6}, 14},
      {{100.6, 1}, 14},
  });
  cout << "passed " << passed << "/" << passed + failed << " cases!" << endl;
  if (failed) {
    cout << "failed!" << endl;
  }
  return failed;
}
