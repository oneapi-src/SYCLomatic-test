// ===------------- math-ext-half.cu------------------- *- CUDA -* --------===//
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

// Arithmetic Functions

__global__ void hadd_sat(float *const Result, __half Input1, __half Input2) {
  *Result = __hadd_sat(Input1, Input2);
}

void testHadd_sat(float *const Result, __half Input1, __half Input2) {
  hadd_sat<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHadd_satCases(
    const vector<pair<pair<__half, __half>, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHadd_sat(Result, TestCase.first.first, TestCase.first.second);
    checkResult("__hadd_sat", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void hfma_sat(float *const Result, __half Input1, __half Input2,
                         __half Input3) {
  *Result = __hfma_sat(Input1, Input2, Input3);
}

void testHfma_sat(float *const Result, __half Input1, __half Input2,
                  __half Input3) {
  hfma_sat<<<1, 1>>>(Result, Input1, Input2, Input3);
  cudaDeviceSynchronize();
}

void testHfma_satCases(const vector<pair<vector<__half>, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHfma_sat(Result, TestCase.first[0], TestCase.first[1],
                 TestCase.first[2]);
    checkResult("__hfma_sat", TestCase.first, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void hmul_sat(float *const Result, __half Input1, __half Input2) {
  *Result = __hmul_sat(Input1, Input2);
}

void testHmul_sat(float *const Result, __half Input1, __half Input2) {
  hmul_sat<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHmul_satCases(
    const vector<pair<pair<__half, __half>, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHmul_sat(Result, TestCase.first.first, TestCase.first.second);
    checkResult("__hmul_sat", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void hsub_sat(float *const Result, __half Input1, __half Input2) {
  *Result = __hsub_sat(Input1, Input2);
}

void testHsub_sat(float *const Result, __half Input1, __half Input2) {
  hsub_sat<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHsub_satCases(
    const vector<pair<pair<__half, __half>, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHsub_sat(Result, TestCase.first.first, TestCase.first.second);
    checkResult("__hsub_sat", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

// Comparison Functions

void checkResult(const string &FuncName, const vector<__half> &Inputs,
                 const bool &Expect, const bool &Result) {
  cout << FuncName << "(" << __half2float(Inputs[0]);
  for (size_t i = 1; i < Inputs.size(); ++i) {
    cout << ", " << __half2float(Inputs[i]);
  }
  cout << ") = " << Result << " (expect " << Expect << ")";
  check(Result == Expect);
}

__global__ void hequ(bool *const Result, __half Input1, __half Input2) {
  *Result = __hequ(Input1, Input2);
}

void testHequ(bool *const Result, __half Input1, __half Input2) {
  hequ<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHequCases(const vector<pair<pair<__half, __half>, bool>> &TestCases) {
  bool *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHequ(Result, TestCase.first.first, TestCase.first.second);
    checkResult("__hequ", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *Result);
  }
}

__global__ void hgeu(bool *const Result, __half Input1, __half Input2) {
  *Result = __hgeu(Input1, Input2);
}

void testHgeu(bool *const Result, __half Input1, __half Input2) {
  hgeu<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHgeuCases(const vector<pair<pair<__half, __half>, bool>> &TestCases) {
  bool *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHgeu(Result, TestCase.first.first, TestCase.first.second);
    checkResult("__hgeu", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *Result);
  }
}

__global__ void hgtu(bool *const Result, __half Input1, __half Input2) {
  *Result = __hgtu(Input1, Input2);
}

void testHgtu(bool *const Result, __half Input1, __half Input2) {
  hgtu<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHgtuCases(const vector<pair<pair<__half, __half>, bool>> &TestCases) {
  bool *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHgtu(Result, TestCase.first.first, TestCase.first.second);
    checkResult("__hgtu", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *Result);
  }
}

__global__ void hleu(bool *const Result, __half Input1, __half Input2) {
  *Result = __hleu(Input1, Input2);
}

void testHleu(bool *const Result, __half Input1, __half Input2) {
  hleu<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHleuCases(const vector<pair<pair<__half, __half>, bool>> &TestCases) {
  bool *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHleu(Result, TestCase.first.first, TestCase.first.second);
    checkResult("__hleu", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *Result);
  }
}

__global__ void hltu(bool *const Result, __half Input1, __half Input2) {
  *Result = __hltu(Input1, Input2);
}

void testHltu(bool *const Result, __half Input1, __half Input2) {
  hltu<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHltuCases(const vector<pair<pair<__half, __half>, bool>> &TestCases) {
  bool *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHltu(Result, TestCase.first.first, TestCase.first.second);
    checkResult("__hltu", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *Result);
  }
}

__global__ void hneu(bool *const Result, __half Input1, __half Input2) {
  *Result = __hneu(Input1, Input2);
}

void testHneu(bool *const Result, __half Input1, __half Input2) {
  hneu<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHneuCases(const vector<pair<pair<__half, __half>, bool>> &TestCases) {
  bool *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHneu(Result, TestCase.first.first, TestCase.first.second);
    checkResult("__hneu", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *Result);
  }
}

int main() {
  testHadd_satCases({
      {{-0.3, -0.4}, {0, 37}},
      {{0.3, -0.4}, {0, 37}},
      {{0.3, 0.4}, {0.7001953125, 16}},
      {{0.3, 0.8}, {1, 15}},
      {{3, 4}, {1, 15}},
  });
  testHfma_satCases({
      {{-0.3, -0.4, -0.2}, {0, 37}},
      {{0.3, -0.4, -0.1}, {0, 37}},
      {{0.3, 0.4, 0.1}, {0.219970703125, 16}},
      {{0.3, 0.4, 0}, {0.1199951171875, 17}},
      {{3, 4, 5}, {1, 15}},
  });
  testHmul_satCases({
      {{-0.3, 0.4}, {0, 37}},
      {{0.3, -4}, {0, 37}},
      {{0.3, 0.4}, {0.1199951171875, 17}},
      {{0.3, 0.8}, {0.239990234375, 16}},
      {{3, 4}, {1, 15}},
  });
  testHsub_satCases({
      {{0, -0.4}, {0.39990234375, 16}},
      {{0.3, -0.4}, {0.7001953125, 16}},
      {{0.3, 0.4}, {0, 37}},
      {{0.3, -0.8}, {1, 15}},
      {{1, 4}, {0, 37}},
  });
  testHequCases({
      {{0, -0.4}, false},
      {{0.7, 0.4}, false},
      {{0.7, 0.7}, true},
      {{1, 4}, false},
      {{NAN, 1}, true},
  });
  testHgeuCases({
      {{0, -0.4}, true},
      {{0.7, 0.4}, true},
      {{0.7, 0.7}, true},
      {{1, 4}, false},
      {{NAN, 1}, true},
  });
  testHgtuCases({
      {{0, -0.4}, true},
      {{0.7, 0.4}, true},
      {{0.7, 0.7}, false},
      {{1, 4}, false},
      {{NAN, 1}, true},
  });
  testHleuCases({
      {{0, -0.4}, false},
      {{0.7, 0.4}, false},
      {{0.7, 0.7}, true},
      {{1, 4}, true},
      {{NAN, 1}, true},
  });
  testHltuCases({
      {{0, -0.4}, false},
      {{0.7, 0.4}, false},
      {{0.7, 0.7}, false},
      {{1, 4}, true},
      {{NAN, 1}, true},
  });
  testHneuCases({
      {{0, -0.4}, true},
      {{0.7, 0.4}, true},
      {{0.7, 0.7}, false},
      {{1, 4}, true},
      {{NAN, 1}, true},
  });
  cout << "passed " << passed << "/" << passed + failed << " cases!" << endl;
  if (failed) {
    cout << "failed!" << endl;
  }
  return failed;
}
