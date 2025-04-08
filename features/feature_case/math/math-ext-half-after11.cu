// ===--------- math-ext-half-after11.cu -------------- *- CUDA -* --------===//
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
typedef pair<__nv_half, int> nv_hi_pair;

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

__global__ void hadd_rn(float *const Result, __half Input1, __half Input2) {
  *Result = __hadd_rn(Input1, Input2);
}
__global__ void hadd_rn_2(float *const Result, __nv_half Input1, __nv_half Input2) {
  *Result = __hadd_rn(Input1, Input2);
}

void testHadd_rnCases(
    const vector<pair<pair<__half, __half>, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hadd_rn<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hadd_rn", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

void testHadd_rnCases_2(
    const vector<pair<pair<__nv_half, __nv_half>, nv_hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hadd_rn<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hadd_rn", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void hfma_relu(float *const Result, __half Input1, __half Input2,
                          __half Input3) {
  *Result = __hfma_relu(Input1, Input2, Input3);
}

void testHfma_reluCases(
    const vector<pair<vector<__half>, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hfma_relu<<<1, 1>>>(Result, TestCase.first[0], TestCase.first[1],
                        TestCase.first[2]);
    cudaDeviceSynchronize();
    checkResult("__hfma_relu", TestCase.first, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void hmul_rn(float *const Result, __half Input1, __half Input2) {
  *Result = __hmul_rn(Input1, Input2);
}

void testHmul_rnCases(
    const vector<pair<pair<__half, __half>, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hmul_rn<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hmul_rn", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void hsub_rn(float *const Result, __half Input1, __half Input2) {
  *Result = __hsub_rn(Input1, Input2);
}

void testHsub_rnCases(
    const vector<pair<pair<__half, __half>, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hsub_rn<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hsub_rn", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

// Comparison Functions

__global__ void hmax(float *const Result, __half Input1, __half Input2) {
  *Result = __hmax(Input1, Input2);
}

void testHmaxCases(
    const vector<pair<pair<__half, __half>, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  // Boundary values.
  hmax<<<1, 1>>>(Result, NAN, NAN);
  cudaDeviceSynchronize();
  cout << "__hmax(nan, nan) = " << *Result << " (expect nan)";
  check(isnan(*Result));
  for (const auto &TestCase : TestCases) {
    hmax<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hmax", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void hmax_nan(float *const Result, __half Input1, __half Input2) {
  *Result = __hmax_nan(Input1, Input2);
}

void testHmax_nanCases(
    const vector<pair<pair<__half, __half>, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  // Boundary values.
  hmax_nan<<<1, 1>>>(Result, NAN, NAN);
  cudaDeviceSynchronize();
  cout << "__hmax_nan(nan, nan) = " << *Result << " (expect nan)";
  check(isnan(*Result));
  hmax_nan<<<1, 1>>>(Result, NAN, 1);
  cudaDeviceSynchronize();
  cout << "__hmax_nan(nan, 1) = " << *Result << " (expect nan)";
  check(isnan(*Result));
  hmax_nan<<<1, 1>>>(Result, 1, NAN);
  cudaDeviceSynchronize();
  cout << "__hmax_nan(1, nan) = " << *Result << " (expect nan)";
  check(isnan(*Result));
  for (const auto &TestCase : TestCases) {
    hmax_nan<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hmax_nan", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void hmin(float *const Result, __half Input1, __half Input2) {
  *Result = __hmin(Input1, Input2);
}

void testHminCases(
    const vector<pair<pair<__half, __half>, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  // Boundary values.
  hmin<<<1, 1>>>(Result, NAN, NAN);
  cudaDeviceSynchronize();
  cout << "__hmin(nan, nan) = " << *Result << " (expect nan)";
  check(isnan(*Result));
  for (const auto &TestCase : TestCases) {
    hmin<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hmin", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void hmin_nan(float *const Result, __half Input1, __half Input2) {
  *Result = __hmin_nan(Input1, Input2);
}

void testHmin_nanCases(
    const vector<pair<pair<__half, __half>, hi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  // Boundary values.
  hmin_nan<<<1, 1>>>(Result, NAN, NAN);
  cudaDeviceSynchronize();
  cout << "__hmin_nan(nan, nan) = " << *Result << " (expect nan)";
  check(isnan(*Result));
  hmin_nan<<<1, 1>>>(Result, NAN, 1);
  cudaDeviceSynchronize();
  cout << "__hmin_nan(nan, 1) = " << *Result << " (expect nan)";
  check(isnan(*Result));
  hmin_nan<<<1, 1>>>(Result, 1, NAN);
  cudaDeviceSynchronize();
  cout << "__hmin_nan(1, nan) = " << *Result << " (expect nan)";
  check(isnan(*Result));
  for (const auto &TestCase : TestCases) {
    hmin_nan<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hmin_nan", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

int main() {
  testHadd_rnCases({
      {{-0.3, -0.4}, {-0.7001953125, 16}},
      {{0.3, -0.4}, {-0.099853515625, 17}},
      {{0.3, 0.4}, {0.7001953125, 16}},
      {{0.3, 0.8}, {1.099609375, 15}},
      {{3, 4}, {7, 15}},
  });
  testHadd_rnCases_2({
      {{-0.3, -0.4}, {-0.7001953125, 16}},
      {{0.3, -0.4}, {-0.099853515625, 17}},
      {{0.3, 0.4}, {0.7001953125, 16}},
      {{0.3, 0.8}, {1.099609375, 15}},
      {{3, 4}, {7, 15}},
  });
  testHfma_reluCases({
      {{-0.3, -0.4, -0.2}, {0, 37}},
      {{0.3, -0.4, -0.1}, {0, 37}},
      {{0.3, 0.4, 0.1}, {0.219970703125, 16}},
      {{0.3, 0.4, 0}, {0.1199951171875, 17}},
      {{3, -4, -5}, {0, 37}},
  });
  testHmul_rnCases({
      {{-0.3, 0.4}, {-0.1199951171875, 17}},
      {{0.3, -4}, {-1.2001953125, 15}},
      {{0.3, 0.4}, {0.1199951171875, 17}},
      {{0.3, 0.8}, {0.239990234375, 16}},
      {{3, 4}, {12, 15}},
  });
  testHsub_rnCases({
      {{0, -0.4}, {0.39990234375, 16}},
      {{0.3, -0.4}, {0.7001953125, 16}},
      {{0.3, 0.4}, {-0.099853515625, 17}},
      {{0.3, -0.8}, {1.099609375, 15}},
      {{1, 4}, {-3, 15}},
  });
  testHmaxCases({
      {{0, -0.4}, {0, 37}},
      {{0.7, 0.7}, {0.7001953125, 16}},
      {{1, 4}, {4, 15}},
      {{NAN, 1}, {1, 15}},
      {{1, NAN}, {1, 15}},
  });
  testHmax_nanCases({
      {{0, -0.4}, {0, 37}},
      {{0.7, 0.7}, {0.7001953125, 16}},
      {{1, 4}, {4, 15}},
  });
  testHminCases({
      {{0, -0.4}, {-0.39990234375, 16}},
      {{0.7, 0.7}, {0.7001953125, 16}},
      {{1, 4}, {1, 15}},
      {{NAN, 1}, {1, 15}},
      {{1, NAN}, {1, 15}},
  });
  testHmin_nanCases({
      {{0, -0.4}, {-0.39990234375, 16}},
      {{0.7, 0.7}, {0.7001953125, 16}},
      {{1, 4}, {1, 15}},
  });
  cout << "passed " << passed << "/" << passed + failed << " cases!" << endl;
  if (failed) {
    cout << "failed!" << endl;
  }
  return failed;
}
