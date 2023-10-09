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

void checkResult(const string &FuncName, const vector<float> &Inputs,
                 const bool &Expect, const bool &Result) {
  cout << FuncName << "(" << Inputs[0];
  for (size_t i = 1; i < Inputs.size(); ++i) {
    cout << ", " << Inputs[i];
  }
  cout << ") = " << Result << " (expect " << Expect << ")";
  check(Result == Expect);
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

// Bfloat16 Comparison Functions

__global__ void hisnan(bool *const Result, __nv_bfloat16 Input1) {
  *Result = __hisnan(Input1);
}

void testHisnanCases(const vector<pair<__nv_bfloat16, bool>> &TestCases) {
  bool *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hisnan<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__hisnan", {TestCase.first}, TestCase.second, *Result);
  }
}

__global__ void hmax(float *const Result, __nv_bfloat16 Input1,
                     __nv_bfloat16 Input2) {
  *Result = __hmax(Input1, Input2);
}

void testHmaxCases(
    const vector<pair<pair<__nv_bfloat16, __nv_bfloat16>, bf16i_pair>>
        &TestCases) {
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

__global__ void hmin(float *const Result, __nv_bfloat16 Input1,
                     __nv_bfloat16 Input2) {
  *Result = __hmin(Input1, Input2);
}

void testHminCases(
    const vector<pair<pair<__nv_bfloat16, __nv_bfloat16>, bf16i_pair>>
        &TestCases) {
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

// Bfloat16 Math Functions

__global__ void _hceil(float *const Result, __nv_bfloat16 Input1) {
  *Result = hceil(Input1);
}

void testHceilCases(const vector<pair<__nv_bfloat16, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _hceil<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("hceil", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void _hcos(float *const Result, __nv_bfloat16 Input1) {
  *Result = hcos(Input1);
}

void testHcosCases(const vector<pair<__nv_bfloat16, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _hcos<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("hcos", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void _hexp(float *const Result, __nv_bfloat16 Input1) {
  *Result = hexp(Input1);
}

void testHexpCases(const vector<pair<__nv_bfloat16, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _hexp<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("hexp", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void _hexp10(float *const Result, __nv_bfloat16 Input1) {
  *Result = hexp10(Input1);
}

void testHexp10Cases(const vector<pair<__nv_bfloat16, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _hexp10<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("hexp10", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void _hexp2(float *const Result, __nv_bfloat16 Input1) {
  *Result = hexp2(Input1);
}

void testHexp2Cases(const vector<pair<__nv_bfloat16, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _hexp2<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("hexp2", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void _hfloor(float *const Result, __nv_bfloat16 Input1) {
  *Result = hfloor(Input1);
}

void testHfloorCases(const vector<pair<__nv_bfloat16, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _hfloor<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("hfloor", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void _hlog(float *const Result, __nv_bfloat16 Input1) {
  *Result = hlog(Input1);
}

void testHlogCases(const vector<pair<__nv_bfloat16, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _hlog<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("hlog", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void _hlog10(float *const Result, __nv_bfloat16 Input1) {
  *Result = hlog10(Input1);
}

void testHlog10Cases(const vector<pair<__nv_bfloat16, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _hlog10<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("hlog10", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void _hlog2(float *const Result, __nv_bfloat16 Input1) {
  *Result = hlog2(Input1);
}

void testHlog2Cases(const vector<pair<__nv_bfloat16, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _hlog2<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("hlog2", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void _hrint(float *const Result, __nv_bfloat16 Input1) {
  *Result = hrint(Input1);
}

void testHrintCases(const vector<pair<__nv_bfloat16, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _hrint<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("hrint", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void _hrsqrt(float *const Result, __nv_bfloat16 Input1) {
  *Result = hrsqrt(Input1);
}

void testHrsqrtCases(const vector<pair<__nv_bfloat16, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _hrsqrt<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("hrsqrt", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void _hsin(float *const Result, __nv_bfloat16 Input1) {
  *Result = hsin(Input1);
}

void testHsinCases(const vector<pair<__nv_bfloat16, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _hsin<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("hsin", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void _hsqrt(float *const Result, __nv_bfloat16 Input1) {
  *Result = hsqrt(Input1);
}

void testHsqrtCases(const vector<pair<__nv_bfloat16, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _hsqrt<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("hsqrt", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void _htrunc(float *const Result, __nv_bfloat16 Input1) {
  *Result = htrunc(Input1);
}

void testHtruncCases(const vector<pair<__nv_bfloat16, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _htrunc<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("htrunc", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
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
  testHisnanCases({
      {-0.3, false},
      {0.34, false},
      {0.8, false},
      {INFINITY, false},
      {NAN, true},
  });
  testHmaxCases({
      {{0, -0.4}, {0, 37}},
      {{0.7, 0.7}, {0.69921875, 16}},
      {{1, 4}, {4, 15}},
      {{NAN, 1}, {1, 15}},
      {{1, NAN}, {1, 15}},
  });
  testHminCases({
      {{0, -0.4}, {-0.400390625, 16}},
      {{0.7, 0.7}, {0.69921875, 16}},
      {{1, 4}, {1, 15}},
      {{NAN, 1}, {1, 15}},
      {{1, NAN}, {1, 15}},
  });
  testHceilCases({
      {-0.3, {0, 37}},
      {0.34, {1, 15}},
      {0.8, {1, 15}},
      {23, {23, 14}},
      {-12, {-12, 15}},
  });
  testHcosCases({
      {-0.3, {0.96, 2}},
      {0.34, {0.94140625, 16}},
      {0.8, {0.6953125, 16}},
      {23, {-0.53125, 16}},
      {-12, {0.84375, 16}},
  });
  testHexpCases({
      {-0.3, {0.74, 2}},
      {0.34, {1.40625, 15}},
      {0.8, {2.234375, 15}},
      {10, {22016, 11}},
      {-12, {0.00000613927841186523, 20}},
  });
  testHexp10Cases({
      {-0.3, {0.5, 16}},
      {0.34, {2.1875, 15}},
      {0.8, {6.3125, 15}},
      {4, {9984, 12}},
      {-12, {0.000000000001001865257421741, 27}},
  });
  testHexp2Cases({
      {-0.3, {0.8125, 16}},
      {0.34, {1.265625, 15}},
      {0.8, {1.7421875, 15}},
      {12, {4096, 12}},
      {-12, {0.000244140625, 19}},
  });
  testHfloorCases({
      {-0.3, {-1, 15}},
      {0.34, {0, 37}},
      {0.8, {0, 37}},
      {23, {23, 14}},
      {-12, {-12, 15}},
  });
  testHlogCases({
      {0.3, {-1.203125, 15}},
      {0.34, {-1.078125, 15}},
      {0.8, {-0.222, 3}},
      {23, {3.140625, 15}},
      {12, {2.484375, 15}},
  });
  testHlog10Cases({
      {0.3, {-0.5234375, 16}},
      {0.34, {-0.46875, 16}},
      {0.8, {-0.0966796875, 17}},
      {23, {1.359375, 15}},
      {12, {1.078125, 15}},
  });
  testHlog2Cases({
      {0.3, {-1.734375, 15}},
      {0.34, {-1.5546875, 15}},
      {0.8, {-0.3203125, 2}},
      {23, {4.53125, 15}},
      {12, {3.578125, 15}},
  });
  testHrintCases({
      {-0.3, {0, 37}},
      {0.34, {0., 37}},
      {0.8, {1, 15}},
      {23, {23, 14}},
      {-12, {-12, 14}},
  });
  testHrsqrtCases({
      {0.3, {1.8203125, 15}},
      {0.34, {1.71875, 15}},
      {0.8, {1.1171875, 15}},
      {23, {0.209, 3}},
      {12, {0.2890625, 16}},
  });
  testHsinCases({
      {-0.3, {-0.296875, 16}},
      {0.34, {0.333984375, 16}},
      {0.8, {0.71875, 16}},
      {23, {-0.84765625, 16}},
      {-12, {0.53515625, 16}},
  });
  testHsqrtCases({
      {0.3, {0.546875, 16}},
      {0.34, {0.58203125, 16}},
      {0.8, {0.89453125, 16}},
      {23, {4.78125, 15}},
      {12, {3.46875, 15}},
  });
  testHtruncCases({
      {-0.3, {0, 37}},
      {0.34, {0, 37}},
      {0.8, {0, 37}},
      {23, {23, 14}},
      {-12, {-12, 15}},
  });
  cout << "passed " << passed << "/" << passed + failed << " cases!" << endl;
  if (failed) {
    cout << "failed!" << endl;
  }
  return failed;
}
