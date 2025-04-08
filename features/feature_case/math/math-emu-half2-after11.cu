// ====--------- math-emu-half2-after11.cu --------- *- CUDA -* -----------===//
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

typedef pair<__half2, __half2> half2_pair;
typedef pair<__nv_half2, __nv_half2> nv_half2_pair;
typedef vector<__half2> half2_vector;
typedef pair<__half2, int> h2i_pair;
typedef pair<__half2, int> nv_h2i_pair;

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
  cout << ") = {" << fixed << setprecision(precision) << Result.x << ", "
       << Result.y << "} (expect {" << Expect.x - pow(10, -precision) << " ~ "
       << Expect.x + pow(10, -precision) << ", "
       << Expect.y - pow(10, -precision) << " ~ "
       << Expect.y + pow(10, -precision) << "})";
  cout.unsetf(ios::fixed);
  check(abs(Result.x - Expect.x) < pow(10, -precision) &&
        abs(Result.y - Expect.y) < pow(10, -precision));
}

void checkResult(const string &FuncName, const half2_vector &Inputs,
                 const __half2 &Expect, const float2 &Result,
                 const int precision) {
  vector<float2> FInputs;
  for (const auto &it : Inputs) {
    FInputs.push_back(__half22float2(it));
  }
  float2 FExpect = __half22float2(Expect);
  checkResult(FuncName, FInputs, FExpect, Result, precision);
}

__global__ void hadd2_rn(float *const Result, __half2 Input1, __half2 Input2) {
  auto ret = __hadd2_rn(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

__global__ void nv_hadd2_rn(float *const Result, __nv_half2 Input1, __nv_half2 Input2) {
  auto ret = __hadd2_rn(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}
void testHadd2_rnCases_2(const vector<pair<nv_half2_pair, nv_h2i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    nv_hadd2_rn<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("nv__hadd2_rn", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

void testHadd2_rnCases(const vector<pair<half2_pair, h2i_pair>> &TestCases) {
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

__global__ void hcmadd(float *const Result, __half2 Input1, __half2 Input2,
                       __half2 Input3) {
  auto ret = __hcmadd(Input1, Input2, Input3);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHcmaddCases(const vector<pair<half2_vector, h2i_pair>> &TestCases) {
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

__global__ void hfma2_relu(float *const Result, __half2 Input1, __half2 Input2,
                           __half2 Input3) {
  auto ret = __hfma2_relu(Input1, Input2, Input3);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHfma2_reluCases(
    const vector<pair<half2_vector, h2i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hfma2_relu<<<1, 1>>>(Result, TestCase.first[0], TestCase.first[1],
                         TestCase.first[2]);
    cudaDeviceSynchronize();
    checkResult("__hfma2_relu", TestCase.first, TestCase.second.first,
                {Result[0], Result[1]}, TestCase.second.second);
    if (TestCase.first.size() != 3) {
      failed++;
      cout << " ---- failed" << endl;
      return;
    }
  }
}

__global__ void hmul2_rn(float *const Result, __half2 Input1, __half2 Input2) {
  auto ret = __hmul2_rn(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHmul2_rnCases(const vector<pair<half2_pair, h2i_pair>> &TestCases) {
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

__global__ void hsub2_rn(float *const Result, __half2 Input1, __half2 Input2) {
  auto ret = __hsub2_rn(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHsub2_rnCases(const vector<pair<half2_pair, h2i_pair>> &TestCases) {
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

__global__ void hmax2(float *const Result, __half2 Input1, __half2 Input2) {
  auto ret = __hmax2(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHmax2Cases(const vector<pair<half2_pair, h2i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  // Boundary values.
  hmax2<<<1, 1>>>(Result, {NAN, NAN}, {NAN, NAN});
  cudaDeviceSynchronize();
  cout << "__hmax2({nan, nan}, {nan, nan}) = {" << Result[0] << ", "
       << Result[1] << "} (expect {nan, nan})";
  check(isnan(Result[0]) && isnan(Result[1]));
  for (const auto &TestCase : TestCases) {
    hmax2<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hmax2", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hmax2_nan(float *const Result, __half2 Input1, __half2 Input2) {
  auto ret = __hmax2_nan(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHmax2_nanCases(const vector<pair<half2_pair, h2i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  // Boundary values.
  hmax2_nan<<<1, 1>>>(Result, {NAN, NAN}, {NAN, NAN});
  cudaDeviceSynchronize();
  cout << "__hmax2_nan({nan, nan}, {nan, nan}) = {" << Result[0] << ", "
       << Result[1] << "} (expect {nan, nan})";
  check(isnan(Result[0]) && isnan(Result[1]));
  hmax2_nan<<<1, 1>>>(Result, {NAN, 1}, {1, NAN});
  cudaDeviceSynchronize();
  cout << "__hmax2_nan({nan, 1}, {1, nan}) = {" << Result[0] << ", "
       << Result[1] << "} (expect {nan, nan})";
  check(isnan(Result[0]) && isnan(Result[1]));
  hmax2_nan<<<1, 1>>>(Result, {1, NAN}, {NAN, 1});
  cudaDeviceSynchronize();
  cout << "__hmax2_nan({1, nan}, {nan, 1}) = {" << Result[0] << ", "
       << Result[1] << "} (expect {nan, nan})";
  check(isnan(Result[0]) && isnan(Result[1]));
  for (const auto &TestCase : TestCases) {
    hmax2_nan<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hmax2_nan", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hmin2(float *const Result, __half2 Input1, __half2 Input2) {
  auto ret = __hmin2(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHmin2Cases(const vector<pair<half2_pair, h2i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  // Boundary values.
  hmin2<<<1, 1>>>(Result, {NAN, NAN}, {NAN, NAN});
  cudaDeviceSynchronize();
  cout << "__hmin2({nan, nan}, {nan, nan}) = {" << Result[0] << ", "
       << Result[1] << "} (expect {nan, nan})";
  check(isnan(Result[0]) && isnan(Result[1]));
  for (const auto &TestCase : TestCases) {
    hmin2<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hmin2", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hmin2_nan(float *const Result, __half2 Input1, __half2 Input2) {
  auto ret = __hmin2_nan(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHmin2_nanCases(const vector<pair<half2_pair, h2i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  // Boundary values.
  hmin2_nan<<<1, 1>>>(Result, {NAN, NAN}, {NAN, NAN});
  cudaDeviceSynchronize();
  cout << "__hmin2_nan({nan, nan}, {nan, nan}) = {" << Result[0] << ", "
       << Result[1] << "} (expect {nan, nan})";
  check(isnan(Result[0]) && isnan(Result[1]));
  hmin2_nan<<<1, 1>>>(Result, {NAN, 1}, {1, NAN});
  cudaDeviceSynchronize();
  cout << "__hmin2_nan({nan, 1}, {1, nan}) = {" << Result[0] << ", "
       << Result[1] << "} (expect {nan, nan})";
  check(isnan(Result[0]) && isnan(Result[1]));
  hmin2_nan<<<1, 1>>>(Result, {1, NAN}, {NAN, 1});
  cudaDeviceSynchronize();
  cout << "__hmin2_nan({1, nan}, {nan, 1}) = {" << Result[0] << ", "
       << Result[1] << "} (expect {nan, nan})";
  check(isnan(Result[0]) && isnan(Result[1]));
  for (const auto &TestCase : TestCases) {
    hmin2_nan<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hmin2_nan", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

int main() {
  testHadd2_rnCases({
      {{{-0.3, -0.5}, {-0.4, -0.6}}, {{-0.7001953125, -1.099609375}, 15}},
      {{{0.3, 0.5}, {-0.4, 0.6}}, {{-0.099853515625, 1.099609375}, 15}},
      {{{0.3, 0.5}, {0.4, 0.2}}, {{0.7001953125, 0.7001953125}, 16}},
      {{{0.3, 0.5}, {0.4, 0.6}}, {{0.7001953125, 1.099609375}, 15}},
      {{{3, 5}, {4, 6}}, {{7, 11}, 15}},
  });
  testHadd2_rnCases_2({
      {{{-0.3, -0.5}, {-0.4, -0.6}}, {{-0.7001953125, -1.099609375}, 15}},
      {{{0.3, 0.5}, {-0.4, 0.6}}, {{-0.099853515625, 1.099609375}, 15}},
      {{{0.3, 0.5}, {0.4, 0.2}}, {{0.7001953125, 0.7001953125}, 16}},
      {{{0.3, 0.5}, {0.4, 0.6}}, {{0.7001953125, 1.099609375}, 15}},
      {{{3, 5}, {4, 6}}, {{7, 11}, 15}},
  });
  testHcmaddCases({
      // Notice: Use no-fast-math, so will has different precision.
      {{{-0.3, -0.5}, {-0.4, -0.6}, {-0.2, -0.7}}, {{-0.38, -0.32}, 3}},
      {{{0.3, 0.5}, {-0.4, 0.6}, {-0.1, 0.2}}, {{-0.52, 0.18}, 3}},
      {{{0.3, 0.5}, {0.4, 0.2}, {0.1, 0.1}},
       {{0.1199951171875, 0.35986328125}, 16}},
      {{{0.3, 0.5}, {0.4, 0.6}, {0, 0.3}}, {{-0.18, 0.68}, 3}},
      {{{3, 5}, {4, 6}, {5, 8}}, {{-13, 46}, 14}},
  });
  testHfma2_reluCases({
      {{{-0.3, -0.5}, {-0.4, -0.6}, {-0.2, -0.7}}, {{0, 0}, 37}},
      {{{0.3, 0.5}, {-0.4, 0.6}, {-0.1, 0.2}}, {{0, 0.5}, 16}},
      {{{0.3, 0.5}, {0.4, 0.2}, {0.1, 0.1}},
       {{0.219970703125, 0.199951171875}, 16}},
      {{{0.3, 0.5}, {0.4, 0.6}, {0, 0.3}},
       {{0.1199951171875, 0.60009765625}, 16}},
      {{{3, 5}, {4, 6}, {5, 8}}, {{17, 38}, 14}},
  });
  testHmul2_rnCases({
      {{{-0.3, -5}, {0.4, 6}}, {{-0.1199951171875, -30}, 14}},
      {{{0.3, 5}, {-4, 0.6}}, {{-1.2001953125, 3}, 15}},
      {{{0.3, 0.5}, {0.4, 0.2}}, {{0.1199951171875, 0.0999755859375}, 17}},
      {{{0.3, 0.5}, {0.4, 0.6}}, {{0.1199951171875, 0.300048828125}, 16}},
      {{{3, 5}, {4, 6}}, {{12, 30}, 14}},
  });
  testHsub2_rnCases({
      {{{0, 0}, {-0.4, -0.6}}, {{0.39990234375, 0.60009765625}, 16}},
      {{{0, 1}, {0.4, 0.6}}, {{-0.39990234375, 0.39990234375}, 16}},
      {{{0.7, 0.7}, {0.4, 0.2}}, {{0.30029296875, 0.5}, 16}},
      {{{0.7, 2}, {0.4, 0.6}}, {{0.30029296875, 1.400390625}, 15}},
      {{{1, 1}, {4, 6}}, {{-3, -5}, 15}},
  });
  testHmax2Cases({
      {{{0, 0}, {-0.4, -0.6}}, {{0, 0}, 37}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {{0.7001953125, 0.7001953125}, 16}},
      {{{0.7, 2}, {0.7, 2}}, {{0.7001953125, 2}, 15}},
      {{{1, 1}, {4, NAN}}, {{4, 1}, 15}},
      {{{NAN, 1}, {1, 1}}, {{1, 1}, 15}},
  });
  testHmax2_nanCases({
      {{{0, 0}, {-0.4, -0.6}}, {{0, 0}, 37}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {{0.7001953125, 0.7001953125}, 16}},
      {{{0.7, 2}, {0.7, 2}}, {{0.7001953125, 2}, 15}},
  });
  testHmin2Cases({
      {{{0, 0}, {-0.4, -0.6}}, {{-0.39990234375, -0.60009765625}, 16}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {{0.39990234375, 0.7001953125}, 16}},
      {{{0.7, 2}, {0.7, 2}}, {{0.7001953125, 2}, 15}},
      {{{1, 1}, {4, NAN}}, {{1, 1}, 15}},
      {{{NAN, 1}, {1, 1}}, {{1, 1}, 15}},
  });
  testHmin2_nanCases({
      {{{0, 0}, {-0.4, -0.6}}, {{-0.39990234375, -0.60009765625}, 16}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {{0.39990234375, 0.7001953125}, 16}},
      {{{0.7, 2}, {0.7, 2}}, {{0.7001953125, 2}, 15}},
  });
  cout << "passed " << passed << "/" << passed + failed << " cases!" << endl;
  if (failed) {
    cout << "failed!" << endl;
  }
  return failed;
}
