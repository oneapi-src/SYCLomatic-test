// ====---------- math-emu-half2.cu---------- *- CUDA -* ------------------===//
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
typedef vector<__half2> half2_vector;
typedef pair<__half2, int> h2i_pair;

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

__global__ void hadd2_sat(float *const Result, __half2 Input1, __half2 Input2) {
  auto ret = __hadd2_sat(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHadd2_sat(float *const Result, __half2 Input1, __half2 Input2) {
  hadd2_sat<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHadd2_satCases(const vector<pair<half2_pair, h2i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHadd2_sat(Result, TestCase.first.first, TestCase.first.second);
    checkResult("__hadd2_sat", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hfma2_sat(float *const Result, __half2 Input1, __half2 Input2,
                          __half2 Input3) {
  auto ret = __hfma2_sat(Input1, Input2, Input3);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHfma2_sat(float *const Result, __half2 Input1, __half2 Input2,
                   __half2 Input3) {
  hfma2_sat<<<1, 1>>>(Result, Input1, Input2, Input3);
  cudaDeviceSynchronize();
}

void testHfma2_satCases(const vector<pair<half2_vector, h2i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHfma2_sat(Result, TestCase.first[0], TestCase.first[1],
                  TestCase.first[2]);
    checkResult("__hfma2_sat", TestCase.first, TestCase.second.first,
                {Result[0], Result[1]}, TestCase.second.second);
    if (TestCase.first.size() != 3) {
      failed++;
      cout << " ---- failed" << endl;
      return;
    }
  }
}

__global__ void hmul2_sat(float *const Result, __half2 Input1, __half2 Input2) {
  auto ret = __hmul2_sat(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHmul2_sat(float *const Result, __half2 Input1, __half2 Input2) {
  hmul2_sat<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHmul2_satCases(const vector<pair<half2_pair, h2i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHmul2_sat(Result, TestCase.first.first, TestCase.first.second);
    checkResult("__hmul2_sat", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hsub2_sat(float *const Result, __half2 Input1, __half2 Input2) {
  auto ret = __hsub2_sat(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHsub2_sat(float *const Result, __half2 Input1, __half2 Input2) {
  hsub2_sat<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHsub2_satCases(const vector<pair<half2_pair, h2i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHsub2_sat(Result, TestCase.first.first, TestCase.first.second);
    checkResult("__hsub2_sat", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

void printResultBool(const string &FuncName, const half2_vector &Inputs,
                     const bool &Expect, const bool &Result) {
  cout << FuncName << "({" << __low2float(Inputs[0]) << ", "
       << __high2float(Inputs[0]) << "}";
  for (size_t i = 1; i < Inputs.size(); ++i) {
    cout << ", {" << __low2float(Inputs[i]) << ", " << __high2float(Inputs[i])
         << "}";
  }
  cout << ") = " << Result << " (expect " << Expect << ")";
  check(Result == Expect);
}

__global__ void hbeq2(bool *const Result, __half2 Input1, __half2 Input2) {
  *Result = __hbeq2(Input1, Input2);
}

void testHbeq2(bool *const Result, __half2 Input1, __half2 Input2) {
  hbeq2<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHbeq2Cases(const vector<pair<half2_pair, bool>> &TestCases) {
  bool *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHbeq2(Result, TestCase.first.first, TestCase.first.second);
    printResultBool("__hbeq2", {TestCase.first.first, TestCase.first.second},
                    TestCase.second, *Result);
  }
}

__global__ void hbequ2(bool *const Result, __half2 Input1, __half2 Input2) {
  *Result = __hbequ2(Input1, Input2);
}

void testHbequ2(bool *const Result, __half2 Input1, __half2 Input2) {
  hbequ2<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHbequ2Cases(const vector<pair<half2_pair, bool>> &TestCases) {
  bool *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHbequ2(Result, TestCase.first.first, TestCase.first.second);
    printResultBool("__hbequ2", {TestCase.first.first, TestCase.first.second},
                    TestCase.second, *Result);
  }
}

__global__ void hbge2(bool *const Result, __half2 Input1, __half2 Input2) {
  *Result = __hbge2(Input1, Input2);
}

void testHbge2(bool *const Result, __half2 Input1, __half2 Input2) {
  hbge2<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHbge2Cases(const vector<pair<half2_pair, bool>> &TestCases) {
  bool *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHbge2(Result, TestCase.first.first, TestCase.first.second);
    printResultBool("__hbge2", {TestCase.first.first, TestCase.first.second},
                    TestCase.second, *Result);
  }
}

__global__ void hbgeu2(bool *const Result, __half2 Input1, __half2 Input2) {
  *Result = __hbgeu2(Input1, Input2);
}

void testHbgeu2(bool *const Result, __half2 Input1, __half2 Input2) {
  hbgeu2<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHbgeu2Cases(const vector<pair<half2_pair, bool>> &TestCases) {
  bool *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHbgeu2(Result, TestCase.first.first, TestCase.first.second);
    printResultBool("__hbgeu2", {TestCase.first.first, TestCase.first.second},
                    TestCase.second, *Result);
  }
}

__global__ void hbgt2(bool *const Result, __half2 Input1, __half2 Input2) {
  *Result = __hbgt2(Input1, Input2);
}

void testHbgt2(bool *const Result, __half2 Input1, __half2 Input2) {
  hbgt2<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHbgt2Cases(const vector<pair<half2_pair, bool>> &TestCases) {
  bool *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHbgt2(Result, TestCase.first.first, TestCase.first.second);
    printResultBool("__hbgt2", {TestCase.first.first, TestCase.first.second},
                    TestCase.second, *Result);
  }
}

__global__ void hbgtu2(bool *const Result, __half2 Input1, __half2 Input2) {
  *Result = __hbgtu2(Input1, Input2);
}

void testHbgtu2(bool *const Result, __half2 Input1, __half2 Input2) {
  hbgtu2<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHbgtu2Cases(const vector<pair<half2_pair, bool>> &TestCases) {
  bool *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHbgtu2(Result, TestCase.first.first, TestCase.first.second);
    printResultBool("__hbgtu2", {TestCase.first.first, TestCase.first.second},
                    TestCase.second, *Result);
  }
}

__global__ void hble2(bool *const Result, __half2 Input1, __half2 Input2) {
  *Result = __hble2(Input1, Input2);
}

void testHble2(bool *const Result, __half2 Input1, __half2 Input2) {
  hble2<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHble2Cases(const vector<pair<half2_pair, bool>> &TestCases) {
  bool *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHble2(Result, TestCase.first.first, TestCase.first.second);
    printResultBool("__hble2", {TestCase.first.first, TestCase.first.second},
                    TestCase.second, *Result);
  }
}

__global__ void hbleu2(bool *const Result, __half2 Input1, __half2 Input2) {
  *Result = __hbleu2(Input1, Input2);
}

void testHbleu2(bool *const Result, __half2 Input1, __half2 Input2) {
  hbleu2<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHbleu2Cases(const vector<pair<half2_pair, bool>> &TestCases) {
  bool *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHbleu2(Result, TestCase.first.first, TestCase.first.second);
    printResultBool("__hbleu2", {TestCase.first.first, TestCase.first.second},
                    TestCase.second, *Result);
  }
}

__global__ void hblt2(bool *const Result, __half2 Input1, __half2 Input2) {
  *Result = __hblt2(Input1, Input2);
}

void testHblt2(bool *const Result, __half2 Input1, __half2 Input2) {
  hblt2<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHblt2Cases(const vector<pair<half2_pair, bool>> &TestCases) {
  bool *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHblt2(Result, TestCase.first.first, TestCase.first.second);
    printResultBool("__hblt2", {TestCase.first.first, TestCase.first.second},
                    TestCase.second, *Result);
  }
}

__global__ void hbltu2(bool *const Result, __half2 Input1, __half2 Input2) {
  *Result = __hbltu2(Input1, Input2);
}

void testHbltu2(bool *const Result, __half2 Input1, __half2 Input2) {
  hbltu2<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHbltu2Cases(const vector<pair<half2_pair, bool>> &TestCases) {
  bool *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHbltu2(Result, TestCase.first.first, TestCase.first.second);
    printResultBool("__hbltu2", {TestCase.first.first, TestCase.first.second},
                    TestCase.second, *Result);
  }
}

__global__ void hbne2(bool *const Result, __half2 Input1, __half2 Input2) {
  *Result = __hbne2(Input1, Input2);
}

void testHbne2(bool *const Result, __half2 Input1, __half2 Input2) {
  hbne2<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHbne2Cases(const vector<pair<half2_pair, bool>> &TestCases) {
  bool *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHbne2(Result, TestCase.first.first, TestCase.first.second);
    printResultBool("__hbne2", {TestCase.first.first, TestCase.first.second},
                    TestCase.second, *Result);
  }
}

__global__ void hbneu2(bool *const Result, __half2 Input1, __half2 Input2) {
  *Result = __hbneu2(Input1, Input2);
}

void testHbneu2(bool *const Result, __half2 Input1, __half2 Input2) {
  hbneu2<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHbneu2Cases(const vector<pair<half2_pair, bool>> &TestCases) {
  bool *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHbneu2(Result, TestCase.first.first, TestCase.first.second);
    printResultBool("__hbneu2", {TestCase.first.first, TestCase.first.second},
                    TestCase.second, *Result);
  }
}

__global__ void heq2(float *const Result, __half2 Input1, __half2 Input2) {
  auto ret = __heq2(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHeq2(float *const Result, __half2 Input1, __half2 Input2) {
  heq2<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHeq2Cases(const vector<pair<half2_pair, h2i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHeq2(Result, TestCase.first.first, TestCase.first.second);
    checkResult("__heq2", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hequ2(float *const Result, __half2 Input1, __half2 Input2) {
  auto ret = __hequ2(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHequ2(float *const Result, __half2 Input1, __half2 Input2) {
  hequ2<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHequ2Cases(const vector<pair<half2_pair, h2i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHequ2(Result, TestCase.first.first, TestCase.first.second);
    checkResult("__hequ2", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hge2(float *const Result, __half2 Input1, __half2 Input2) {
  auto ret = __hge2(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHge2(float *const Result, __half2 Input1, __half2 Input2) {
  hge2<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHge2Cases(const vector<pair<half2_pair, h2i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHge2(Result, TestCase.first.first, TestCase.first.second);
    checkResult("__hge2", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hgeu2(float *const Result, __half2 Input1, __half2 Input2) {
  auto ret = __hgeu2(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHgeu2(float *const Result, __half2 Input1, __half2 Input2) {
  hgeu2<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHgeu2Cases(const vector<pair<half2_pair, h2i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHgeu2(Result, TestCase.first.first, TestCase.first.second);
    checkResult("__hgeu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hgt2(float *const Result, __half2 Input1, __half2 Input2) {
  auto ret = __hgt2(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHgt2(float *const Result, __half2 Input1, __half2 Input2) {
  hgt2<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHgt2Cases(const vector<pair<half2_pair, h2i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHgt2(Result, TestCase.first.first, TestCase.first.second);
    checkResult("__hgt2", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hgtu2(float *const Result, __half2 Input1, __half2 Input2) {
  auto ret = __hgtu2(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHgtu2(float *const Result, __half2 Input1, __half2 Input2) {
  hgtu2<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHgtu2Cases(const vector<pair<half2_pair, h2i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHgtu2(Result, TestCase.first.first, TestCase.first.second);
    checkResult("__hgtu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hisnan2(float *const Result, __half2 Input1) {
  auto ret = __hisnan2(Input1);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHisnan2(float *const Result, __half2 Input1) {
  hisnan2<<<1, 1>>>(Result, Input1);
  cudaDeviceSynchronize();
}

void testHisnan2Cases(const vector<pair<half2, h2i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, 2 * sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHisnan2(Result, TestCase.first);
    checkResult("__hisnan2", {TestCase.first}, TestCase.second.first,
                {Result[0], Result[1]}, TestCase.second.second);
  }
}

__global__ void hle2(float *const Result, __half2 Input1, __half2 Input2) {
  auto ret = __hle2(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHle2(float *const Result, __half2 Input1, __half2 Input2) {
  hle2<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHle2Cases(const vector<pair<half2_pair, h2i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHle2(Result, TestCase.first.first, TestCase.first.second);
    checkResult("__hle2", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hleu2(float *const Result, __half2 Input1, __half2 Input2) {
  auto ret = __hleu2(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHleu2(float *const Result, __half2 Input1, __half2 Input2) {
  hleu2<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHleu2Cases(const vector<pair<half2_pair, h2i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHleu2(Result, TestCase.first.first, TestCase.first.second);
    checkResult("__hleu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hlt2(float *const Result, __half2 Input1, __half2 Input2) {
  auto ret = __hlt2(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHlt2(float *const Result, __half2 Input1, __half2 Input2) {
  hlt2<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHlt2Cases(const vector<pair<half2_pair, h2i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHlt2(Result, TestCase.first.first, TestCase.first.second);
    checkResult("__hlt2", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hltu2(float *const Result, __half2 Input1, __half2 Input2) {
  auto ret = __hltu2(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHltu2(float *const Result, __half2 Input1, __half2 Input2) {
  hltu2<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHltu2Cases(const vector<pair<half2_pair, h2i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHltu2(Result, TestCase.first.first, TestCase.first.second);
    checkResult("__hltu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hne2(float *const Result, __half2 Input1, __half2 Input2) {
  auto ret = __hne2(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHne2(float *const Result, __half2 Input1, __half2 Input2) {
  hne2<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHne2Cases(const vector<pair<half2_pair, h2i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHne2(Result, TestCase.first.first, TestCase.first.second);
    checkResult("__hne2", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

__global__ void hneu2(float *const Result, __half2 Input1, __half2 Input2) {
  auto ret = __hneu2(Input1, Input2);
  Result[0] = __low2float(ret);
  Result[1] = __high2float(ret);
}

void testHneu2(float *const Result, __half2 Input1, __half2 Input2) {
  hneu2<<<1, 1>>>(Result, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHneu2Cases(const vector<pair<half2_pair, h2i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    testHneu2(Result, TestCase.first.first, TestCase.first.second);
    checkResult("__hneu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, {Result[0], Result[1]},
                TestCase.second.second);
  }
}

int main() {
  testHadd2_satCases({
      {{{-0.3, -0.5}, {-0.4, -0.6}}, {{0, 0}, 37}},
      {{{0.3, 0.5}, {-0.4, 0.6}}, {{0, 1}, 15}},
      {{{0.3, 0.5}, {0.4, 0.2}}, {{0.7001953125, 0.7001953125}, 16}},
      {{{0.3, 0.5}, {0.4, 0.6}}, {{0.7001953125, 1}, 15}},
      {{{3, 5}, {4, 6}}, {{1, 1}, 15}},
  });
  testHfma2_satCases({
      {{{-0.3, -0.5}, {-0.4, -0.6}, {-0.2, -0.7}}, {{0, 0}, 37}},
      {{{0.3, 0.5}, {-0.4, 0.6}, {-0.1, 0.2}}, {{0, 0.5}, 16}},
      {{{0.3, 0.5}, {0.4, 0.2}, {0.1, 0.1}},
       {{0.219970703125, 0.199951171875}, 16}},
      {{{0.3, 0.5}, {0.4, 0.6}, {0, 0.3}},
       {{0.1199951171875, 0.60009765625}, 16}},
      {{{3, 5}, {4, 6}, {5, 8}}, {{1, 1}, 15}},
  });
  testHmul2_satCases({
      {{{-0.3, -5}, {0.4, 6}}, {{0, 0}, 37}},
      {{{0.3, 5}, {-4, 0.6}}, {{0, 1}, 15}},
      {{{0.3, 0.5}, {0.4, 0.2}}, {{0.1199951171875, 0.0999755859375}, 17}},
      {{{0.3, 0.5}, {0.4, 0.6}}, {{0.1199951171875, 0.300048828125}, 16}},
      {{{3, 5}, {4, 6}}, {{1, 1}, 15}},
  });
  testHsub2_satCases({
      {{{0, 0}, {-0.4, -0.6}}, {{0.39990234375, 0.60009765625}, 16}},
      {{{0, 1}, {0.4, 0.6}}, {{0, 0.39990234375}, 16}},
      {{{0.7, 0.7}, {0.4, 0.2}}, {{0.30029296875, 0.5}, 16}},
      {{{0.7, 2}, {0.4, 0.6}}, {{0.30029296875, 1}, 15}},
      {{{1, 1}, {4, 6}}, {{0, 0}, 37}},
  });
  testHbeq2Cases({
      {{{0, 0}, {-0.4, -0.6}}, false},
      {{{0.7, 0.7}, {0.4, 0.7}}, false},
      {{{0.7, 2}, {0.7, 2}}, true},
      {{{1, 1}, {4, 6}}, false},
      {{{NAN, 1}, {1, 1}}, false},
  });
  testHbequ2Cases({
      {{{0, 0}, {-0.4, -0.6}}, false},
      {{{0.7, 0.7}, {0.4, 0.7}}, false},
      {{{0.7, 2}, {0.7, 2}}, true},
      {{{1, 1}, {4, 6}}, false},
      {{{NAN, 1}, {1, 1}}, true},
  });
  testHbge2Cases({
      {{{0, 0}, {-0.4, -0.6}}, true},
      {{{0.7, 0.7}, {0.4, 0.7}}, true},
      {{{0.7, 2}, {0.7, 2}}, true},
      {{{1, 1}, {4, 6}}, false},
      {{{NAN, 1}, {1, 1}}, false},
  });
  testHbgeu2Cases({
      {{{0, 0}, {-0.4, -0.6}}, true},
      {{{0.7, 0.7}, {0.4, 0.7}}, true},
      {{{0.7, 2}, {0.7, 2}}, true},
      {{{1, 1}, {4, 6}}, false},
      {{{NAN, 1}, {1, 1}}, true},
  });
  testHbgt2Cases({
      {{{0, 0}, {-0.4, -0.6}}, true},
      {{{0.7, 0.7}, {0.4, 0.7}}, false},
      {{{0.7, 2}, {0.7, 2}}, false},
      {{{1, 1}, {4, 6}}, false},
      {{{NAN, 2}, {1, 1}}, false},
  });
  testHbgtu2Cases({
      {{{0, 0}, {-0.4, -0.6}}, true},
      {{{0.7, 0.7}, {0.4, 0.7}}, false},
      {{{0.7, 2}, {0.7, 2}}, false},
      {{{1, 1}, {4, 6}}, false},
      {{{NAN, 2}, {1, 1}}, true},
  });
  testHble2Cases({
      {{{0, 0}, {-0.4, -0.6}}, false},
      {{{0.7, 0.7}, {0.4, 0.7}}, false},
      {{{0.7, 2}, {0.7, 2}}, true},
      {{{1, 1}, {4, 6}}, true},
      {{{NAN, 1}, {1, 1}}, false},
  });
  testHbleu2Cases({
      {{{0, 0}, {-0.4, -0.6}}, false},
      {{{0.7, 0.7}, {0.4, 0.7}}, false},
      {{{0.7, 2}, {0.7, 2}}, true},
      {{{1, 1}, {4, 6}}, true},
      {{{NAN, 1}, {1, 1}}, true},
  });
  testHblt2Cases({
      {{{0, 0}, {-0.4, -0.6}}, false},
      {{{0.7, 0.7}, {0.4, 0.7}}, false},
      {{{0.7, 2}, {0.7, 2}}, false},
      {{{1, 1}, {4, 6}}, true},
      {{{NAN, 1}, {1, 2}}, false},
  });
  testHbltu2Cases({
      {{{0, 0}, {-0.4, -0.6}}, false},
      {{{0.7, 0.7}, {0.4, 0.7}}, false},
      {{{0.7, 2}, {0.7, 2}}, false},
      {{{1, 1}, {4, 6}}, true},
      {{{NAN, 1}, {1, 2}}, true},
  });
  testHbne2Cases({
      {{{0, 0}, {-0.4, -0.6}}, true},
      {{{0.7, 0.7}, {0.4, 0.7}}, false},
      {{{0.7, 2}, {0.7, 2}}, false},
      {{{1, 1}, {4, 6}}, true},
      {{{NAN, 1}, {1, 2}}, false},
  });
  testHbneu2Cases({
      {{{0, 0}, {-0.4, -0.6}}, true},
      {{{0.7, 0.7}, {0.4, 0.7}}, false},
      {{{0.7, 2}, {0.7, 2}}, false},
      {{{1, 1}, {4, 6}}, true},
      {{{NAN, 1}, {1, 2}}, true},
  });
  testHeq2Cases({
      {{{0, 0}, {-0.4, -0.6}}, {{0, 0}, 37}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {{0, 1}, 15}},
      {{{0.7, 2}, {0.7, 2}}, {{1, 1}, 15}},
      {{{1, 1}, {4, 6}}, {{0, 0}, 37}},
      {{{NAN, 1}, {1, 1}}, {{0, 1}, 15}},
  });
  testHequ2Cases({
      {{{0, 0}, {-0.4, -0.6}}, {{0, 0}, 37}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {{0, 1}, 15}},
      {{{0.7, 2}, {0.7, 2}}, {{1, 1}, 15}},
      {{{1, 1}, {4, 6}}, {{0, 0}, 37}},
      {{{NAN, 1}, {1, 1}}, {{1, 1}, 15}},
  });
  testHge2Cases({
      {{{0, 0}, {-0.4, -0.6}}, {{1, 1}, 15}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {{1, 1}, 15}},
      {{{0.7, 2}, {0.7, 2}}, {{1, 1}, 15}},
      {{{1, 1}, {4, 6}}, {{0, 0}, 37}},
      {{{NAN, 1}, {1, 1}}, {{0, 1}, 15}},
  });
  testHgeu2Cases({
      {{{0, 0}, {-0.4, -0.6}}, {{1, 1}, 15}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {{1, 1}, 15}},
      {{{0.7, 2}, {0.7, 2}}, {{1, 1}, 15}},
      {{{1, 1}, {4, 6}}, {{0, 0}, 37}},
      {{{NAN, 1}, {1, 1}}, {{1, 1}, 15}},
  });
  testHgt2Cases({
      {{{0, 0}, {-0.4, -0.6}}, {{1, 1}, 15}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {{1, 0}, 15}},
      {{{0.7, 2}, {0.7, 2}}, {{0, 0}, 37}},
      {{{1, 1}, {4, 6}}, {{0, 0}, 37}},
      {{{NAN, 1}, {1, 1}}, {{0, 0}, 37}},
  });
  testHgtu2Cases({
      {{{0, 0}, {-0.4, -0.6}}, {{1, 1}, 15}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {{1, 0}, 15}},
      {{{0.7, 2}, {0.7, 2}}, {{0, 0}, 37}},
      {{{1, 1}, {4, 6}}, {{0, 0}, 37}},
      {{{NAN, 1}, {1, 1}}, {{1, 0}, 15}},
  });
  testHisnan2Cases({
      {{0, 0}, {{0, 0}, 37}},
      {{0.7, 2}, {{0, 0}, 37}},
      {{NAN, 1}, {{1, 0}, 15}},
      {{NAN, NAN}, {{1, 1}, 15}},
      {{0, NAN}, {{0, 1}, 15}},
  });
  testHle2Cases({
      {{{0, 0}, {-0.4, -0.6}}, {{0, 0}, 37}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {{0, 1}, 15}},
      {{{0.7, 2}, {0.7, 2}}, {{1, 1}, 15}},
      {{{1, 1}, {4, 6}}, {{1, 1}, 15}},
      {{{NAN, 1}, {1, 1}}, {{0, 1}, 15}},
  });
  testHleu2Cases({
      {{{0, 0}, {-0.4, -0.6}}, {{0, 0}, 37}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {{0, 1}, 15}},
      {{{0.7, 2}, {0.7, 2}}, {{1, 1}, 15}},
      {{{1, 1}, {4, 6}}, {{1, 1}, 15}},
      {{{NAN, 1}, {1, 1}}, {{1, 1}, 15}},
  });
  testHlt2Cases({
      {{{0, 0}, {-0.4, -0.6}}, {{0, 0}, 37}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {{0, 0}, 37}},
      {{{0.7, 2}, {0.7, 2}}, {{0, 0}, 37}},
      {{{1, 1}, {4, 6}}, {{1, 1}, 15}},
      {{{NAN, 1}, {1, 1}}, {{0, 0}, 37}},
  });
  testHltu2Cases({
      {{{0, 0}, {-0.4, -0.6}}, {{0, 0}, 37}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {{0, 0}, 37}},
      {{{0.7, 2}, {0.7, 2}}, {{0, 0}, 37}},
      {{{1, 1}, {4, 6}}, {{1, 1}, 15}},
      {{{NAN, 1}, {1, 1}}, {{1, 0}, 15}},
  });
  testHne2Cases({
      {{{0, 0}, {-0.4, -0.6}}, {{1, 1}, 15}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {{1, 0}, 15}},
      {{{0.7, 2}, {0.7, 2}}, {{0, 0}, 37}},
      {{{1, 1}, {4, 6}}, {{1, 1}, 15}},
      {{{NAN, 1}, {1, 1}}, {{0, 0}, 37}},
  });
  testHneu2Cases({
      {{{0, 0}, {-0.4, -0.6}}, {{1, 1}, 15}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {{1, 0}, 15}},
      {{{0.7, 2}, {0.7, 2}}, {{0, 0}, 37}},
      {{{1, 1}, {4, 6}}, {{1, 1}, 15}},
      {{{NAN, 1}, {1, 1}}, {{1, 0}, 15}},
  });
  cout << "passed " << passed << "/" << passed + failed << " cases!" << endl;
  if (failed) {
    cout << "failed!" << endl;
  }
  return failed;
}
