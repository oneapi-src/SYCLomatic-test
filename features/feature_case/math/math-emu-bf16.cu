//====---------- math-emu-bf16.cu ----------- *- CUDA -* ------------------===//
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

__global__ void hadd(float *const Result, __nv_bfloat16 Input1,
                     __nv_bfloat16 Input2) {
  *Result = __hadd(Input1, Input2);
}

void testHaddCases(
    const vector<pair<pair<__nv_bfloat16, __nv_bfloat16>, bf16i_pair>>
        &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hadd<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hadd", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void hadd_rn(float *const Result, __nv_bfloat16 Input1,
                        __nv_bfloat16 Input2) {
  *Result = __hadd_rn(Input1, Input2);
}

void testHadd_rnCases(
    const vector<pair<pair<__nv_bfloat16, __nv_bfloat16>, bf16i_pair>>
        &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hadd_rn<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hadd_rn", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void hadd_sat(float *const Result, __nv_bfloat16 Input1,
                         __nv_bfloat16 Input2) {
  *Result = __hadd_sat(Input1, Input2);
}

void testHadd_satCases(
    const vector<pair<pair<__nv_bfloat16, __nv_bfloat16>, bf16i_pair>>
        &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hadd_sat<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hadd_sat", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void hdiv(float *const Result, __nv_bfloat16 Input1,
                     __nv_bfloat16 Input2) {
  *Result = __hdiv(Input1, Input2);
}

void testHdivCases(
    const vector<pair<pair<__nv_bfloat16, __nv_bfloat16>, bf16i_pair>>
        &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hdiv<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hdiv", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
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

__global__ void hmul(float *const Result, __nv_bfloat16 Input1,
                     __nv_bfloat16 Input2) {
  *Result = __hmul(Input1, Input2);
}

void testHmulCases(
    const vector<pair<pair<__nv_bfloat16, __nv_bfloat16>, bf16i_pair>>
        &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hmul<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hmul", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void hmul_rn(float *const Result, __nv_bfloat16 Input1,
                        __nv_bfloat16 Input2) {
  *Result = __hmul_rn(Input1, Input2);
}

void testHmul_rnCases(
    const vector<pair<pair<__nv_bfloat16, __nv_bfloat16>, bf16i_pair>>
        &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hmul_rn<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hmul_rn", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void hmul_sat(float *const Result, __nv_bfloat16 Input1,
                         __nv_bfloat16 Input2) {
  *Result = __hmul_sat(Input1, Input2);
}

void testHmul_satCases(
    const vector<pair<pair<__nv_bfloat16, __nv_bfloat16>, bf16i_pair>>
        &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hmul_sat<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hmul_sat", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void hneg(float *const Result, __nv_bfloat16 Input1) {
  *Result = __hneg(Input1);
}

void testHnegCases(const vector<pair<__nv_bfloat16, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hneg<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__hneg", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void hsub(float *const Result, __nv_bfloat16 Input1,
                     __nv_bfloat16 Input2) {
  *Result = __hsub(Input1, Input2);
}

void testHsubCases(
    const vector<pair<pair<__nv_bfloat16, __nv_bfloat16>, bf16i_pair>>
        &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hsub<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hsub", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void hsub_rn(float *const Result, __nv_bfloat16 Input1,
                        __nv_bfloat16 Input2) {
  *Result = __hsub_rn(Input1, Input2);
}

void testHsub_rnCases(
    const vector<pair<pair<__nv_bfloat16, __nv_bfloat16>, bf16i_pair>>
        &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hsub_rn<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hsub_rn", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void hsub_sat(float *const Result, __nv_bfloat16 Input1,
                         __nv_bfloat16 Input2) {
  *Result = __hsub_sat(Input1, Input2);
}

void testHsub_satCases(
    const vector<pair<pair<__nv_bfloat16, __nv_bfloat16>, bf16i_pair>>
        &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hsub_sat<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hsub_sat", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
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
  testHaddCases({
      {{-0.3, -0.4}, {-0.703125, 16}},
      {{0.3, -0.4}, {-0.099609375, 17}},
      {{0.3, 0.4}, {0.703125, 16}},
      {{0.3, 0.8}, {1.1015625, 15}},
      {{3, 4}, {7, 15}},
  });
  testHadd_rnCases({
      {{-0.3, -0.4}, {-0.703125, 16}},
      {{0.3, -0.4}, {-0.099609375, 17}},
      {{0.3, 0.4}, {0.703125, 16}},
      {{0.3, 0.8}, {1.1015625, 15}},
      {{3, 4}, {7, 15}},
  });
  testHadd_satCases({
      {{-0.3, -0.4}, {0, 37}},
      {{0.3, -0.4}, {0, 37}},
      {{0.3, 0.4}, {0.703125, 16}},
      {{0.3, 0.8}, {1, 15}},
      {{3, 4}, {1, 15}},
  });
  testHdivCases({
      {{-0.3, -0.4}, {0.75, 16}},
      {{0.3, -0.4}, {-0.75, 16}},
      {{0.3, 0.4}, {0.75, 16}},
      {{0.3, 0.8}, {0.375, 16}},
      {{3, 4}, {0.75, 16}},
  });
  testHfmaCases({
      {{-0.3, -0.4, -0.2}, {-0.07958984375, 17}},
      {{0.3, -0.4, -0.1}, {-0.220703125, 16}},
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
  testHmulCases({
      {{-0.3, -0.4}, {0.12060546875, 17}},
      {{0.3, -0.4}, {-0.12060546875, 17}},
      {{0.3, 0.4}, {0.12060546875, 17}},
      {{0.3, 0.8}, {0.2412109375, 16}},
      {{3, 4}, {12, 15}},
  });
  testHmul_rnCases({
      {{-0.3, -0.4}, {0.12060546875, 17}},
      {{0.3, -0.4}, {-0.12060546875, 17}},
      {{0.3, 0.4}, {0.12060546875, 17}},
      {{0.3, 0.8}, {0.2412109375, 16}},
      {{3, 4}, {12, 15}},
  });
  testHmul_satCases({
      {{-0.3, -0.4}, {0.12060546875, 17}},
      {{0.3, -0.4}, {0, 37}},
      {{0.3, 0.4}, {0.12060546875, 17}},
      {{0.3, 0.8}, {0.2412109375, 16}},
      {{3, 4}, {1, 15}},
  });
  testHnegCases({
      {{-0.3}, {0.30078125, 16}},
      {{0.3}, {-0.30078125, 16}},
      {{0.5}, {-0.5, 16}},
      {{0.4}, {-0.400390625, 16}},
      {{6}, {-6, 15}},
  });
  testHsubCases({
      {{-0.3, -0.4}, {0.099609375, 17}},
      {{0.3, -0.4}, {0.703125, 16}},
      {{0.3, 0.4}, {-0.099609375, 17}},
      {{0.3, -0.8}, {1.1015625, 15}},
      {{3, 4}, {-1, 15}},
  });
  testHsub_rnCases({
      {{-0.3, -0.4}, {0.099609375, 17}},
      {{0.3, -0.4}, {0.703125, 16}},
      {{0.3, 0.4}, {-0.099609375, 17}},
      {{0.3, -0.8}, {1.1015625, 15}},
      {{3, 4}, {-1, 15}},
  });
  testHsub_satCases({
      {{-0.3, -0.4}, {0.099609375, 17}},
      {{0.3, -0.4}, {0.703125, 16}},
      {{0.3, 0.4}, {0, 37}},
      {{0.3, -0.8}, {1, 15}},
      {{3, 4}, {0, 37}},
  });
  cout << "passed " << passed << "/" << passed + failed << " cases!" << endl;
  if (failed) {
    cout << "failed!" << endl;
  }
  return failed;
}
