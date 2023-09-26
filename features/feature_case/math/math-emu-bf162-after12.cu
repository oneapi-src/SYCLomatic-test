// ===---------- math-emu-bf162-after12.cu --------- *- CUDA -* -----------===//
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
                 const unsigned &Expect, const unsigned &Result) {
  cout << FuncName << "({" << Inputs[0].x << ", " << Inputs[0].y << "}";
  for (size_t i = 1; i < Inputs.size(); ++i) {
    cout << ", {" << Inputs[i].x << ", " << Inputs[i].y << "}";
  }
  cout << ") = " << Result << " (expect " << Expect << ")";
  check(Result == Expect);
}

void checkResult(const string &FuncName, const vector<__nv_bfloat162> &Inputs,
                 const unsigned &Expect, const unsigned &Result) {
  vector<float2> FInputs;
  for (const auto &it : Inputs) {
    FInputs.push_back({__bfloat162float(it.x), __bfloat162float(it.y)});
  }
  checkResult(FuncName, FInputs, Expect, Result);
}

// Bfloat162 Comparison Functions

__global__ void heq2_mask(unsigned *const Result, __nv_bfloat162 Input1,
                          __nv_bfloat162 Input2) {
  *Result = __heq2_mask(Input1, Input2);
}

void testHeq2_maskCases(const vector<pair<bf162_pair, unsigned>> &TestCases) {
  unsigned *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    heq2_mask<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__heq2_mask", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *Result);
  }
}

__global__ void hequ2_mask(unsigned *const Result, __nv_bfloat162 Input1,
                           __nv_bfloat162 Input2) {
  *Result = __hequ2_mask(Input1, Input2);
}

void testHequ2_maskCases(const vector<pair<bf162_pair, unsigned>> &TestCases) {
  unsigned *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hequ2_mask<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hequ2_mask", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *Result);
  }
}

__global__ void hge2_mask(unsigned *const Result, __nv_bfloat162 Input1,
                          __nv_bfloat162 Input2) {
  *Result = __hge2_mask(Input1, Input2);
}

void testHge2_maskCases(const vector<pair<bf162_pair, unsigned>> &TestCases) {
  unsigned *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hge2_mask<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hge2_mask", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *Result);
  }
}

__global__ void hgeu2_mask(unsigned *const Result, __nv_bfloat162 Input1,
                           __nv_bfloat162 Input2) {
  *Result = __hgeu2_mask(Input1, Input2);
}

void testHgeu2_maskCases(const vector<pair<bf162_pair, unsigned>> &TestCases) {
  unsigned *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hgeu2_mask<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hgeu2_mask", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *Result);
  }
}

__global__ void hgt2_mask(unsigned *const Result, __nv_bfloat162 Input1,
                          __nv_bfloat162 Input2) {
  *Result = __hgt2_mask(Input1, Input2);
}

void testHgt2_maskCases(const vector<pair<bf162_pair, unsigned>> &TestCases) {
  unsigned *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hgt2_mask<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hgt2_mask", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *Result);
  }
}

__global__ void hgtu2_mask(unsigned *const Result, __nv_bfloat162 Input1,
                           __nv_bfloat162 Input2) {
  *Result = __hgtu2_mask(Input1, Input2);
}

void testHgtu2_maskCases(const vector<pair<bf162_pair, unsigned>> &TestCases) {
  unsigned *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hgtu2_mask<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hgtu2_mask", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *Result);
  }
}

__global__ void hle2_mask(unsigned *const Result, __nv_bfloat162 Input1,
                          __nv_bfloat162 Input2) {
  *Result = __hle2_mask(Input1, Input2);
}

void testHle2_maskCases(const vector<pair<bf162_pair, unsigned>> &TestCases) {
  unsigned *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hle2_mask<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hle2_mask", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *Result);
  }
}

__global__ void hleu2_mask(unsigned *const Result, __nv_bfloat162 Input1,
                           __nv_bfloat162 Input2) {
  *Result = __hleu2_mask(Input1, Input2);
}

void testHleu2_maskCases(const vector<pair<bf162_pair, unsigned>> &TestCases) {
  unsigned *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hleu2_mask<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hleu2_mask", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *Result);
  }
}

__global__ void hlt2_mask(unsigned *const Result, __nv_bfloat162 Input1,
                          __nv_bfloat162 Input2) {
  *Result = __hlt2_mask(Input1, Input2);
}

void testHlt2_maskCases(const vector<pair<bf162_pair, unsigned>> &TestCases) {
  unsigned *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hlt2_mask<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hlt2_mask", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *Result);
  }
}

__global__ void hltu2_mask(unsigned *const Result, __nv_bfloat162 Input1,
                           __nv_bfloat162 Input2) {
  *Result = __hltu2_mask(Input1, Input2);
}

void testHltu2_maskCases(const vector<pair<bf162_pair, unsigned>> &TestCases) {
  unsigned *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hltu2_mask<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hltu2_mask", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *Result);
  }
}

__global__ void hne2_mask(unsigned *const Result, __nv_bfloat162 Input1,
                          __nv_bfloat162 Input2) {
  *Result = __hne2_mask(Input1, Input2);
}

void testHne2_maskCases(const vector<pair<bf162_pair, unsigned>> &TestCases) {
  unsigned *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hne2_mask<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hne2_mask", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *Result);
  }
}

__global__ void hneu2_mask(unsigned *const Result, __nv_bfloat162 Input1,
                           __nv_bfloat162 Input2) {
  *Result = __hneu2_mask(Input1, Input2);
}

void testHneu2_maskCases(const vector<pair<bf162_pair, unsigned>> &TestCases) {
  unsigned *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    hneu2_mask<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__hneu2_mask", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *Result);
  }
}

int main() {
  testHeq2_maskCases({
      {{{0, 0}, {-0.4, -0.6}}, 0},
      {{{0.7, 0.7}, {0.4, 0.7}}, 4294901760},
      {{{0.7, 2}, {0.7, 2}}, 4294967295},
      {{{1, 1}, {4, 6}}, 0},
      {{{NAN, 1}, {1, 1}}, 4294901760},
  });
  testHequ2_maskCases({
      {{{0, 0}, {-0.4, -0.6}}, 0},
      {{{0.7, 0.7}, {0.4, 0.7}}, 4294901760},
      {{{0.7, 2}, {0.7, 2}}, 4294967295},
      {{{1, 1}, {4, 6}}, 0},
      {{{NAN, 1}, {1, 1}}, 4294967295},
  });
  testHge2_maskCases({
      {{{0, 0}, {-0.4, -0.6}}, 4294967295},
      {{{0.7, 0.7}, {0.4, 0.7}}, 4294967295},
      {{{0.7, 2}, {0.7, 2}}, 4294967295},
      {{{1, 1}, {4, 6}}, 0},
      {{{NAN, 1}, {1, 1}}, 4294901760},
  });
  testHgeu2_maskCases({
      {{{0, 0}, {-0.4, -0.6}}, 4294967295},
      {{{0.7, 0.7}, {0.4, 0.7}}, 4294967295},
      {{{0.7, 2}, {0.7, 2}}, 4294967295},
      {{{1, 1}, {4, 6}}, 0},
      {{{NAN, 1}, {1, 1}}, 4294967295},
  });
  testHgt2_maskCases({
      {{{0, 0}, {-0.4, -0.6}}, 4294967295},
      {{{0.7, 0.7}, {0.4, 0.7}}, 65535},
      {{{0.7, 2}, {0.7, 2}}, 0},
      {{{1, 1}, {4, 6}}, 0},
      {{{NAN, 1}, {1, 1}}, 0},
  });
  testHgtu2_maskCases({
      {{{0, 0}, {-0.4, -0.6}}, 4294967295},
      {{{0.7, 0.7}, {0.4, 0.7}}, 65535},
      {{{0.7, 2}, {0.7, 2}}, 0},
      {{{1, 1}, {4, 6}}, 0},
      {{{NAN, 1}, {1, 1}}, 65535},
  });
  testHle2_maskCases({
      {{{0, 0}, {-0.4, -0.6}}, 0},
      {{{0.7, 0.7}, {0.4, 0.7}}, 4294901760},
      {{{0.7, 2}, {0.7, 2}}, 4294967295},
      {{{1, 1}, {4, 6}}, 4294967295},
      {{{NAN, 1}, {1, 1}}, 4294901760},
  });
  testHleu2_maskCases({
      {{{0, 0}, {-0.4, -0.6}}, 0},
      {{{0.7, 0.7}, {0.4, 0.7}}, 4294901760},
      {{{0.7, 2}, {0.7, 2}}, 4294967295},
      {{{1, 1}, {4, 6}}, 4294967295},
      {{{NAN, 1}, {1, 1}}, 4294967295},
  });
  testHlt2_maskCases({
      {{{0, 0}, {-0.4, -0.6}}, 0},
      {{{0.7, 0.7}, {0.4, 0.7}}, 0},
      {{{0.7, 2}, {0.7, 2}}, 0},
      {{{1, 1}, {4, 6}}, 4294967295},
      {{{NAN, 1}, {1, 1}}, 0},
  });
  testHltu2_maskCases({
      {{{0, 0}, {-0.4, -0.6}}, 0},
      {{{0.7, 0.7}, {0.4, 0.7}}, 0},
      {{{0.7, 2}, {0.7, 2}}, 0},
      {{{1, 1}, {4, 6}}, 4294967295},
      {{{NAN, 1}, {1, 1}}, 65535},
  });
  testHne2_maskCases({
      {{{0, 0}, {-0.4, -0.6}}, 4294967295},
      {{{0.7, 0.7}, {0.4, 0.7}}, 65535},
      {{{0.7, 2}, {0.7, 2}}, 0},
      {{{1, 1}, {4, 6}}, 4294967295},
      {{{NAN, 1}, {1, 1}}, 0},
  });
  testHneu2_maskCases({
      {{{0, 0}, {-0.4, -0.6}}, 4294967295},
      {{{0.7, 0.7}, {0.4, 0.7}}, 65535},
      {{{0.7, 2}, {0.7, 2}}, 0},
      {{{1, 1}, {4, 6}}, 4294967295},
      {{{NAN, 1}, {1, 1}}, 65535},
  });
  cout << "passed " << passed << "/" << passed + failed << " cases!" << endl;
  if (failed) {
    cout << "failed!" << endl;
  }
  return failed;
}
