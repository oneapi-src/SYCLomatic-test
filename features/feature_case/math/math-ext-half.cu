// ====------ math-ext-half.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <iostream>
#include <vector>

#include "cuda_fp16.h"

using namespace std;

int ret = 0;

void check(bool IsPassed) {
  if (IsPassed) {
    std::cout << " ---- passed" << std::endl;
  } else {
    std::cout << " ---- failed" << std::endl;
    ret++;
  }
}

void printResultHalf(const string &FuncName, const vector<__half> &Inputs,
                     const __half &Expect, const float &DeviceResult) {
  const float Precision = 0.001;
  std::cout << FuncName << "(" << __half2float(Inputs[0]);
  for (size_t i = 1; i < Inputs.size(); ++i) {
    std::cout << ", " << __half2float(Inputs[i]);
  }
  std::cout << ") = " << __half2float(DeviceResult) << " (expect "
            << __half2float(Expect) - Precision << " ~ "
            << __half2float(Expect) + Precision << ")";
  check(__half2float(DeviceResult) > __half2float(Expect) - Precision &&
        __half2float(DeviceResult) < __half2float(Expect) + Precision);
}

__global__ void hadd_sat(float *const DeviceResult, __half Input1,
                         __half Input2) {
  *DeviceResult = __hadd_sat(Input1, Input2);
}

void testHadd_sat(float *const DeviceResult, __half Input1, __half Input2) {
  hadd_sat<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHadd_satCases(
    const vector<pair<pair<__half, __half>, __half>> &TestCases) {
  float *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testHadd_sat(DeviceResult, TestCase.first.first, TestCase.first.second);
    printResultHalf("__hadd_sat", {TestCase.first.first, TestCase.first.second},
                    TestCase.second, *DeviceResult);
  }
}

__global__ void hfma_sat(float *const DeviceResult, __half Input1,
                         __half Input2, __half Input3) {
  *DeviceResult = __hfma_sat(Input1, Input2, Input3);
}

void testHfma_sat(float *const DeviceResult, __half Input1, __half Input2,
                  __half Input3) {
  hfma_sat<<<1, 1>>>(DeviceResult, Input1, Input2, Input3);
  cudaDeviceSynchronize();
}

void testHfma_satCases(const vector<pair<vector<__half>, __half>> &TestCases) {
  float *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testHfma_sat(DeviceResult, TestCase.first[0], TestCase.first[1],
                 TestCase.first[2]);
    printResultHalf("__hfma_sat", TestCase.first, TestCase.second,
                    *DeviceResult);
  }
}

__global__ void hmul_sat(float *const DeviceResult, __half Input1,
                         __half Input2) {
  *DeviceResult = __hmul_sat(Input1, Input2);
}

void testHmul_sat(float *const DeviceResult, __half Input1, __half Input2) {
  hmul_sat<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHmul_satCases(
    const vector<pair<pair<__half, __half>, __half>> &TestCases) {
  float *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testHmul_sat(DeviceResult, TestCase.first.first, TestCase.first.second);
    printResultHalf("__hmul_sat", {TestCase.first.first, TestCase.first.second},
                    TestCase.second, *DeviceResult);
  }
}

__global__ void hsub_sat(float *const DeviceResult, __half Input1,
                         __half Input2) {
  *DeviceResult = __hsub_sat(Input1, Input2);
}

void testHsub_sat(float *const DeviceResult, __half Input1, __half Input2) {
  hsub_sat<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHsub_satCases(
    const vector<pair<pair<__half, __half>, __half>> &TestCases) {
  float *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testHsub_sat(DeviceResult, TestCase.first.first, TestCase.first.second);
    printResultHalf("__hsub_sat", {TestCase.first.first, TestCase.first.second},
                    TestCase.second, *DeviceResult);
  }
}

void printResultBool(const string &FuncName, const vector<__half> &Inputs,
                     const bool &Expect, const bool &DeviceResult) {
  std::cout << FuncName << "(" << __half2float(Inputs[0]);
  for (size_t i = 1; i < Inputs.size(); ++i) {
    std::cout << ", " << __half2float(Inputs[i]);
  }
  std::cout << ") = " << DeviceResult << " (expect " << Expect << ")";
  check(DeviceResult == Expect);
}

__global__ void hequ(bool *const DeviceResult, __half Input1, __half Input2) {
  *DeviceResult = __hequ(Input1, Input2);
}

void testHequ(bool *const DeviceResult, __half Input1, __half Input2) {
  hequ<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHequCases(const vector<pair<pair<__half, __half>, bool>> &TestCases) {
  bool *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testHequ(DeviceResult, TestCase.first.first, TestCase.first.second);
    printResultBool("__hequ", {TestCase.first.first, TestCase.first.second},
                    TestCase.second, *DeviceResult);
  }
}

__global__ void hgeu(bool *const DeviceResult, __half Input1, __half Input2) {
  *DeviceResult = __hgeu(Input1, Input2);
}

void testHgeu(bool *const DeviceResult, __half Input1, __half Input2) {
  hgeu<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHgeuCases(const vector<pair<pair<__half, __half>, bool>> &TestCases) {
  bool *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testHgeu(DeviceResult, TestCase.first.first, TestCase.first.second);
    printResultBool("__hgeu", {TestCase.first.first, TestCase.first.second},
                    TestCase.second, *DeviceResult);
  }
}

__global__ void hgtu(bool *const DeviceResult, __half Input1, __half Input2) {
  *DeviceResult = __hgtu(Input1, Input2);
}

void testHgtu(bool *const DeviceResult, __half Input1, __half Input2) {
  hgtu<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHgtuCases(const vector<pair<pair<__half, __half>, bool>> &TestCases) {
  bool *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testHgtu(DeviceResult, TestCase.first.first, TestCase.first.second);
    printResultBool("__hgtu", {TestCase.first.first, TestCase.first.second},
                    TestCase.second, *DeviceResult);
  }
}

__global__ void hleu(bool *const DeviceResult, __half Input1, __half Input2) {
  *DeviceResult = __hleu(Input1, Input2);
}

void testHleu(bool *const DeviceResult, __half Input1, __half Input2) {
  hleu<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHleuCases(const vector<pair<pair<__half, __half>, bool>> &TestCases) {
  bool *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testHleu(DeviceResult, TestCase.first.first, TestCase.first.second);
    printResultBool("__hleu", {TestCase.first.first, TestCase.first.second},
                    TestCase.second, *DeviceResult);
  }
}

__global__ void hltu(bool *const DeviceResult, __half Input1, __half Input2) {
  *DeviceResult = __hltu(Input1, Input2);
}

void testHltu(bool *const DeviceResult, __half Input1, __half Input2) {
  hltu<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHltuCases(const vector<pair<pair<__half, __half>, bool>> &TestCases) {
  bool *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testHltu(DeviceResult, TestCase.first.first, TestCase.first.second);
    printResultBool("__hltu", {TestCase.first.first, TestCase.first.second},
                    TestCase.second, *DeviceResult);
  }
}

__global__ void hneu(bool *const DeviceResult, __half Input1, __half Input2) {
  *DeviceResult = __hneu(Input1, Input2);
}

void testHneu(bool *const DeviceResult, __half Input1, __half Input2) {
  hneu<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHneuCases(const vector<pair<pair<__half, __half>, bool>> &TestCases) {
  bool *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testHneu(DeviceResult, TestCase.first.first, TestCase.first.second);
    printResultBool("__hneu", {TestCase.first.first, TestCase.first.second},
                    TestCase.second, *DeviceResult);
  }
}

int main() {
  testHadd_satCases({
      {{-0.3, -0.4}, 0},
      {{0.3, -0.4}, 0},
      {{0.3, 0.4}, 0.7},
      {{0.3, 0.8}, 1},
      {{3, 4}, 1},
  });
  testHfma_satCases({
      {{-0.3, -0.4, -0.2}, 0},
      {{0.3, -0.4, -0.1}, 0},
      {{0.3, 0.4, 0.1}, 0.22},
      {{0.3, 0.4, 0}, 0.12},
      {{3, 4, 5}, 1},
  });
  testHmul_satCases({
      {{-0.3, 0.4}, 0},
      {{0.3, -4}, 0},
      {{0.3, 0.4}, 0.12},
      {{0.3, 0.8}, 0.24},
      {{3, 4}, 1},
  });
  testHsub_satCases({
      {{0, -0.4}, 0.4},
      {{0.3, -0.4}, 0.7},
      {{0.3, 0.4}, 0},
      {{0.3, -0.8}, 1},
      {{1, 4}, 0},
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
  std::cout << "ret = " << ret << std::endl;
  return ret;
}
