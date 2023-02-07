// ====------ math-ext-half2.cu---------- *- CUDA -* ----===////
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

typedef pair<__half2, __half2> half2_pair;
typedef vector<__half2> half2_vector;

int ret = 0;

void check(bool IsPassed) {
  if (IsPassed) {
    std::cout << " ---- passed" << std::endl;
  } else {
    std::cout << " ---- failed" << std::endl;
    ret++;
  }
}

void printResultHalf2(const string &FuncName, const half2_vector &Inputs,
                      const __half2 &Expect,
                      const vector<float> &DeviceResult) {
  const float Precision = 0.001;
  std::cout << FuncName << "({" << __low2float(Inputs[0]) << ", "
            << __high2float(Inputs[0]) << "}";
  for (size_t i = 1; i < Inputs.size(); ++i) {
    std::cout << ", {" << __low2float(Inputs[i]) << ", "
              << __high2float(Inputs[i]) << "}";
  }
  std::cout << ") = {" << DeviceResult[0] << ", " << DeviceResult[1] << "}"
            << " (expect {" << __low2float(Expect) - Precision << " ~ "
            << __low2float(Expect) + Precision << ", "
            << DeviceResult[1] - Precision << " ~ "
            << DeviceResult[1] + Precision << "})";
  check(DeviceResult[0] > __low2float(Expect) - Precision &&
        DeviceResult[0] < __low2float(Expect) + Precision &&
        DeviceResult[1] > __high2float(Expect) - Precision &&
        DeviceResult[1] < __high2float(Expect) + Precision);
}

__global__ void hadd2_sat(float *const DeviceResult, __half2 Input1,
                          __half2 Input2) {
  auto ret = __hadd2_sat(Input1, Input2);
  DeviceResult[0] = __low2float(ret);
  DeviceResult[1] = __high2float(ret);
}

void testHadd2_sat(float *const DeviceResult, __half2 Input1, __half2 Input2) {
  hadd2_sat<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHadd2_satCases(const vector<pair<half2_pair, __half2>> &TestCases) {
  float *DeviceResult;
  cudaMallocManaged(&DeviceResult, 2 * sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testHadd2_sat(DeviceResult, TestCase.first.first, TestCase.first.second);
    printResultHalf2("__hadd2_sat",
                     {TestCase.first.first, TestCase.first.second},
                     TestCase.second, {DeviceResult[0], DeviceResult[1]});
  }
}

__global__ void hfma2_sat(float *const DeviceResult, __half2 Input1,
                          __half2 Input2, __half2 Input3) {
  auto ret = __hfma2_sat(Input1, Input2, Input3);
  DeviceResult[0] = __low2float(ret);
  DeviceResult[1] = __high2float(ret);
}

void testHfma2_sat(float *const DeviceResult, __half2 Input1, __half2 Input2,
                   __half2 Input3) {
  hfma2_sat<<<1, 1>>>(DeviceResult, Input1, Input2, Input3);
  cudaDeviceSynchronize();
}

void testHfma2_satCases(const vector<pair<half2_vector, __half2>> &TestCases) {
  float *DeviceResult;
  cudaMallocManaged(&DeviceResult, 2 * sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testHfma2_sat(DeviceResult, TestCase.first[0], TestCase.first[1],
                  TestCase.first[2]);
    printResultHalf2("__hfma2_sat", TestCase.first, TestCase.second,
                     {DeviceResult[0], DeviceResult[1]});
    if (TestCase.first.size() != 3) {
      ret++;
      std::cout << " ---- failed" << std::endl;
    }
  }
}

__global__ void hmul2_sat(float *const DeviceResult, __half2 Input1,
                          __half2 Input2) {
  auto ret = __hmul2_sat(Input1, Input2);
  DeviceResult[0] = __low2float(ret);
  DeviceResult[1] = __high2float(ret);
}

void testHmul2_sat(float *const DeviceResult, __half2 Input1, __half2 Input2) {
  hmul2_sat<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHmul2_satCases(const vector<pair<half2_pair, __half2>> &TestCases) {
  float *DeviceResult;
  cudaMallocManaged(&DeviceResult, 2 * sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testHmul2_sat(DeviceResult, TestCase.first.first, TestCase.first.second);
    printResultHalf2("__hmul2_sat",
                     {TestCase.first.first, TestCase.first.second},
                     TestCase.second, {DeviceResult[0], DeviceResult[1]});
  }
}

__global__ void hsub2_sat(float *const DeviceResult, __half2 Input1,
                          __half2 Input2) {
  auto ret = __hsub2_sat(Input1, Input2);
  DeviceResult[0] = __low2float(ret);
  DeviceResult[1] = __high2float(ret);
}

void testHsub2_sat(float *const DeviceResult, __half2 Input1, __half2 Input2) {
  hsub2_sat<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHsub2_satCases(const vector<pair<half2_pair, __half2>> &TestCases) {
  float *DeviceResult;
  cudaMallocManaged(&DeviceResult, 2 * sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testHsub2_sat(DeviceResult, TestCase.first.first, TestCase.first.second);
    printResultHalf2("__hsub2_sat",
                     {TestCase.first.first, TestCase.first.second},
                     TestCase.second, {DeviceResult[0], DeviceResult[1]});
  }
}

void printResultBool(const string &FuncName, const half2_vector &Inputs,
                     const bool &Expect, const bool &DeviceResult) {
  std::cout << FuncName << "({" << __low2float(Inputs[0]) << ", "
            << __high2float(Inputs[0]) << "}";
  for (size_t i = 1; i < Inputs.size(); ++i) {
    std::cout << ", {" << __low2float(Inputs[i]) << ", "
              << __high2float(Inputs[i]) << "}";
  }
  std::cout << ") = " << DeviceResult << " (expect " << Expect << ")";
  check(DeviceResult == Expect);
}

__global__ void hbeq2(bool *const DeviceResult, __half2 Input1,
                      __half2 Input2) {
  *DeviceResult = __hbeq2(Input1, Input2);
}

void testHbeq2(bool *const DeviceResult, __half2 Input1, __half2 Input2) {
  hbeq2<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHbeq2Cases(const vector<pair<half2_pair, bool>> &TestCases) {
  bool *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testHbeq2(DeviceResult, TestCase.first.first, TestCase.first.second);
    printResultBool("__hbeq2", {TestCase.first.first, TestCase.first.second},
                    TestCase.second, *DeviceResult);
  }
}

__global__ void hbequ2(bool *const DeviceResult, __half2 Input1,
                       __half2 Input2) {
  *DeviceResult = __hbequ2(Input1, Input2);
}

void testHbequ2(bool *const DeviceResult, __half2 Input1, __half2 Input2) {
  hbequ2<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHbequ2Cases(const vector<pair<half2_pair, bool>> &TestCases) {
  bool *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testHbequ2(DeviceResult, TestCase.first.first, TestCase.first.second);
    printResultBool("__hbequ2", {TestCase.first.first, TestCase.first.second},
                    TestCase.second, *DeviceResult);
  }
}

__global__ void hbge2(bool *const DeviceResult, __half2 Input1,
                      __half2 Input2) {
  *DeviceResult = __hbge2(Input1, Input2);
}

void testHbge2(bool *const DeviceResult, __half2 Input1, __half2 Input2) {
  hbge2<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHbge2Cases(const vector<pair<half2_pair, bool>> &TestCases) {
  bool *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testHbge2(DeviceResult, TestCase.first.first, TestCase.first.second);
    printResultBool("__hbge2", {TestCase.first.first, TestCase.first.second},
                    TestCase.second, *DeviceResult);
  }
}

__global__ void hbgeu2(bool *const DeviceResult, __half2 Input1,
                       __half2 Input2) {
  *DeviceResult = __hbgeu2(Input1, Input2);
}

void testHbgeu2(bool *const DeviceResult, __half2 Input1, __half2 Input2) {
  hbgeu2<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHbgeu2Cases(const vector<pair<half2_pair, bool>> &TestCases) {
  bool *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testHbgeu2(DeviceResult, TestCase.first.first, TestCase.first.second);
    printResultBool("__hbgeu2", {TestCase.first.first, TestCase.first.second},
                    TestCase.second, *DeviceResult);
  }
}

__global__ void hbgt2(bool *const DeviceResult, __half2 Input1,
                      __half2 Input2) {
  *DeviceResult = __hbgt2(Input1, Input2);
}

void testHbgt2(bool *const DeviceResult, __half2 Input1, __half2 Input2) {
  hbgt2<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHbgt2Cases(const vector<pair<half2_pair, bool>> &TestCases) {
  bool *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testHbgt2(DeviceResult, TestCase.first.first, TestCase.first.second);
    printResultBool("__hbgt2", {TestCase.first.first, TestCase.first.second},
                    TestCase.second, *DeviceResult);
  }
}

__global__ void hbgtu2(bool *const DeviceResult, __half2 Input1,
                       __half2 Input2) {
  *DeviceResult = __hbgtu2(Input1, Input2);
}

void testHbgtu2(bool *const DeviceResult, __half2 Input1, __half2 Input2) {
  hbgtu2<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHbgtu2Cases(const vector<pair<half2_pair, bool>> &TestCases) {
  bool *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testHbgtu2(DeviceResult, TestCase.first.first, TestCase.first.second);
    printResultBool("__hbgtu2", {TestCase.first.first, TestCase.first.second},
                    TestCase.second, *DeviceResult);
  }
}

__global__ void hble2(bool *const DeviceResult, __half2 Input1,
                      __half2 Input2) {
  *DeviceResult = __hble2(Input1, Input2);
}

void testHble2(bool *const DeviceResult, __half2 Input1, __half2 Input2) {
  hble2<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHble2Cases(const vector<pair<half2_pair, bool>> &TestCases) {
  bool *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testHble2(DeviceResult, TestCase.first.first, TestCase.first.second);
    printResultBool("__hble2", {TestCase.first.first, TestCase.first.second},
                    TestCase.second, *DeviceResult);
  }
}

__global__ void hbleu2(bool *const DeviceResult, __half2 Input1,
                       __half2 Input2) {
  *DeviceResult = __hbleu2(Input1, Input2);
}

void testHbleu2(bool *const DeviceResult, __half2 Input1, __half2 Input2) {
  hbleu2<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHbleu2Cases(const vector<pair<half2_pair, bool>> &TestCases) {
  bool *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testHbleu2(DeviceResult, TestCase.first.first, TestCase.first.second);
    printResultBool("__hbleu2", {TestCase.first.first, TestCase.first.second},
                    TestCase.second, *DeviceResult);
  }
}

__global__ void hblt2(bool *const DeviceResult, __half2 Input1,
                      __half2 Input2) {
  *DeviceResult = __hblt2(Input1, Input2);
}

void testHblt2(bool *const DeviceResult, __half2 Input1, __half2 Input2) {
  hblt2<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHblt2Cases(const vector<pair<half2_pair, bool>> &TestCases) {
  bool *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testHblt2(DeviceResult, TestCase.first.first, TestCase.first.second);
    printResultBool("__hblt2", {TestCase.first.first, TestCase.first.second},
                    TestCase.second, *DeviceResult);
  }
}

__global__ void hbltu2(bool *const DeviceResult, __half2 Input1,
                       __half2 Input2) {
  *DeviceResult = __hbltu2(Input1, Input2);
}

void testHbltu2(bool *const DeviceResult, __half2 Input1, __half2 Input2) {
  hbltu2<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHbltu2Cases(const vector<pair<half2_pair, bool>> &TestCases) {
  bool *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testHbltu2(DeviceResult, TestCase.first.first, TestCase.first.second);
    printResultBool("__hbltu2", {TestCase.first.first, TestCase.first.second},
                    TestCase.second, *DeviceResult);
  }
}

__global__ void hbne2(bool *const DeviceResult, __half2 Input1,
                      __half2 Input2) {
  *DeviceResult = __hbne2(Input1, Input2);
}

void testHbne2(bool *const DeviceResult, __half2 Input1, __half2 Input2) {
  hbne2<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHbne2Cases(const vector<pair<half2_pair, bool>> &TestCases) {
  bool *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testHbne2(DeviceResult, TestCase.first.first, TestCase.first.second);
    printResultBool("__hbne2", {TestCase.first.first, TestCase.first.second},
                    TestCase.second, *DeviceResult);
  }
}

__global__ void hbneu2(bool *const DeviceResult, __half2 Input1,
                       __half2 Input2) {
  *DeviceResult = __hbneu2(Input1, Input2);
}

void testHbneu2(bool *const DeviceResult, __half2 Input1, __half2 Input2) {
  hbneu2<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHbneu2Cases(const vector<pair<half2_pair, bool>> &TestCases) {
  bool *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testHbneu2(DeviceResult, TestCase.first.first, TestCase.first.second);
    printResultBool("__hbneu2", {TestCase.first.first, TestCase.first.second},
                    TestCase.second, *DeviceResult);
  }
}

__global__ void heq2(float *const DeviceResult, __half2 Input1,
                     __half2 Input2) {
  auto ret = __heq2(Input1, Input2);
  DeviceResult[0] = __low2float(ret);
  DeviceResult[1] = __high2float(ret);
}

void testHeq2(float *const DeviceResult, __half2 Input1, __half2 Input2) {
  heq2<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHeq2Cases(const vector<pair<half2_pair, __half2>> &TestCases) {
  float *DeviceResult;
  cudaMallocManaged(&DeviceResult, 2 * sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testHeq2(DeviceResult, TestCase.first.first, TestCase.first.second);
    printResultHalf2("__heq2", {TestCase.first.first, TestCase.first.second},
                     TestCase.second, {DeviceResult[0], DeviceResult[1]});
  }
}

__global__ void hequ2(float *const DeviceResult, __half2 Input1,
                      __half2 Input2) {
  auto ret = __hequ2(Input1, Input2);
  DeviceResult[0] = __low2float(ret);
  DeviceResult[1] = __high2float(ret);
}

void testHequ2(float *const DeviceResult, __half2 Input1, __half2 Input2) {
  hequ2<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHequ2Cases(const vector<pair<half2_pair, __half2>> &TestCases) {
  float *DeviceResult;
  cudaMallocManaged(&DeviceResult, 2 * sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testHequ2(DeviceResult, TestCase.first.first, TestCase.first.second);
    printResultHalf2("__hequ2", {TestCase.first.first, TestCase.first.second},
                     TestCase.second, {DeviceResult[0], DeviceResult[1]});
  }
}

__global__ void hge2(float *const DeviceResult, __half2 Input1,
                     __half2 Input2) {
  auto ret = __hge2(Input1, Input2);
  DeviceResult[0] = __low2float(ret);
  DeviceResult[1] = __high2float(ret);
}

void testHge2(float *const DeviceResult, __half2 Input1, __half2 Input2) {
  hge2<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHge2Cases(const vector<pair<half2_pair, __half2>> &TestCases) {
  float *DeviceResult;
  cudaMallocManaged(&DeviceResult, 2 * sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testHge2(DeviceResult, TestCase.first.first, TestCase.first.second);
    printResultHalf2("__hge2", {TestCase.first.first, TestCase.first.second},
                     TestCase.second, {DeviceResult[0], DeviceResult[1]});
  }
}

__global__ void hgeu2(float *const DeviceResult, __half2 Input1,
                      __half2 Input2) {
  auto ret = __hgeu2(Input1, Input2);
  DeviceResult[0] = __low2float(ret);
  DeviceResult[1] = __high2float(ret);
}

void testHgeu2(float *const DeviceResult, __half2 Input1, __half2 Input2) {
  hgeu2<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHgeu2Cases(const vector<pair<half2_pair, __half2>> &TestCases) {
  float *DeviceResult;
  cudaMallocManaged(&DeviceResult, 2 * sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testHgeu2(DeviceResult, TestCase.first.first, TestCase.first.second);
    printResultHalf2("__hgeu2", {TestCase.first.first, TestCase.first.second},
                     TestCase.second, {DeviceResult[0], DeviceResult[1]});
  }
}

__global__ void hgt2(float *const DeviceResult, __half2 Input1,
                     __half2 Input2) {
  auto ret = __hgt2(Input1, Input2);
  DeviceResult[0] = __low2float(ret);
  DeviceResult[1] = __high2float(ret);
}

void testHgt2(float *const DeviceResult, __half2 Input1, __half2 Input2) {
  hgt2<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHgt2Cases(const vector<pair<half2_pair, __half2>> &TestCases) {
  float *DeviceResult;
  cudaMallocManaged(&DeviceResult, 2 * sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testHgt2(DeviceResult, TestCase.first.first, TestCase.first.second);
    printResultHalf2("__hgt2", {TestCase.first.first, TestCase.first.second},
                     TestCase.second, {DeviceResult[0], DeviceResult[1]});
  }
}

__global__ void hgtu2(float *const DeviceResult, __half2 Input1,
                      __half2 Input2) {
  auto ret = __hgtu2(Input1, Input2);
  DeviceResult[0] = __low2float(ret);
  DeviceResult[1] = __high2float(ret);
}

void testHgtu2(float *const DeviceResult, __half2 Input1, __half2 Input2) {
  hgtu2<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHgtu2Cases(const vector<pair<half2_pair, __half2>> &TestCases) {
  float *DeviceResult;
  cudaMallocManaged(&DeviceResult, 2 * sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testHgtu2(DeviceResult, TestCase.first.first, TestCase.first.second);
    printResultHalf2("__hgtu2", {TestCase.first.first, TestCase.first.second},
                     TestCase.second, {DeviceResult[0], DeviceResult[1]});
  }
}

__global__ void hisnan2(float *const DeviceResult, __half2 Input1) {
  auto ret = __hisnan2(Input1);
  DeviceResult[0] = __low2float(ret);
  DeviceResult[1] = __high2float(ret);
}

void testHisnan2(float *const DeviceResult, __half2 Input1) {
  hisnan2<<<1, 1>>>(DeviceResult, Input1);
  cudaDeviceSynchronize();
}

void testHisnan2Cases(const vector<half2_pair> &TestCases) {
  float *DeviceResult;
  cudaMallocManaged(&DeviceResult, 2 * sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testHisnan2(DeviceResult, TestCase.first);
    printResultHalf2("__hisnan2", {TestCase.first}, TestCase.second,
                     {DeviceResult[0], DeviceResult[1]});
  }
}

__global__ void hle2(float *const DeviceResult, __half2 Input1,
                     __half2 Input2) {
  auto ret = __hle2(Input1, Input2);
  DeviceResult[0] = __low2float(ret);
  DeviceResult[1] = __high2float(ret);
}

void testHle2(float *const DeviceResult, __half2 Input1, __half2 Input2) {
  hle2<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHle2Cases(const vector<pair<half2_pair, __half2>> &TestCases) {
  float *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testHle2(DeviceResult, TestCase.first.first, TestCase.first.second);
    printResultHalf2("__hle2", {TestCase.first.first, TestCase.first.second},
                     TestCase.second, {DeviceResult[0], DeviceResult[1]});
  }
}

__global__ void hleu2(float *const DeviceResult, __half2 Input1,
                     __half2 Input2) {
  auto ret = __hleu2(Input1, Input2);
  DeviceResult[0] = __low2float(ret);
  DeviceResult[1] = __high2float(ret);
}

void testHleu2(float *const DeviceResult, __half2 Input1, __half2 Input2) {
  hleu2<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHleu2Cases(const vector<pair<half2_pair, __half2>> &TestCases) {
  float *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testHleu2(DeviceResult, TestCase.first.first, TestCase.first.second);
    printResultHalf2("__hleu2", {TestCase.first.first, TestCase.first.second},
                     TestCase.second, {DeviceResult[0], DeviceResult[1]});
  }
}

__global__ void hlt2(float *const DeviceResult, __half2 Input1,
                     __half2 Input2) {
  auto ret = __hlt2(Input1, Input2);
  DeviceResult[0] = __low2float(ret);
  DeviceResult[1] = __high2float(ret);
}

void testHlt2(float *const DeviceResult, __half2 Input1, __half2 Input2) {
  hlt2<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHlt2Cases(const vector<pair<half2_pair, __half2>> &TestCases) {
  float *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testHlt2(DeviceResult, TestCase.first.first, TestCase.first.second);
    printResultHalf2("__hlt2", {TestCase.first.first, TestCase.first.second},
                     TestCase.second, {DeviceResult[0], DeviceResult[1]});
  }
}

__global__ void hltu2(float *const DeviceResult, __half2 Input1,
                      __half2 Input2) {
  auto ret = __hltu2(Input1, Input2);
  DeviceResult[0] = __low2float(ret);
  DeviceResult[1] = __high2float(ret);
}

void testHltu2(float *const DeviceResult, __half2 Input1, __half2 Input2) {
  hltu2<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHltu2Cases(const vector<pair<half2_pair, __half2>> &TestCases) {
  float *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testHltu2(DeviceResult, TestCase.first.first, TestCase.first.second);
    printResultHalf2("__hltu2", {TestCase.first.first, TestCase.first.second},
                     TestCase.second, {DeviceResult[0], DeviceResult[1]});
  }
}

__global__ void hne2(float *const DeviceResult, __half2 Input1,
                     __half2 Input2) {
  auto ret = __hne2(Input1, Input2);
  DeviceResult[0] = __low2float(ret);
  DeviceResult[1] = __high2float(ret);
}

void testHne2(float *const DeviceResult, __half2 Input1, __half2 Input2) {
  hne2<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHne2Cases(const vector<pair<half2_pair, __half2>> &TestCases) {
  float *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testHne2(DeviceResult, TestCase.first.first, TestCase.first.second);
    printResultHalf2("__hne2", {TestCase.first.first, TestCase.first.second},
                     TestCase.second, {DeviceResult[0], DeviceResult[1]});
  }
}

__global__ void hneu2(float *const DeviceResult, __half2 Input1,
                      __half2 Input2) {
  auto ret = __hneu2(Input1, Input2);
  DeviceResult[0] = __low2float(ret);
  DeviceResult[1] = __high2float(ret);
}

void testHneu2(float *const DeviceResult, __half2 Input1, __half2 Input2) {
  hneu2<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
}

void testHneu2Cases(const vector<pair<half2_pair, __half2>> &TestCases) {
  float *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testHneu2(DeviceResult, TestCase.first.first, TestCase.first.second);
    printResultHalf2("__hneu2", {TestCase.first.first, TestCase.first.second},
                     TestCase.second, {DeviceResult[0], DeviceResult[1]});
  }
}

int main() {
  testHadd2_satCases({
      {{{-0.3, -0.5}, {-0.4, -0.6}}, {0, 0}},
      {{{0.3, 0.5}, {-0.4, 0.6}}, {0, 1}},
      {{{0.3, 0.5}, {0.4, 0.2}}, {0.7, 0.7}},
      {{{0.3, 0.5}, {0.4, 0.6}}, {0.7, 1}},
      {{{3, 5}, {4, 6}}, {1, 1}},
  });
  testHfma2_satCases({
      {{{-0.3, -0.5}, {-0.4, -0.6}, {-0.2, -0.7}}, {0, 0}},
      {{{0.3, 0.5}, {-0.4, 0.6}, {-0.1, 0.2}}, {0, 0.5}},
      {{{0.3, 0.5}, {0.4, 0.2}, {0.1, 0.1}}, {0.22, 0.2}},
      {{{0.3, 0.5}, {0.4, 0.6}, {0, 0.3}}, {0.12, 0.6}},
      {{{3, 5}, {4, 6}, {5, 8}}, {1, 1}},
  });
  testHmul2_satCases({
      {{{-0.3, -5}, {0.4, 6}}, {0, 0}},
      {{{0.3, 5}, {-4, 0.6}}, {0, 1}},
      {{{0.3, 0.5}, {0.4, 0.2}}, {0.12, 0.1}},
      {{{0.3, 0.5}, {0.4, 0.6}}, {0.12, 0.3}},
      {{{3, 5}, {4, 6}}, {1, 1}},
  });
  testHsub2_satCases({
      {{{0, 0}, {-0.4, -0.6}}, {0.4, 0.6}},
      {{{0, 1}, {0.4, 0.6}}, {0, 0.4}},
      {{{0.7, 0.7}, {0.4, 0.2}}, {0.3, 0.5}},
      {{{0.7, 2}, {0.4, 0.6}}, {0.3, 1}},
      {{{1, 1}, {4, 6}}, {0, 0}},
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
      {{{0, 0}, {-0.4, -0.6}}, {0, 0}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {0, 1}},
      {{{0.7, 2}, {0.7, 2}}, {1, 1}},
      {{{1, 1}, {4, 6}}, {0, 0}},
      {{{NAN, 1}, {1, 1}}, {0, 1}},
  });
  testHequ2Cases({
      {{{0, 0}, {-0.4, -0.6}}, {0, 0}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {0, 1}},
      {{{0.7, 2}, {0.7, 2}}, {1, 1}},
      {{{1, 1}, {4, 6}}, {0, 0}},
      {{{NAN, 1}, {1, 1}}, {1, 1}},
  });
  testHge2Cases({
      {{{0, 0}, {-0.4, -0.6}}, {1, 1}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {1, 1}},
      {{{0.7, 2}, {0.7, 2}}, {1, 1}},
      {{{1, 1}, {4, 6}}, {0, 0}},
      {{{NAN, 1}, {1, 1}}, {0, 1}},
  });
  testHgeu2Cases({
      {{{0, 0}, {-0.4, -0.6}}, {1, 1}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {1, 1}},
      {{{0.7, 2}, {0.7, 2}}, {1, 1}},
      {{{1, 1}, {4, 6}}, {0, 0}},
      {{{NAN, 1}, {1, 1}}, {1, 1}},
  });
  testHgt2Cases({
      {{{0, 0}, {-0.4, -0.6}}, {1, 1}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {1, 0}},
      {{{0.7, 2}, {0.7, 2}}, {0, 0}},
      {{{1, 1}, {4, 6}}, {0, 0}},
      {{{NAN, 1}, {1, 1}}, {0, 0}},
  });
  testHgtu2Cases({
      {{{0, 0}, {-0.4, -0.6}}, {1, 1}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {1, 0}},
      {{{0.7, 2}, {0.7, 2}}, {0, 0}},
      {{{1, 1}, {4, 6}}, {0, 0}},
      {{{NAN, 1}, {1, 1}}, {1, 0}},
  });
  testHisnan2Cases({
      {{0, 0}, {0, 0}},
      {{0.7, 2}, {0, 0}},
      {{NAN, 1}, {1, 0}},
      {{NAN, NAN}, {1, 1}},
      {{0, NAN}, {0, 1}},
  });
  testHle2Cases({
      {{{0, 0}, {-0.4, -0.6}}, {0, 0}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {0, 1}},
      {{{0.7, 2}, {0.7, 2}}, {1, 1}},
      {{{1, 1}, {4, 6}}, {1, 1}},
      {{{NAN, 1}, {1, 1}}, {0, 1}},
  });
  testHleu2Cases({
      {{{0, 0}, {-0.4, -0.6}}, {0, 0}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {0, 1}},
      {{{0.7, 2}, {0.7, 2}}, {1, 1}},
      {{{1, 1}, {4, 6}}, {1, 1}},
      {{{NAN, 1}, {1, 1}}, {1, 1}},
  });
  testHlt2Cases({
      {{{0, 0}, {-0.4, -0.6}}, {0, 0}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {0, 0}},
      {{{0.7, 2}, {0.7, 2}}, {0, 0}},
      {{{1, 1}, {4, 6}}, {1, 1}},
      {{{NAN, 1}, {1, 1}}, {0, 0}},
  });
  testHltu2Cases({
      {{{0, 0}, {-0.4, -0.6}}, {0, 0}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {0, 0}},
      {{{0.7, 2}, {0.7, 2}}, {0, 0}},
      {{{1, 1}, {4, 6}}, {1, 1}},
      {{{NAN, 1}, {1, 1}}, {1, 0}},
  });
  testHne2Cases({
      {{{0, 0}, {-0.4, -0.6}}, {1, 1}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {1, 0}},
      {{{0.7, 2}, {0.7, 2}}, {0, 0}},
      {{{1, 1}, {4, 6}}, {1, 1}},
      {{{NAN, 1}, {1, 1}}, {0, 0}},
  });
  testHneu2Cases({
      {{{0, 0}, {-0.4, -0.6}}, {1, 1}},
      {{{0.7, 0.7}, {0.4, 0.7}}, {1, 0}},
      {{{0.7, 2}, {0.7, 2}}, {0, 0}},
      {{{1, 1}, {4, 6}}, {1, 1}},
      {{{NAN, 1}, {1, 1}}, {1, 0}},
  });
  std::cout << "ret = " << ret << std::endl;
  return ret;
}
