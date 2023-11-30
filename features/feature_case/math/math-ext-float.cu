// ====------------ math-ext-float.cu---------- *- CUDA -* --------------===////
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

using namespace std;

typedef vector<float> f_vector;
typedef pair<float, int> fi_pair;

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

template <typename T = float>
void checkResult(const string &FuncName, const vector<T> &Inputs,
                 const float &Expect, const float &DeviceResult,
                 const int precision) {
  cout << FuncName << "(" << Inputs[0];
  for (size_t i = 1; i < Inputs.size(); ++i) {
    cout << ", " << Inputs[i];
  }
  cout << ") = " << fixed << setprecision(precision) << DeviceResult
       << " (expect " << Expect - pow(10, -precision) << " ~ "
       << Expect + pow(10, -precision) << ")";
  cout.unsetf(ios::fixed);
  check(abs(DeviceResult - Expect) < pow(10, -precision));
}

__global__ void cylBesselI0f(float *const Result, float Input1) {
  *Result = cyl_bessel_i0f(Input1);
}

void testCylBesselI0fCases(const vector<pair<float, fi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    cylBesselI0f<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("cyl_bessel_i0f", {TestCase.first}, TestCase.second.first,
                *Result, TestCase.second.second);
  }
}

__global__ void cylBesselI1f(float *const Result, float Input1) {
  *Result = cyl_bessel_i1f(Input1);
}

void testCylBesselI1fCases(const vector<pair<float, fi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    cylBesselI1f<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("cyl_bessel_i1f", {TestCase.first}, TestCase.second.first,
                *Result, TestCase.second.second);
  }
}

__global__ void _erfcinvf(float *const DeviceResult, float Input) {
  *DeviceResult = erfcinvf(Input);
}

void testErfcinvf(float *const DeviceResult, float Input) {
  _erfcinvf<<<1, 1>>>(DeviceResult, Input);
  cudaDeviceSynchronize();
  // TODO: Need test host side.
}

void testErfcinvfCases(const vector<pair<float, fi_pair>> &TestCases) {
  float *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  // Boundary values.
  testErfcinvf(DeviceResult, 0);
  cout << "erfcinvf(" << 0 << ") = " << *DeviceResult << " (expect inf)";
  check(*DeviceResult > 999999.9);
  testErfcinvf(DeviceResult, 2);
  cout << "erfcinvf(" << 2 << ") = " << *DeviceResult << " (expect -inf)";
  check(*DeviceResult < -999999.9);
  // Other test values.
  for (const auto &TestCase : TestCases) {
    testErfcinvf(DeviceResult, TestCase.first);
    checkResult("erfcinvf", {TestCase.first}, TestCase.second.first,
                *DeviceResult, TestCase.second.second);
  }
}

__global__ void _erfinvf(float *const DeviceResult, float Input) {
  *DeviceResult = erfinvf(Input);
}

void testErfinvf(float *const DeviceResult, float Input) {
  _erfinvf<<<1, 1>>>(DeviceResult, Input);
  cudaDeviceSynchronize();
  // Call from host.
}

void testErfinvfCases(const vector<pair<float, fi_pair>> &TestCases) {
  float *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  // Boundary values.
  testErfinvf(DeviceResult, -1);
  cout << "erfinvf(" << -1 << ") = " << *DeviceResult << " (expect -inf)";
  check(*DeviceResult < -999999.9);
  testErfinvf(DeviceResult, 1);
  cout << "erfinvf(" << 1 << ") = " << *DeviceResult << " (expect inf)";
  check(*DeviceResult > 999999.9);
  // Other test values.
  for (const auto &TestCase : TestCases) {
    testErfinvf(DeviceResult, TestCase.first);
    checkResult("erfinvf", {TestCase.first}, TestCase.second.first,
                *DeviceResult, TestCase.second.second);
  }
}

__global__ void _j0f(float *const Result, float Input1) {
  *Result = j0f(Input1);
}

void testJ0fCases(const vector<pair<float, fi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _j0f<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("j0f", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void _j1f(float *const Result, float Input1) {
  *Result = j1f(Input1);
}

void testJ1fCases(const vector<pair<float, fi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _j1f<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("j1f", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void _normcdff(float *const DeviceResult, float Input) {
  *DeviceResult = normcdff(Input);
}

void testNormcdff(float *const DeviceResult, float Input) {
  _normcdff<<<1, 1>>>(DeviceResult, Input);
  cudaDeviceSynchronize();
  // Call from host.
}

void testNormcdffCases(const vector<pair<float, fi_pair>> &TestCases) {
  float *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testNormcdff(DeviceResult, TestCase.first);
    checkResult("normcdff", {TestCase.first}, TestCase.second.first,
                *DeviceResult, TestCase.second.second);
  }
}

__global__ void _normcdfinvf(float *const DeviceResult, float Input) {
  *DeviceResult = normcdfinvf(Input);
}

void testNormcdfinvf(float *const DeviceResult, float Input) {
  _normcdfinvf<<<1, 1>>>(DeviceResult, Input);
  cudaDeviceSynchronize();
  // Call from host.
}

void testNormcdfinvfCases(const vector<pair<float, fi_pair>> &TestCases) {
  float *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  // Boundary values.
  testNormcdfinvf(DeviceResult, 0);
  cout << "normcdfinvf(" << 0 << ") = " << *DeviceResult << " (expect -inf)";
  check(*DeviceResult < -999999.9);
  testNormcdfinvf(DeviceResult, 1);
  cout << "normcdfinvf(" << 1 << ") = " << *DeviceResult << " (expect inf)";
  check(*DeviceResult > 999999.9);
  // Other test values.
  for (const auto &TestCase : TestCases) {
    testNormcdfinvf(DeviceResult, TestCase.first);
    checkResult("normcdfinvf", {TestCase.first}, TestCase.second.first,
                *DeviceResult, TestCase.second.second);
  }
}

__global__ void setVecValue(float *Input1, const float Input2) {
  *Input1 = Input2;
}

__global__ void _normf(float *const DeviceResult, int Input1,
                       const float *Input2) {
  *DeviceResult = normf(Input1, Input2);
}

void testNormf(float *const DeviceResult, int Input1, const float *Input2) {
  _normf<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
  // Call from host.
}

void testNormfCases(const vector<pair<f_vector, fi_pair>> &TestCases) {
  float *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    float *Input;
    cudaMallocManaged(&Input, TestCase.first.size() * sizeof(*Input));
    for (size_t i = 0; i < TestCase.first.size(); ++i) {
      // Notice: cannot set value from host!
      setVecValue<<<1, 1>>>(Input + i, TestCase.first[i]);
      cudaDeviceSynchronize();
    }
    testNormf(DeviceResult, TestCase.first.size(), Input);
    string arg = "&{";
    for (size_t i = 0; i < TestCase.first.size() - 1; ++i) {
      arg += to_string(TestCase.first[i]) + ", ";
    }
    arg += to_string(TestCase.first.back()) + "}";
    checkResult<string>("normf", {to_string(TestCase.first.size()), arg},
                        TestCase.second.first, *DeviceResult,
                        TestCase.second.second);
  }
}

__global__ void _rnormf(float *const DeviceResult, int Input1,
                        const float *Input2) {
  *DeviceResult = rnormf(Input1, Input2);
}

void testRnormf(float *const DeviceResult, int Input1, const float *Input2) {
  _rnormf<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
  // Call from host.
}

void testRnormfCases(const vector<pair<f_vector, fi_pair>> &TestCases) {
  float *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    float *Input;
    cudaMallocManaged(&Input, TestCase.first.size() * sizeof(*Input));
    for (size_t i = 0; i < TestCase.first.size(); ++i) {
      // Notice: cannot set value from host!
      setVecValue<<<1, 1>>>(Input + i, TestCase.first[i]);
      cudaDeviceSynchronize();
    }
    testRnormf(DeviceResult, TestCase.first.size(), Input);
    string arg = "&{";
    for (size_t i = 0; i < TestCase.first.size() - 1; ++i) {
      arg += to_string(TestCase.first[i]) + ", ";
    }
    arg += to_string(TestCase.first.back()) + "}";
    checkResult<string>("rnormf", {to_string(TestCase.first.size()), arg},
                        TestCase.second.first, *DeviceResult,
                        TestCase.second.second);
  }
}

__global__ void _y0f(float *const Result, float Input1) {
  *Result = y0f(Input1);
}

void testY0fCases(const vector<pair<float, fi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _y0f<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("y0f", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void _y1f(float *const Result, float Input1) {
  *Result = y1f(Input1);
}

// Single Precision Intrinsics

void testY1fCases(const vector<pair<float, fi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _y1f<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("y1f", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void fadd_rd(float *const Result, float Input1, float Input2) {
  *Result = __fadd_rd(Input1, Input2);
}

void testFadd_rdCases(
    const vector<pair<pair<float, float>, fi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    fadd_rd<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__fadd_rd", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void fadd_rn(float *const Result, float Input1, float Input2) {
  *Result = __fadd_rn(Input1, Input2);
}

void testFadd_rnCases(
    const vector<pair<pair<float, float>, fi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    fadd_rn<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__fadd_rn", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void fadd_ru(float *const Result, float Input1, float Input2) {
  *Result = __fadd_ru(Input1, Input2);
}

void testFadd_ruCases(
    const vector<pair<pair<float, float>, fi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    fadd_ru<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__fadd_ru", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void fadd_rz(float *const Result, float Input1, float Input2) {
  *Result = __fadd_rz(Input1, Input2);
}

void testFadd_rzCases(
    const vector<pair<pair<float, float>, fi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    fadd_rz<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__fadd_rz", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void fmul_rd(float *const Result, float Input1, float Input2) {
  *Result = __fmul_rd(Input1, Input2);
}

void testFmul_rdCases(
    const vector<pair<pair<float, float>, fi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    fmul_rd<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__fmul_rd", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void fmul_rn(float *const Result, float Input1, float Input2) {
  *Result = __fmul_rn(Input1, Input2);
}

void testFmul_rnCases(
    const vector<pair<pair<float, float>, fi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    fmul_rn<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__fmul_rn", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void fmul_ru(float *const Result, float Input1, float Input2) {
  *Result = __fmul_ru(Input1, Input2);
}

void testFmul_ruCases(
    const vector<pair<pair<float, float>, fi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    fmul_ru<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__fmul_ru", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void fmul_rz(float *const Result, float Input1, float Input2) {
  *Result = __fmul_rz(Input1, Input2);
}

void testFmul_rzCases(
    const vector<pair<pair<float, float>, fi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    fmul_rz<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__fmul_rz", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void fsub_rd(float *const Result, float Input1, float Input2) {
  *Result = __fsub_rd(Input1, Input2);
}

void testFsub_rdCases(
    const vector<pair<pair<float, float>, fi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    fsub_rd<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__fsub_rd", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void fsub_rn(float *const Result, float Input1, float Input2) {
  *Result = __fsub_rn(Input1, Input2);
}

void testFsub_rnCases(
    const vector<pair<pair<float, float>, fi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    fsub_rn<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__fsub_rn", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void fsub_ru(float *const Result, float Input1, float Input2) {
  *Result = __fsub_ru(Input1, Input2);
}

void testFsub_ruCases(
    const vector<pair<pair<float, float>, fi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    fsub_ru<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__fsub_ru", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

__global__ void fsub_rz(float *const Result, float Input1, float Input2) {
  *Result = __fsub_rz(Input1, Input2);
}

void testFsub_rzCases(
    const vector<pair<pair<float, float>, fi_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    fsub_rz<<<1, 1>>>(Result, TestCase.first.first, TestCase.first.second);
    cudaDeviceSynchronize();
    checkResult("__fsub_rz", {TestCase.first.first, TestCase.first.second},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

int main() {
  testCylBesselI0fCases({
      {0.3, {1.022626876831055, 15}},
      {0.5, {1.063483357429504, 15}},
      {0.8, {1.166514992713928, 15}},
      {1.6, {1.749980688095093, 15}},
      {-5, {27.23987197875977, 14}},
  });
  testCylBesselI1fCases({
      {0.3, {0.1516939, 7}},
      {0.5, {0.2578943073749542, 16}},
      {0.8, {0.4328648149967194, 16}},
      {1.6, {1.084811, 6}},
      {-5, {-24.33564186096191, 14}},
  });
  testErfcinvfCases({
      {0.3, {0.7328690886497498, 16}},
      {0.5, {0.4769362807273865, 16}},
      {0.8, {0.1791434437036514, 16}},
      {1.6, {-0.5951161, 7}},
  });
  testErfinvfCases({
      {-0.3, {-0.2724627256393433, 16}},
      {-0.5, {-0.4769362807273865, 16}},
      {0, {0, 37}},
      {0.5, {0.4769362807273865, 16}},
  });
  testJ0fCases({
      {0.3, {0.9776262, 7}},
      {0.5, {0.9384698271751404, 16}},
      {0.8, {0.8462873, 7}},
      {1.6, {0.4554022, 7}},
      {-5, {-0.1775968, 7}},
  });
  testJ1fCases({
      {0.3, {0.1483188, 7}},
      {0.5, {0.2422684580087662, 16}},
      {0.8, {0.3688420653343201, 16}},
      {1.6, {0.569896, 7}},
      {-5, {0.3275791406631470, 16}},
  });
  testNormcdffCases({
      {-5, {0.0000002866515842470108, 22}},
      {-3, {0.001349898055195808, 18}},
      {0, {0.5, 16}},
      {1, {0.8413447141647339, 16}},
      {5, {0.9999997019767761, 16}},
  });
  testNormcdfinvfCases({
      {0.3, {-0.5244004130363464, 16}},
      {0.5, {0, 37}},
      {0.8, {0.8416212, 7}},
  });
  testNormfCases({
      {{-0.3, -0.34, -0.98}, {1.079814791679382, 15}},
      {{0.3, 0.34, 0.98}, {1.079814791679382, 15}},
      {{0.5}, {0.5, 16}},
      {{23, 432, 23, 456, 23}, {629.402099609375, 13}},
  });
  testRnormfCases({
      {{-0.3, -0.34, -0.98}, {0.9260847, 7}},
      {{0.3, 0.34, 0.98}, {0.9260847, 7}},
      {{0.5}, {2, 15}},
      {{23, 432, 23, 456, 23}, {0.001588809420354664, 18}},
  });
  testY0fCases({
      {0.3, {-0.8072735, 7}},
      {0.5, {-0.4445187, 7}},
      {0.8, {-0.08680226, 8}},
      {1.6, {0.4204270, 7}},
      {5, {-0.3085176050662994, 16}},
  });
  testY1fCases({
      {0.3, {-2.293104887008667, 15}},
      {0.5, {-1.471472, 6}},
      {0.8, {-0.9781441, 7}},
      {1.6, {-0.3475780, 7}},
      {5, {0.1478631347417831, 16}},
  });
  testFadd_rdCases({
      {{-0.3, -0.4}, {-0.7000000476837158, 16}},
      {{0.3, -0.4}, {-0.09999999403953552, 17}},
      {{0.3, 0.4}, {0.699999988079071, 16}},
      {{0.3, 0.8}, {1.100000023841858, 15}},
      {{3, 4}, {7, 15}},
  });
  testFadd_rnCases({
      {{-0.3, -0.4}, {-0.7000000476837158, 16}},
      {{0.3, -0.4}, {-0.09999999403953552, 17}},
      {{0.3, 0.4}, {0.7000000476837158, 16}},
      {{0.3, 0.8}, {1.100000023841858, 15}},
      {{3, 4}, {7, 15}},
  });
  testFadd_ruCases({
      {{-0.3, -0.4}, {-0.699999988079071, 16}},
      {{0.3, -0.4}, {-0.09999999403953552, 17}},
      {{0.3, 0.4}, {0.7000000476837158, 16}},
      {{0.3, 0.8}, {1.100000023841858, 15}},
      {{3, 4}, {7, 15}},
  });
  testFadd_rzCases({
      {{-0.3, -0.4}, {-0.699999988079071, 16}},
      {{0.3, -0.4}, {-0.09999999403953552, 17}},
      {{0.3, 0.4}, {0.699999988079071, 16}},
      {{0.3, 0.8}, {1.100000023841858, 15}},
      {{3, 4}, {7, 15}},
  });
  testFmul_rdCases({
      {{-0.3, -0.4}, {0.12000000476837158, 17}},
      {{0.3, -0.4}, {-0.12000001221895218, 17}},
      {{0.3, 0.4}, {0.12000000476837158, 17}},
      {{0.3, 0.8}, {0.2400000095367432, 16}},
      {{3, 4}, {12, 15}},
  });
  testFmul_rnCases({
      {{-0.3, -0.4}, {0.12000000476837158, 17}},
      {{0.3, -0.4}, {-0.12000000476837158, 17}},
      {{0.3, 0.4}, {0.12000000476837158, 17}},
      {{0.3, 0.8}, {0.2400000095367432, 16}},
      {{3, 4}, {12, 15}},
  });
  testFmul_ruCases({
      {{-0.3, -0.4}, {0.12000001221895218, 17}},
      {{0.3, -0.4}, {-0.12000000476837158, 17}},
      {{0.3, 0.4}, {0.12000001221895218, 17}},
      {{0.3, 0.8}, {0.2400000244379044, 16}},
      {{3, 4}, {12, 15}},
  });
  testFmul_rzCases({
      {{-0.3, -0.4}, {0.12000000476837158, 17}},
      {{0.3, -0.4}, {-0.12000000476837158, 17}},
      {{0.3, 0.4}, {0.12000000476837158, 17}},
      {{0.3, 0.8}, {0.2400000095367432, 16}},
      {{3, 4}, {12, 15}},
  });
  testFsub_rdCases({
      {{-0.3, -0.4}, {0.09999999403953552, 17}},
      {{0.3, -0.4}, {0.699999988079071, 16}},
      {{0.3, 0.4}, {-0.09999999403953552, 17}},
      {{0.3, 0.8}, {-0.5, 16}},
      {{3, 4}, {-1, 15}},
  });
  testFsub_rnCases({
      {{-0.3, -0.4}, {0.09999999403953552, 17}},
      {{0.3, -0.4}, {0.7000000476837158, 16}},
      {{0.3, 0.4}, {-0.09999999403953552, 17}},
      {{0.3, 0.8}, {-0.5, 16}},
      {{3, 4}, {-1, 15}},
  });
  testFsub_ruCases({
      {{-0.3, -0.4}, {0.09999999403953552, 17}},
      {{0.3, -0.4}, {0.7000000476837158, 16}},
      {{0.3, 0.4}, {-0.09999999403953552, 17}},
      {{0.3, 0.8}, {-0.5, 16}},
      {{3, 4}, {-1, 15}},
  });
  testFsub_rzCases({
      {{-0.3, -0.4}, {0.09999999403953552, 17}},
      {{0.3, -0.4}, {0.699999988079071, 16}},
      {{0.3, 0.4}, {-0.09999999403953552, 17}},
      {{0.3, 0.8}, {-0.5, 16}},
      {{3, 4}, {-1, 15}},
  });
  cout << "passed " << passed << "/" << passed + failed << " cases!" << endl;
  if (failed) {
    cout << "failed!" << endl;
  }
  return failed;
}
