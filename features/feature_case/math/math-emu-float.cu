// ====------------ math-emu-double.cu---------- *- CUDA -* -------------===////
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
typedef tuple<float, float, float> f_tuple3;
typedef tuple<float, float, float, float> f_tuple4;
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

__global__ void _norm3df(float *const DeviceResult, float Input1, float Input2,
                         float Input3) {
  *DeviceResult = norm3df(Input1, Input2, Input3);
}

void testNorm3df(float *const DeviceResult, float Input1, float Input2,
                 float Input3) {
  _norm3df<<<1, 1>>>(DeviceResult, Input1, Input2, Input3);
  cudaDeviceSynchronize();
  // TODO: Need test host side.
}

void testNorm3dfCases(const vector<pair<f_tuple3, fi_pair>> &TestCases) {
  float *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testNorm3df(DeviceResult, get<0>(TestCase.first), get<1>(TestCase.first),
                get<2>(TestCase.first));
    checkResult("norm3df",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first)},
                TestCase.second.first, *DeviceResult, TestCase.second.second);
  }
}

__global__ void _norm4d(float *const DeviceResult, float Input1, float Input2,
                        float Input3, float Input4) {
  *DeviceResult = norm4df(Input1, Input2, Input3, Input4);
}

void testNorm4df(float *const DeviceResult, float Input1, float Input2,
                 float Input3, float Input4) {
  _norm4d<<<1, 1>>>(DeviceResult, Input1, Input2, Input3, Input4);
  cudaDeviceSynchronize();
  // Call from host.
}

void testNorm4dfCases(const vector<pair<f_tuple4, fi_pair>> &TestCases) {
  float *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testNorm4df(DeviceResult, get<0>(TestCase.first), get<1>(TestCase.first),
                get<2>(TestCase.first), get<3>(TestCase.first));
    checkResult("norm4d",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first), get<3>(TestCase.first)},
                TestCase.second.first, *DeviceResult, TestCase.second.second);
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
  // Other test values.
  for (const auto &TestCase : TestCases) {
    testNormcdff(DeviceResult, TestCase.first);
    checkResult("normcdff", {TestCase.first}, TestCase.second.first,
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
  // Other test values.
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

__global__ void _rcbrtf(float *const DeviceResult, float Input1) {
  *DeviceResult = rcbrtf(Input1);
}

void testRcbrtfCases(const vector<pair<float, fi_pair>> &TestCases) {
  float *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    _rcbrtf<<<1, 1>>>(DeviceResult, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("rcbrtf", {TestCase.first}, TestCase.second.first,
                *DeviceResult, TestCase.second.second);
  }
}

__global__ void _rnorm3df(float *const DeviceResult, float Input1, float Input2,
                          float Input3) {
  *DeviceResult = rnorm3df(Input1, Input2, Input3);
}

void testRnorm3df(float *const DeviceResult, float Input1, float Input2,
                  float Input3) {
  _rnorm3df<<<1, 1>>>(DeviceResult, Input1, Input2, Input3);
  cudaDeviceSynchronize();
  // Call from host.
}

void testRnorm3dfCases(const vector<pair<f_tuple3, fi_pair>> &TestCases) {
  float *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testRnorm3df(DeviceResult, get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first));
    checkResult("rnorm3df",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first)},
                TestCase.second.first, *DeviceResult, TestCase.second.second);
  }
}

__global__ void _rnorm4df(float *const DeviceResult, float Input1, float Input2,
                          float Input3, float Input4) {
  *DeviceResult = rnorm4df(Input1, Input2, Input3, Input4);
}

void testRnorm4df(float *const DeviceResult, float Input1, float Input2,
                  float Input3, float Input4) {
  _rnorm4df<<<1, 1>>>(DeviceResult, Input1, Input2, Input3, Input4);
  cudaDeviceSynchronize();
  // Call from host.
}

void testRnorm4dfCases(const vector<pair<f_tuple4, fi_pair>> &TestCases) {
  float *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testRnorm4df(DeviceResult, get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first), get<3>(TestCase.first));
    checkResult("rnorm4df",
                {get<0>(TestCase.first), get<1>(TestCase.first),
                 get<2>(TestCase.first), get<3>(TestCase.first)},
                TestCase.second.first, *DeviceResult, TestCase.second.second);
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
  // Other test values.
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

int main() {
  testNorm3dfCases({
      {{-0.3, -0.34, -0.98}, {1.079814791679382, 15}},
      {{0.3, 0.34, 0.98}, {1.079814791679382, 15}},
      {{0.5, 456, 23}, {456.5799560546875, 13}},
      {{23, 432, 23}, {433.2228088378906, 13}},
  });
  testNorm4dfCases({
      {{-0.3, -0.34, -0.98, 1}, {1.471734, 6}},
      {{0.3, 0.34, 0.98, 1}, {1.471734, 6}},
      {{0.5, 456, 23, 1}, {456.5810546875, 13}},
      {{23, 432, 23, 1}, {433.2239685058594, 13}},
  });
  testNormcdffCases({
      {-5, {0.0000002866515842470108, 22}},
      {-3, {0.001349898055195808, 18}},
      {0, {0.5, 16}},
      {1, {0.8413447, 7}},
      {5, {0.9999997019767761, 16}},
  });
  testNormfCases({
      {{-0.3, -0.34, -0.98}, {1.079814791679382, 15}},
      {{0.3, 0.34, 0.98}, {1.079814791679382, 15}},
      {{0.5}, {0.5, 16}},
      {{23, 432, 23, 456, 23}, {629.402099609375, 13}},
  });
  testRcbrtfCases({
      {-0.3, {-1.494, 3}},
      {0.3, {1.494, 3}},
      {0.5, {1.26, 3}},
      {23, {0.3516, 4}},
  });
  testRnorm3dfCases({
      {{-0.3, -0.34, -0.98}, {0.9261, 4}},
      {{0.3, 0.34, 0.98}, {0.9261, 4}},
      {{0.5, 456, 23}, {0.0021902, 7}},
      {{23, 432, 23}, {0.0023083, 7}},
  });
  testRnorm4dfCases({
      {{-0.3, -0.34, -0.98, 1}, {0.6795, 4}},
      {{0.3, 0.34, 0.98, 1}, {0.6795, 4}},
      {{0.5, 456, 23, 1}, {0.0021902, 7}},
      {{23, 432, 23, 1}, {0.0023083, 7}},
  });
  testRnormfCases({
      {{-0.3, -0.34, -0.98}, {0.9261, 4}},
      {{0.3, 0.34, 0.98}, {0.9261, 4}},
      {{0.5}, {2, 3}},
      {{23, 432, 23, 456, 23}, {0.0015888, 7}},
  });
  cout << "passed " << passed << "/" << passed + failed << " cases!" << endl;
  if (failed) {
    cout << "failed!" << endl;
  }
  return failed;
}
