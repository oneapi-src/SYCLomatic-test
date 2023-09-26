// ====------------ math-ext-double.cu---------- *- CUDA -* -------------===////
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

typedef pair<__nv_bfloat16, int> bf16i_pair;
typedef vector<double> d_vector;
typedef pair<double, int> di_pair;

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
  cout << FuncName << "(" << Inputs[0] << "";
  for (size_t i = 1; i < Inputs.size(); ++i) {
    cout << ", " << Inputs[i];
  }
  cout << ") = " << fixed << setprecision(precision < 0 ? 0 : precision)
       << Result << " (expect " << Expect - pow(10, -precision) << " ~ "
       << Expect + pow(10, -precision) << ")";
  cout.unsetf(ios::fixed);
  check(abs(Result - Expect) < pow(10, -precision));
}

void checkResult(const string &FuncName, const vector<float> &Inputs,
                 const __nv_bfloat16 &Expect, const float &Result,
                 const int precision) {
  float FExpect = __bfloat162float(Expect);
  checkResult(FuncName, Inputs, FExpect, Result, precision);
}

template <typename T = double>
void checkResult(const string &FuncName, const vector<T> &Inputs,
                 const double &Expect, const double &DeviceResult,
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

// Bfloat16 Precision Conversion and Data Movement

__global__ void double2bfloat16(float *const Result, double Input1) {
  *Result = __double2bfloat16(Input1);
}

void testDouble2bfloat16Cases(
    const vector<pair<double, bf16i_pair>> &TestCases) {
  float *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    double2bfloat16<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("__double2bfloat16", {(float)TestCase.first},
                TestCase.second.first, *Result, TestCase.second.second);
  }
}

// Double Precision Mathematical Functions

__global__ void cylBesselI0(double *const Result, double Input1) {
  *Result = cyl_bessel_i0(Input1);
}

void testCylBesselI0Cases(const vector<pair<double, di_pair>> &TestCases) {
  double *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    cylBesselI0<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("cyl_bessel_i0", {TestCase.first}, TestCase.second.first,
                *Result, TestCase.second.second);
  }
}

__global__ void cylBesselI1(double *const Result, double Input1) {
  *Result = cyl_bessel_i1(Input1);
}

void testCylBesselI1Cases(const vector<pair<double, di_pair>> &TestCases) {
  double *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    cylBesselI1<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("cyl_bessel_i1", {TestCase.first}, TestCase.second.first,
                *Result, TestCase.second.second);
  }
}

__global__ void _erfcinv(double *const DeviceResult, double Input) {
  *DeviceResult = erfcinv(Input);
}

void testErfcinv(double *const DeviceResult, double Input) {
  _erfcinv<<<1, 1>>>(DeviceResult, Input);
  cudaDeviceSynchronize();
  // TODO: Need test host side.
}

void testErfcinvCases(const vector<pair<double, di_pair>> &TestCases) {
  double *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  // Boundary values.
  testErfcinv(DeviceResult, 0);
  cout << "erfcinv(" << 0 << ") = " << *DeviceResult << " (expect inf)";
  check(*DeviceResult > 999999.9);
  testErfcinv(DeviceResult, 2);
  cout << "erfcinv(" << 2 << ") = " << *DeviceResult << " (expect -inf)";
  check(*DeviceResult < -999999.9);
  // Other test values.
  for (const auto &TestCase : TestCases) {
    testErfcinv(DeviceResult, TestCase.first);
    checkResult("erfcinv", {TestCase.first}, TestCase.second.first,
                *DeviceResult, TestCase.second.second);
  }
}

__global__ void _erfinv(double *const DeviceResult, double Input) {
  *DeviceResult = erfinv(Input);
}

void testErfinv(double *const DeviceResult, double Input) {
  _erfinv<<<1, 1>>>(DeviceResult, Input);
  cudaDeviceSynchronize();
  // Call from host.
}

void testErfinvCases(const vector<pair<double, di_pair>> &TestCases) {
  double *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  // Boundary values.
  testErfinv(DeviceResult, -1);
  cout << "erfinv(" << -1 << ") = " << *DeviceResult << " (expect -inf)";
  check(*DeviceResult < -999999.9);
  testErfinv(DeviceResult, 1);
  cout << "erfinv(" << 1 << ") = " << *DeviceResult << " (expect inf)";
  check(*DeviceResult > 999999.9);
  // Other test values.
  for (const auto &TestCase : TestCases) {
    testErfinv(DeviceResult, TestCase.first);
    checkResult("erfinv", {TestCase.first}, TestCase.second.first,
                *DeviceResult, TestCase.second.second);
  }
}

__global__ void _j0(double *const Result, double Input1) {
  *Result = j0(Input1);
}

void testJ0Cases(const vector<pair<double, di_pair>> &TestCases) {
  double *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _j0<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("j0", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void _j1(double *const Result, double Input1) {
  *Result = j1(Input1);
}

void testJ1Cases(const vector<pair<double, di_pair>> &TestCases) {
  double *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _j1<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("j1", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void setVecValue(double *Input1, const double Input2) {
  *Input1 = Input2;
}

__global__ void _norm(double *const DeviceResult, int Input1,
                      const double *Input2) {
  *DeviceResult = norm(Input1, Input2);
}

void testNorm(double *const DeviceResult, int Input1, const double *Input2) {
  _norm<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
  // Call from host.
}

void testNormCases(const vector<pair<d_vector, di_pair>> &TestCases) {
  double *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    double *Input;
    cudaMallocManaged(&Input, TestCase.first.size() * sizeof(*Input));
    for (size_t i = 0; i < TestCase.first.size(); ++i) {
      // Notice: cannot set value from host!
      setVecValue<<<1, 1>>>(Input + i, TestCase.first[i]);
      cudaDeviceSynchronize();
    }
    testNorm(DeviceResult, TestCase.first.size(), Input);
    string arg = "&{";
    for (size_t i = 0; i < TestCase.first.size() - 1; ++i) {
      arg += to_string(TestCase.first[i]) + ", ";
    }
    arg += to_string(TestCase.first.back()) + "}";
    checkResult<string>("norm", {to_string(TestCase.first.size()), arg},
                        TestCase.second.first, *DeviceResult,
                        TestCase.second.second);
  }
}

__global__ void _normcdf(double *const DeviceResult, double Input) {
  *DeviceResult = normcdf(Input);
}

void testNormcdf(double *const DeviceResult, double Input) {
  _normcdf<<<1, 1>>>(DeviceResult, Input);
  cudaDeviceSynchronize();
  // Call from host.
}

void testNormcdfCases(const vector<pair<double, di_pair>> &TestCases) {
  double *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    testNormcdf(DeviceResult, TestCase.first);
    checkResult("normcdf", {TestCase.first}, TestCase.second.first,
                *DeviceResult, TestCase.second.second);
  }
}

__global__ void _normcdfinv(double *const DeviceResult, double Input) {
  *DeviceResult = normcdfinv(Input);
}

void testNormcdfinv(double *const DeviceResult, double Input) {
  _normcdfinv<<<1, 1>>>(DeviceResult, Input);
  cudaDeviceSynchronize();
  // Call from host.
}

void testNormcdfinvCases(const vector<pair<double, di_pair>> &TestCases) {
  double *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  // Boundary values.
  testNormcdfinv(DeviceResult, 0);
  cout << "normcdfinv(" << 0 << ") = " << *DeviceResult << " (expect -inf)";
  check(*DeviceResult < -999999.9);
  testNormcdfinv(DeviceResult, 1);
  cout << "normcdfinv(" << 1 << ") = " << *DeviceResult << " (expect inf)";
  check(*DeviceResult > 999999.9);
  // Other test values.
  for (const auto &TestCase : TestCases) {
    testNormcdfinv(DeviceResult, TestCase.first);
    checkResult("normcdfinv", {TestCase.first}, TestCase.second.first,
                *DeviceResult, TestCase.second.second);
  }
}

__global__ void _rnorm(double *const DeviceResult, int Input1,
                       const double *Input2) {
  *DeviceResult = rnorm(Input1, Input2);
}

void testRnorm(double *const DeviceResult, int Input1, const double *Input2) {
  _rnorm<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
  // Call from host.
}

void testRnormCases(const vector<pair<d_vector, di_pair>> &TestCases) {
  double *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  for (const auto &TestCase : TestCases) {
    double *Input;
    cudaMallocManaged(&Input, TestCase.first.size() * sizeof(*Input));
    for (size_t i = 0; i < TestCase.first.size(); ++i) {
      // Notice: cannot set value from host!
      setVecValue<<<1, 1>>>(Input + i, TestCase.first[i]);
      cudaDeviceSynchronize();
    }
    testRnorm(DeviceResult, TestCase.first.size(), Input);
    string arg = "&{";
    for (size_t i = 0; i < TestCase.first.size() - 1; ++i) {
      arg += to_string(TestCase.first[i]) + ", ";
    }
    arg += to_string(TestCase.first.back()) + "}";
    checkResult<string>("rnorm", {to_string(TestCase.first.size()), arg},
                        TestCase.second.first, *DeviceResult,
                        TestCase.second.second);
  }
}

__global__ void _y0(double *const Result, double Input1) {
  *Result = y0(Input1);
}

void testY0Cases(const vector<pair<double, di_pair>> &TestCases) {
  double *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _y0<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("y0", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

__global__ void _y1(double *const Result, double Input1) {
  *Result = y1(Input1);
}

void testY1Cases(const vector<pair<double, di_pair>> &TestCases) {
  double *Result;
  cudaMallocManaged(&Result, sizeof(*Result));
  for (const auto &TestCase : TestCases) {
    _y1<<<1, 1>>>(Result, TestCase.first);
    cudaDeviceSynchronize();
    checkResult("y1", {TestCase.first}, TestCase.second.first, *Result,
                TestCase.second.second);
  }
}

int main() {
  testDouble2bfloat16Cases({
      {-0.3, {-0.30078125, 16}},
      {0.3, {0.30078125, 16}},
      {30, {30, 14}},
      {0.432643, {0.43359375, 16}},
      {1, {1, 15}},
      {10.7, {10.6875, 15}},
  });
  testCylBesselI0Cases({
      {0.3, {1.022626879351597, 15}},
      {0.5, {1.063483370741324, 15}},
      {0.8, {1.166514922869803, 15}},
      {1.6, {1.749980639738909, 15}},
      {-5, {27.23987182360445, 14}},
  });
  testCylBesselI1Cases({
      {0.3, {0.1516938400035928, 16}},
      {0.5, {0.2578943053908963, 16}},
      {0.8, {0.4328648026206398, 16}},
      {1.6, {1.08481063512988, 15}},
      {-5, {-24.33564214245052, 14}},
  });
  testErfcinvCases({
      {0.3, {0.732869077959217, 15}},
      {0.5, {0.4769362762044698, 16}},
      {0.8, {0.1791434546212916, 16}},
      {1.6, {-0.595116081449995, 15}},
  });
  testErfinvCases({
      {-0.3, {-0.2724627147267544, 16}},
      {-0.5, {-0.4769362762044698, 16}},
      {0, {0, 37}},
      {0.5, {0.4769362762044698, 16}},
  });
  testJ0Cases({
      {0.3, {0.977626246538296, 15}},
      {0.5, {0.938469807240813, 15}},
      {0.8, {0.8462873527504802, 16}},
      {1.6, {0.4554021676393806, 16}},
      {-5, {-0.1775967713143383, 16}},
  });
  testJ1Cases({
      {0.3, {0.148318816273104, 16}},
      {0.5, {0.2422684576748739, 16}},
      {0.8, {0.36884204609417, 16}},
      {1.6, {0.56989593526168, 15}},
      {-5, {0.327579137591465, 15}},
  });
  testNormCases({
      {{-0.3, -0.34, -0.98}, {1.079814798935447, 15}},
      {{0.3, 0.34, 0.98}, {1.079814798935447, 15}},
      {{0.5}, {0.5, 16}},
      {{23, 432, 23, 456, 23}, {629.4020972319682, 13}},
  });
  testNormcdfCases({
      {-5, {0.0000002866515718791939, 22}},
      {-3, {0.00134989803163009458, 20}},
      {0, {0.5, 16}},
      {1, {0.841344746068543, 15}},
      {5, {0.9999997133484281, 16}},
  });
  testNormcdfinvCases({
      {0.3, {-0.524400512708041, 15}},
      {0.5, {0, 37}},
      {0.8, {0.841621233572915, 15}},
  });
  testRnormCases({
      {{-0.3, -0.34, -0.98}, {0.926084733220795, 15}},
      {{0.3, 0.34, 0.98}, {0.926084733220795, 15}},
      {{0.5}, {2, 16}},
      {{23, 432, 23, 456, 23}, {0.001588809450108087, 18}},
  });
  testY0Cases({
      {0.3, {-0.8072735778045195, 16}},
      {0.5, {-0.4445187335067065, 16}},
      {0.8, {-0.0868022796566067, 16}},
      {1.6, {0.420426896415748, 15}},
      {5, {-0.308517625249034, 15}},
  });
  testY1Cases({
      {0.3, {-2.293105138388529, 15}},
      {0.5, {-1.471472392670243, 15}},
      {0.8, {-0.978144176683359, 15}},
      {1.6, {-0.3475780082651325, 16}},
      {5, {0.1478631433912269, 16}},
  });
  cout << "passed " << passed << "/" << passed + failed << " cases!" << endl;
  if (failed) {
    cout << "failed!" << endl;
  }
  return failed;
}
