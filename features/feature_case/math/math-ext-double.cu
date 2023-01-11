// ====------ math-ext-double.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <iostream>
#include <vector>

using namespace std;

int ret = 0;

void check(bool IsFailed) {
  if (IsFailed) {
    std::cout << " ---- failed" << std::endl;
    ret++;
  } else {
    std::cout << " ---- passed" << std::endl;
  }
}

__global__ void _erfcinv(double *const DeviceResult, double Input) {
  *DeviceResult = erfcinv(Input);
}

void testErfcinv(double *const DeviceResult, double Input) {
  _erfcinv<<<1, 1>>>(DeviceResult, Input);
  cudaDeviceSynchronize();
  //  auto HostResult = erfcinv(Input);
  //  if (HostResult > *DeviceResult + 0.000001 ||
  //      HostResult < *DeviceResult - 0.000001) {
  //    std::cout << "erfcinv: host result (" << HostResult
  //              << ") is not equals to device result (" << *DeviceResult
  //              << ") ---- failed" << std::endl;
  //    ret++;
  //  }
}

void testErfcinvCases(const vector<pair<double, double>> &TestCases) {
  double *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  // Boundary values.
  testErfcinv(DeviceResult, 0);
  std::cout << "erfcinv(" << 0 << ") = " << *DeviceResult << " (expect inf)";
  check(*DeviceResult < 999999.9);
  testErfcinv(DeviceResult, 2);
  std::cout << "erfcinv(" << 2 << ") = " << *DeviceResult << " (expect -inf)";
  check(*DeviceResult > -999999.9);
  // Other test values.
  for (const auto &TestCase : TestCases) {
    testErfcinv(DeviceResult, TestCase.first);
    std::cout << "erfcinv(" << TestCase.first << ") = " << *DeviceResult
              << " (expect " << TestCase.second - 0.000001 << " ~ "
              << TestCase.second + 0.000001 << ")";
    check(*DeviceResult < TestCase.second - 0.000001 ||
          *DeviceResult > TestCase.second + 0.000001);
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

void testErfinvCases(const vector<pair<double, double>> &TestCases) {
  double *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  // Boundary values.
  testErfinv(DeviceResult, -1);
  std::cout << "erfinv(" << -1 << ") = " << *DeviceResult << " (expect -inf)";
  check(*DeviceResult > -999999.9);
  testErfinv(DeviceResult, 1);
  std::cout << "erfinv(" << 1 << ") = " << *DeviceResult << " (expect inf)";
  check(*DeviceResult < 999999.9);
  // Other test values.
  for (const auto &TestCase : TestCases) {
    testErfinv(DeviceResult, TestCase.first);
    std::cout << "erfinv(" << TestCase.first << ") = " << *DeviceResult
              << " (expect " << TestCase.second - 0.000001 << " ~ "
              << TestCase.second + 0.000001 << ")";
    check(*DeviceResult < TestCase.second - 0.000001 ||
          *DeviceResult > TestCase.second + 0.000001);
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

void testNormcdfCases(const vector<pair<double, double>> &TestCases) {
  double *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  // Other test values.
  for (const auto &TestCase : TestCases) {
    testNormcdf(DeviceResult, TestCase.first);
    std::cout << "normcdf(" << TestCase.first << ") = " << *DeviceResult
              << " (expect " << TestCase.second - 0.000001 << " ~ "
              << TestCase.second + 0.000001 << ")";
    check(*DeviceResult < TestCase.second - 0.000001 ||
          *DeviceResult > TestCase.second + 0.000001);
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

void testNormcdfinvCases(const vector<pair<double, double>> &TestCases) {
  double *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  // Boundary values.
  testNormcdfinv(DeviceResult, 0);
  std::cout << "normcdfinv(" << 0 << ") = " << *DeviceResult
            << " (expect -inf)";
  check(*DeviceResult > -999999.9);
  testNormcdfinv(DeviceResult, 1);
  std::cout << "normcdfinv(" << 1 << ") = " << *DeviceResult << " (expect inf)";
  check(*DeviceResult < 999999.9);
  // Other test values.
  for (const auto &TestCase : TestCases) {
    testNormcdfinv(DeviceResult, TestCase.first);
    std::cout << "normcdfinv(" << TestCase.first << ") = " << *DeviceResult
              << " (expect " << TestCase.second - 0.000001 << " ~ "
              << TestCase.second + 0.000001 << ")";
    check(*DeviceResult < TestCase.second - 0.000001 ||
          *DeviceResult > TestCase.second + 0.000001);
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

void testNormCases(const vector<pair<vector<double>, double>> &TestCases) {
  double *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  // Other test values.
  for (const auto &TestCase : TestCases) {
    double *Input;
    cudaMallocManaged(&Input, TestCase.first.size() * sizeof(*Input));
    for (size_t i = 0; i < TestCase.first.size(); ++i) {
      // Notice: cannot set value from host!
      setVecValue<<<1, 1>>>(Input + i, TestCase.first[i]);
      cudaDeviceSynchronize();
    }
    testNorm(DeviceResult, TestCase.first.size(), Input);
    std::cout << "norm(" << TestCase.first.size() << ", &{";
    for (size_t i = 0; i < TestCase.first.size() - 1; ++i) {
      std::cout << TestCase.first[i] << ", ";
    }
    std::cout << TestCase.first.back() << "}) = " << *DeviceResult
              << " (expect " << TestCase.second - 0.001 << " ~ "
              << TestCase.second + 0.001 << ")";
    check(*DeviceResult < TestCase.second - 0.001 ||
          *DeviceResult > TestCase.second + 0.001);
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

void testRnormCases(const vector<pair<vector<double>, double>> &TestCases) {
  double *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  // Other test values.
  for (const auto &TestCase : TestCases) {
    double *Input;
    cudaMallocManaged(&Input, TestCase.first.size() * sizeof(*Input));
    for (size_t i = 0; i < TestCase.first.size(); ++i) {
      // Notice: cannot set value from host!
      setVecValue<<<1, 1>>>(Input + i, TestCase.first[i]);
      cudaDeviceSynchronize();
    }
    testRnorm(DeviceResult, TestCase.first.size(), Input);
    std::cout << "rnorm(" << TestCase.first.size() << ", &{";
    for (size_t i = 0; i < TestCase.first.size() - 1; ++i) {
      std::cout << TestCase.first[i] << ", ";
    }
    std::cout << TestCase.first.back() << "}) = " << *DeviceResult
              << " (expect " << TestCase.second - 0.000001 << " ~ "
              << TestCase.second + 0.000001 << ")";
    check(*DeviceResult < TestCase.second - 0.000001 ||
          *DeviceResult > TestCase.second + 0.000001);
  }
}

int main() {
  testErfcinvCases({
      {0.3, 0.732869},
      {0.5, 0.476936},
      {0.8, 0.179143},
      {1.6, -0.595116},
  });
  testErfinvCases({
      {-0.3, -0.272463},
      {-0.5, -0.476937},
      {0, 0},
      {0.5, 0.476936},
  });
  testNormcdfCases({
      {-5, 0},
      {-3, 0.001350},
      {0, 0.5},
      {1, 0.841345},
      {5, 1},
  });
  testNormcdfinvCases({
      {0.3, -0.524401},
      {0.5, 0},
      {0.8, 0.841621},
  });
  testNormCases({
      {{-0.3, -0.34, -0.98}, 1.07981},
      {{0.3, 0.34, 0.98}, 1.07981},
      {{0.5}, 0.5},
      {{23, 432, 23, 456, 23}, 629.402},
  });
  testRnormCases({
      {{-0.3, -0.34, -0.98}, 0.926085},
      {{0.3, 0.34, 0.98}, 0.926085},
      {{0.5}, 2},
      {{23, 432, 23, 456, 23}, 0.00158881},
  });
  std::cout << "ret = " << ret << std::endl;
  return ret;
}
