// ====------ math-ext-float.cu---------- *- CUDA -* ----===////
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

__global__ void _erfcinvf(float *const DeviceResult, float Input) {
  *DeviceResult = erfcinvf(Input);
}

void testErfcinvf(float *const DeviceResult, float Input) {
  _erfcinvf<<<1, 1>>>(DeviceResult, Input);
  cudaDeviceSynchronize();
  //  auto HostResult = erfcinvf(Input);
  //  if (HostResult > *DeviceResult + 0.000001 ||
  //      HostResult < *DeviceResult - 0.000001) {
  //    std::cout << "erfcinvf: host result (" << HostResult
  //              << ") is not equals to device result (" << *DeviceResult
  //              << ") ---- failed" << std::endl;
  //    ret++;
  //  }
}

void testErfcinvfCases(const vector<pair<float, float>> &TestCases) {
  float *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  // Boundary values.
  testErfcinvf(DeviceResult, 0);
  std::cout << "erfcinvf(" << 0 << ") = " << *DeviceResult << " (expect inf)";
  check(*DeviceResult < 999999.9);
  testErfcinvf(DeviceResult, 2);
  std::cout << "erfcinvf(" << 2 << ") = " << *DeviceResult << " (expect -inf)";
  check(*DeviceResult > -999999.9);
  // Other test values.
  for (const auto &TestCase : TestCases) {
    testErfcinvf(DeviceResult, TestCase.first);
    std::cout << "erfcinvf(" << TestCase.first << ") = " << *DeviceResult
              << " (expect " << TestCase.second - 0.000001 << " ~ "
              << TestCase.second + 0.000001 << ")";
    check(*DeviceResult < TestCase.second - 0.000001 ||
          *DeviceResult > TestCase.second + 0.000001);
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

void testErfinvfCases(const vector<pair<float, float>> &TestCases) {
  float *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  // Boundary values.
  testErfinvf(DeviceResult, -1);
  std::cout << "erfinvf(" << -1 << ") = " << *DeviceResult << " (expect -inf)";
  check(*DeviceResult > -999999.9);
  testErfinvf(DeviceResult, 1);
  std::cout << "erfinvf(" << 1 << ") = " << *DeviceResult << " (expect inf)";
  check(*DeviceResult < 999999.9);
  // Other test values.
  for (const auto &TestCase : TestCases) {
    testErfinvf(DeviceResult, TestCase.first);
    std::cout << "erfinvf(" << TestCase.first << ") = " << *DeviceResult
              << " (expect " << TestCase.second - 0.000001 << " ~ "
              << TestCase.second + 0.000001 << ")";
    check(*DeviceResult < TestCase.second - 0.000001 ||
          *DeviceResult > TestCase.second + 0.000001);
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

void testNormcdffCases(const vector<pair<float, float>> &TestCases) {
  float *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  // Other test values.
  for (const auto &TestCase : TestCases) {
    testNormcdff(DeviceResult, TestCase.first);
    std::cout << "normcdff(" << TestCase.first << ") = " << *DeviceResult
              << " (expect " << TestCase.second - 0.000001 << " ~ "
              << TestCase.second + 0.000001 << ")";
    check(*DeviceResult < TestCase.second - 0.000001 ||
          *DeviceResult > TestCase.second + 0.000001);
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

void testNormcdfinvfCases(const vector<pair<float, float>> &TestCases) {
  float *DeviceResult;
  cudaMallocManaged(&DeviceResult, sizeof(*DeviceResult));
  // Boundary values.
  testNormcdfinvf(DeviceResult, 0);
  std::cout << "normcdfinvf(" << 0 << ") = " << *DeviceResult
            << " (expect -inf)";
  check(*DeviceResult > -999999.9);
  testNormcdfinvf(DeviceResult, 1);
  std::cout << "normcdfinvf(" << 1 << ") = " << *DeviceResult
            << " (expect inf)";
  check(*DeviceResult < 999999.9);
  // Other test values.
  for (const auto &TestCase : TestCases) {
    testNormcdfinvf(DeviceResult, TestCase.first);
    std::cout << "normcdfinvf(" << TestCase.first << ") = " << *DeviceResult
              << " (expect " << TestCase.second - 0.000001 << " ~ "
              << TestCase.second + 0.000001 << ")";
    check(*DeviceResult < TestCase.second - 0.000001 ||
          *DeviceResult > TestCase.second + 0.000001);
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

void testNormfCases(const vector<pair<vector<float>, float>> &TestCases) {
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
    std::cout << "normf(" << TestCase.first.size() << ", &{";
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

__global__ void _rnormf(float *const DeviceResult, int Input1,
                        const float *Input2) {
  *DeviceResult = rnormf(Input1, Input2);
}

void testRnormf(float *const DeviceResult, int Input1, const float *Input2) {
  _rnormf<<<1, 1>>>(DeviceResult, Input1, Input2);
  cudaDeviceSynchronize();
  // Call from host.
}

void testRnormfCases(const vector<pair<vector<float>, float>> &TestCases) {
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
    std::cout << "rnormf(" << TestCase.first.size() << ", &{";
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
  testErfcinvfCases({
      {0.3, 0.732869},
      {0.5, 0.476936},
      {0.8, 0.179143},
      {1.6, -0.595116},
  });
  testErfinvfCases({
      {-0.3, -0.272463},
      {-0.5, -0.476937},
      {0, 0},
      {0.5, 0.476936},
  });
  testNormcdffCases({
      {-5, 0},
      {-3, 0.001350},
      {0, 0.5},
      {1, 0.841345},
      {5, 1},
  });
  testNormcdfinvfCases({
      {0.3, -0.524401},
      {0.5, 0},
      {0.8, 0.841621},
  });
  testNormfCases({
      {{-0.3, -0.34, -0.98}, 1.07981},
      {{0.3, 0.34, 0.98}, 1.07981},
      {{0.5}, 0.5},
      {{23, 432, 23, 456, 23}, 629.402},
  });
  testRnormfCases({
      {{-0.3, -0.34, -0.98}, 0.926085},
      {{0.3, 0.34, 0.98}, 0.926085},
      {{0.5}, 2},
      {{23, 432, 23, 456, 23}, 0.00158881},
  });
  std::cout << "ret = " << ret << std::endl;
  return ret;
}
