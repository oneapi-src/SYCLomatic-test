// ====------ math-drcp.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

__global__ void drcp(const double Input[1], double Output[4]) {
  Output[0] = __drcp_rd(*Input);
  Output[1] = __drcp_rn(*Input);
  Output[2] = __drcp_ru(*Input);
  Output[3] = __drcp_rz(*Input);
}

static uint64_t DoubleToInteger(double Value) {
  union TypePun {
    double f64;
    uint64_t u64;
  } Pun;
  Pun.f64 = Value;
  return Pun.u64;
}

bool RunTest(double TestValue) {
  const std::vector<std::string> TestFunctionNames = {"__drcp_rd", "__drcp_rn",
                                                      "__drcp_ru", "__drcp_rz"};

  double *Input;
  double *Output;

  cudaMallocManaged(&Input, sizeof(double));
  cudaMallocManaged(&Output, 4 * sizeof(double));

  *Input = TestValue;
  drcp<<<1, 1>>>(Input, Output);
  cudaDeviceSynchronize();

  uint64_t ExpectedResult = DoubleToInteger(1 / *Input);
  for (int i = 0; i < 4; i++) {
    uint64_t ActualResult = DoubleToInteger(Output[i]);

    // ensure that migrated drcp* intrinsics produce results that are at
    // most one-bit off (due to rounding) of standard double-precision division.
    // This ensures that migrated code still has double-precision accuracy.
    if (std::abs(static_cast<int>(ExpectedResult & 0xFF) -
                 static_cast<int>(ActualResult & 0xFF)) > 1) {
      std::cerr << TestFunctionNames[i] << " TEST FAILED" << std::endl;
      std::cerr << "TestValue: " << TestValue << std::endl;
      std::cerr << "ExpectedResult: " << std::hex << ExpectedResult
                << std::endl;
      std::cerr << "ActualResult: " << std::hex << ActualResult << std::endl;
      return false;
    }
  }

  cudaFree(Input);
  cudaFree(Output);

  return true;
}

int main() {
  const std::vector<double> TestValues = {
      1.111111111, 123.456, 3.1416, 0.3, 0.33333333, 123456789.123456789};
  for (const auto &Value : TestValues) {
    bool TestPassed = RunTest(Value);
    if (TestPassed == false) {
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}
