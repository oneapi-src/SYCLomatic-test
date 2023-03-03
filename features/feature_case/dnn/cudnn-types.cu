// ====------ cudnn-types.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <cudnn.h>
#include <iostream>

int main() {
  int did_fail = 0;
  auto check_null_assignment = [&](auto &&arg, auto x) {
    arg = nullptr;
    if (arg) {
      std::cout << x << ": null assignment failed\n";
      did_fail = 1;
    }
  };
  {
    cudnnHandle_t d;
    cudnnCreate(&d);
    check_null_assignment(d, "cudnnHandle_t");
  }
  {
    cudnnTensorDescriptor_t d;
    cudnnCreateTensorDescriptor(&d);
    check_null_assignment(d, "cudnnTensorDescriptor_t");
  }
  {
    cudnnConvolutionDescriptor_t d;
    cudnnCreateConvolutionDescriptor(&d);
    check_null_assignment(d, "cudnnConvolutionDescriptor_t");
  }
  if (!did_fail)
    std::cout << "null assignment pass\n";
  return did_fail;
}
