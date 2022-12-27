// ====------ fft_set_workspace.cpp ------------------------- *- C++ -* ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdio>
#include <oneapi/mkl.hpp>
#include <dpct/fft_utils.hpp>

void foo() {
  dpct::fft::fft_engine *plan;
  dpct::fft::fft_type type = dpct::fft::fft_type::real_float_to_complex_float;
  size_t workSize;
  int batch = 1;
  plan = dpct::fft::fft_engine::create();
  plan->use_internal_workspace(1);
  plan->commit(&dpct::get_default_queue(), 3, type, batch, &workSize);
  void *workArea;
  plan->set_workspace(workArea);
  dpct::fft::fft_engine::destroy(plan);
}

int main() {
  try {
    foo();
  } catch (sycl::exception const &e) {
    std::cerr << "SYCL exception: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
