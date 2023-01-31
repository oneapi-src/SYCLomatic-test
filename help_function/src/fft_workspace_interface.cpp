// ====------ fft_workspace_interface.cpp ------------------- *- C++ -* ----===//
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

// Only test if this case can be compiled or not.

int main() {
  return 0;
}

void foo() {
  dpct::fft::fft_engine *plan;
  int rank;
  long long int *n_ll;
  long long int *inembed_ll;
  long long int istride_ll;
  long long int idist_ll;
  long long int *onembed_ll;
  long long int ostride_ll;
  long long int odist_ll;
  dpct::fft::fft_type type;
  long long int batch_ll;
  size_t *workSize;

  int *n;
  int *inembed;
  int istride;
  int idist;
  int *onembed;
  int ostride;
  int odist;
  int batch;

  plan->create();
  dpct::fft::fft_engine::estimate_size(n[0], type, batch, workSize);
  dpct::fft::fft_engine::estimate_size(n[0], n[1], type, workSize);
  dpct::fft::fft_engine::estimate_size(n[0], n[1], n[2], type, workSize);
  dpct::fft::fft_engine::estimate_size(rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize);
  dpct::fft::fft_engine::estimate_size(rank, n_ll, inembed_ll, istride_ll, idist_ll, onembed_ll, ostride_ll, odist_ll, type, batch_ll, workSize);

  plan->use_internal_workspace(0);
  plan->commit(&dpct::get_default_queue(), n[0], type, batch, workSize);
  plan->get_workspace_size(workSize);

  void *workArea;
  plan->set_workspace(workArea);
}
