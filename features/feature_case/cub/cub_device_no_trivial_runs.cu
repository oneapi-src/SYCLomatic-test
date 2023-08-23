// ====------ cub_device_no_trivial_runs.cu ---------------- *- CUDA -* ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <cub/cub.cuh>

#include <algorithm>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <iostream>
#include <sstream>

void CanFail(cudaError_t E, const char *Fmt, ...) {
  if (E == cudaSuccess)
    return;
  va_list Ap;
  va_start(Ap, Fmt);
  vfprintf(stderr, Fmt, Ap);
  va_end(Ap);
  std::terminate();
}

#define CANTFAIL(E)                                                            \
  { CanFail(E, #E " Failed\n"); }

template <typename T> T *Init(std::initializer_list<T> List) {
  T *Ptr = nullptr;
  size_t Bytes = sizeof(T) * List.size();
  CANTFAIL(cudaMallocManaged(&Ptr, Bytes));
  CANTFAIL(cudaMemcpy(Ptr, List.begin(), Bytes, cudaMemcpyHostToDevice));
  return Ptr;
}

template <typename T> std::string Join(T *Begin, T *End) {
  std::stringstream OS;
  OS << "[";
  for (auto I = Begin; I != End; ++I) {
    OS << *I << (I == End - 1 ? "" : ", ");
  }
  OS << "]";
  return OS.str();
}

template <typename T, size_t N> std::string Join(T (&Arr)[N]) {
  return Join(std::begin(Arr), std::end(Arr));
}

int main() {

  int num_items = 8;
  int *d_in = Init({0, 2, 2, 9, 5, 5, 5, 8});
  int *d_offsets_out = Init({0, 0, 0, 0, 0, 0, 0, 0});
  int *d_lengths_out = Init({0, 0, 0, 0, 0, 0, 0, 0});
  int *d_num_runs_out = Init({0});

  int offsets[] = {1, 4};
  int lengths[] = {2, 3};
  int runs_out = 2;

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceRunLengthEncode::NonTrivialRuns(d_temp_storage, temp_storage_bytes,
                                             d_in, d_offsets_out, d_lengths_out,
                                             d_num_runs_out, num_items);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceRunLengthEncode::NonTrivialRuns(d_temp_storage, temp_storage_bytes,
                                             d_in, d_offsets_out, d_lengths_out,
                                             d_num_runs_out, num_items);
  cudaDeviceSynchronize();

  if (*d_num_runs_out != runs_out) {
    std::cerr << "Expected d_num_runs_out = 2, but got " << *d_num_runs_out
              << "\n";
    return 1;
  }

  if (!std::equal(offsets, offsets + runs_out, d_offsets_out)) {
    std::cerr << "Expected d_offsets_out = " << Join(offsets) << ", but got "
              << Join(d_offsets_out, d_offsets_out + runs_out) << "\n";
    return 1;
  }

  if (!std::equal(lengths, lengths + runs_out, d_lengths_out)) {
    std::cerr << "Expected d_lengths_out = " << Join(lengths) << ", but got "
              << Join(d_lengths_out, d_lengths_out + runs_out) << "\n";
    return 1;
  }

  std::cout << "cub::DeviceRunLengthEncode::NonTrivialRuns PASS\n";

  return 0;
}
