// ====------ cub_device_histgram.cu----------------------- *- CUDA -* ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cub/cub.cuh>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <tuple>
#include <vector>

#define NUM_CHANNELS 4
#define NUM_ACTIVE_CHANNELS 3

void CanFail(cudaError_t E, const char *Fmt, ...) {
  if (E == cudaSuccess)
    return;
  va_list Ap;
  va_start(Ap, Fmt);
  vfprintf(stderr, Fmt, Ap);
  va_end(Ap);
  fprintf(stderr, "cuda error: %s\n", cudaGetErrorString(cudaGetLastError()));
  std::terminate();
}

#define CANTFAIL(E)                                                            \
  { CanFail(E, #E " Failed\n"); }

struct FreeOnExit {
  std::vector<void *> Ptrs;
  FreeOnExit(std::initializer_list<void *> Ps) : Ptrs(Ps) {}
  ~FreeOnExit() {
    for (void *P : Ptrs)
      if (P)
        CANTFAIL(cudaFree(P));
  }
};

template <typename T> T *Init(std::initializer_list<T> List) {
  T *Ptr = nullptr;
  size_t Bytes = sizeof(T) * List.size();
  CANTFAIL(cudaMallocManaged(&Ptr, Bytes));
  CANTFAIL(cudaMemcpy(Ptr, List.begin(), Bytes, cudaMemcpyHostToDevice));
  return Ptr;
}

template <typename T> T *Init(T InitVal, size_t N) {
  T *Ptr = nullptr;
  size_t Bytes = sizeof(T) * N;
  CANTFAIL(cudaMallocManaged(&Ptr, Bytes));
  std::fill(Ptr, Ptr + N, InitVal);
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

bool histgram_even() {
  int num_samples = 10;
  float *d_samples =
      Init<float>({2.2, 6.1, 7.1, 2.9, 3.5, 0.3, 2.9, 2.1, 6.1, 999.5});
  int *d_histogram = Init<int>({0, 0, 0, 0, 0, 0});
  int num_levels = 7;       // (seven level boundaries for six bins)
  float lower_level = 0.0;  // (lower sample value boundary of lowest bin)
  float upper_level = 12.0; // (upper sample value boundary of upper bin)

  int expect_histgram[] = {1, 5, 0, 3, 0, 0};

  // Determine temporary device storage requirements
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
                                      d_samples, d_histogram, num_levels,
                                      lower_level, upper_level, num_samples);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Compute histograms
  cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
                                      d_samples, d_histogram, num_levels,
                                      lower_level, upper_level, num_samples);

  cudaDeviceSynchronize();
  FreeOnExit Clean{d_temp_storage, d_samples, d_histogram};

  if (!std::equal(d_histogram, d_histogram + num_levels - 1, expect_histgram)) {
    std::cout << "Output Histgram:   "
              << Join(d_histogram, d_histogram + num_levels - 1) << std::endl;
    std::cout << "Expected Histgram: " << Join(expect_histgram) << std::endl;
    std::cout << "cub::DeviceHistogram::HistogramEven test failed."
              << std::endl;
    return false;
  }

  std::cout << "cub::DeviceHistogram::HistogramEven test pass." << std::endl;
  return true;
}

bool histgram_even_roi() {
  int num_row_samples = 5;
  int num_rows = 2;
  size_t row_stride_bytes = 7 * sizeof(float);
  float *d_samples = Init<float>(
      {2.2, 6.1, 7.1, 2.9, 3.5, 0, 0, 0.3, 2.9, 2.1, 6.1, 999.5, 0, 0});
  int *d_histogram = Init({0, 0, 0, 0, 0, 0});
  int num_levels = 7;       // (seven level boundaries for six bins)
  float lower_level = 0.0;  // (lower sample value boundary of lowest bin)
  float upper_level = 12.0; // (upper sample value boundary of upper bin)

  int expect_histgram[] = {1, 5, 0, 3, 0, 0};

  // Determine temporary device storage requirements
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::HistogramEven(
      d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels,
      lower_level, upper_level, num_row_samples, num_rows, row_stride_bytes);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Compute histograms
  cub::DeviceHistogram::HistogramEven(
      d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels,
      lower_level, upper_level, num_row_samples, num_rows, row_stride_bytes);

  cudaDeviceSynchronize();
  FreeOnExit Clean{d_temp_storage, d_samples, d_histogram};

  if (!std::equal(d_histogram, d_histogram + num_levels - 1, expect_histgram)) {
    std::cout << "Output Histgram:   "
              << Join(d_histogram, d_histogram + num_levels - 1) << std::endl;
    std::cout << "Expected Histgram: " << Join(expect_histgram) << std::endl;
    std::cout << "cub::DeviceHistogram::HistogramEven ROI test failed."
              << std::endl;
    return false;
  }

  std::cout << "cub::DeviceHistogram::HistogramEven ROI test pass."
            << std::endl;
  return true;
}

bool multi_histgram_even() {
  int num_pixels = 5;
  unsigned char *d_samples = Init<unsigned char>(
      {/*(*/ 2, 6, 7, 5 /*)*/, /*(*/ 3, 0, 2, 1 /*)*/,
       /*(*/ 7, 0, 6, 2 /*)*/,
       /*(*/ 0, 6, 7, 5 /*)*/, /*(*/ 3, 0, 2, 6 /*)*/});
  int *d_histogram[3] = {
      Init(0, 256), Init(0, 256),
      Init(0, 256)}; // e.g., three device pointers to three device buffers,
                     //       each allocated with 256 integer counters
  int num_levels[3] = {257, 257, 257};
  unsigned int lower_level[3] = {0, 0, 0};
  unsigned int upper_level[3] = {256, 256, 256};

  int expect_histgram[3][256] = {{1, 0, 1, 2, 0, 0, 0, 1},
                                 {3, 0, 0, 0, 0, 0, 2, 0},
                                 {0, 0, 2, 0, 0, 0, 1, 2}};

  // Determine temporary device storage requirements
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::MultiHistogramEven<4, 3>(
      d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels,
      lower_level, upper_level, num_pixels);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Compute histograms
  cub::DeviceHistogram::MultiHistogramEven<4, 3>(
      d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels,
      lower_level, upper_level, num_pixels);

  cudaDeviceSynchronize();
  FreeOnExit Clean{d_temp_storage, d_samples, d_histogram[0], d_histogram[1],
                   d_histogram[2]};

  for (int i = 0; i < 3; ++i) {
    if (!std::equal(d_histogram[i], d_histogram[i] + 256, expect_histgram[i])) {
      std::cout << "Output Histgram:   "
                << Join(d_histogram[i], d_histogram[i] + 256) << std::endl;
      std::cout << "Expected Histgram: " << Join(expect_histgram[i])
                << std::endl;
      std::cout << "cub::DeviceHistogram::MultiHistogramEven test failed."
                << std::endl;
      return false;
    }
  }

  std::cout << "cub::DeviceHistogram::MultiHistogramEven test pass."
            << std::endl;

  return true;
}

bool multi_histgram_even_roi() {
  int num_row_pixels = 3;
  int num_rows = 2;
  size_t row_stride_bytes = 4 * sizeof(unsigned char) * NUM_CHANNELS;
  unsigned char *d_samples = Init<unsigned char>(
      {/*(*/ 2, 6, 7, 5 /*)*/, /*(*/ 3, 0, 2, 1 /*)*/,
       /*(*/ 7, 0, 6, 2 /*)*/, /*(*/ 0, 0, 0, 0 /*)*/,
       /*(*/ 0, 6, 7, 5 /*)*/, /*(*/ 3, 0, 2, 6 /*)*/,
       /*(*/ 1, 1, 1, 1 /*)*/, /*(*/ 0, 0, 0, 0 /*)*/});
  int *d_histogram[3] = {
      Init(0, 256), Init(0, 256),
      Init(0, 256)}; // e.g., three device pointers to three device buffers,
                     //       each allocated with 256 integer counters
  int num_levels[3] = {257, 257, 257};
  unsigned int lower_level[3] = {0, 0, 0};
  unsigned int upper_level[3] = {256, 256, 256};

  int expect_histgram[3][256] = {{1, 1, 1, 2, 0, 0, 0, 1},
                                 {3, 1, 0, 0, 0, 0, 2, 0},
                                 {0, 1, 2, 0, 0, 0, 1, 2}};

  // Determine temporary device storage requirements
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::MultiHistogramEven<4, 3>(
      d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels,
      lower_level, upper_level, num_row_pixels, num_rows, row_stride_bytes);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Compute histograms
  cub::DeviceHistogram::MultiHistogramEven<4, 3>(
      d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels,
      lower_level, upper_level, num_row_pixels, num_rows, row_stride_bytes);
  cudaDeviceSynchronize();
  FreeOnExit Clean{d_temp_storage, d_samples, d_histogram[0], d_histogram[1],
                   d_histogram[2]};

  for (int i = 0; i < 3; ++i) {
    if (!std::equal(d_histogram[i], d_histogram[i] + 256, expect_histgram[i])) {
      std::cout << "Output Histgram:   "
                << Join(d_histogram[i], d_histogram[i] + 256) << std::endl;
      std::cout << "Expected Histgram: " << Join(expect_histgram[i])
                << std::endl;
      std::cout << "cub::DeviceHistogram::MultiHistogramEven ROI test failed."
                << std::endl;
      return false;
    }
  }

  std::cout << "cub::DeviceHistogram::MultiHistogramEven ROI test pass."
            << std::endl;

  return true;
}

bool histgram_range() {
  int num_samples = 10;
  float *d_samples =
      Init<float>({2.2, 6.0, 7.1, 2.9, 3.5, 0.3, 2.9, 2.0, 6.1, 999.5});
  int *d_histogram = Init({0, 0, 0, 0, 0, 0});
  int num_levels = 7; // (seven level boundaries for six bins)
  float *d_levels = Init<float>({0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0});

  int expect_histgram[] = {1, 5, 0, 3, 0, 0};

  // Determine temporary device storage requirements
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::HistogramRange(d_temp_storage, temp_storage_bytes,
                                       d_samples, d_histogram, num_levels,
                                       d_levels, num_samples);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Compute histograms
  cub::DeviceHistogram::HistogramRange(d_temp_storage, temp_storage_bytes,
                                       d_samples, d_histogram, num_levels,
                                       d_levels, num_samples);
  cudaDeviceSynchronize();
  FreeOnExit Clean{d_temp_storage, d_samples, d_histogram, d_levels};

  if (!std::equal(d_histogram, d_histogram + num_levels - 1, expect_histgram)) {
    std::cout << "Output Histgram:   "
              << Join(d_histogram, d_histogram + num_levels - 1) << std::endl;
    std::cout << "Expected Histgram: " << Join(expect_histgram) << std::endl;
    std::cout << "cub::DeviceHistogram::HistogramRange ROI test failed."
              << std::endl;
    return false;
  }

  std::cout << "cub::DeviceHistogram::HistogramRange ROI test pass."
            << std::endl;
  return true;
}

bool histgram_range_roi() {
  int num_row_samples = 5;
  int num_rows = 2;
  int row_stride_bytes = 7 * sizeof(float);
  float *d_samples = Init<float>(
      {2.2, 6.0, 7.1, 2.9, 3.5, 0, 0, 0.3, 2.9, 2.0, 6.1, 999.5, 0, 0});
  int *d_histogram = Init({0, 0, 0, 0, 0, 0});
  int num_levels = 7; // (seven level boundaries for six bins)
  float *d_levels = Init<float>({0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0});

  int expect_histgram[] = {1, 5, 0, 3, 0, 0};

  // Determine temporary device storage requirements
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::HistogramRange(
      d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels,
      d_levels, num_row_samples, num_rows, row_stride_bytes);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Compute histograms
  cub::DeviceHistogram::HistogramRange(
      d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels,
      d_levels, num_row_samples, num_rows, row_stride_bytes);
  cudaDeviceSynchronize();
  FreeOnExit Clean{d_temp_storage, d_samples, d_histogram, d_levels};

  if (!std::equal(d_histogram, d_histogram + num_levels - 1, expect_histgram)) {
    std::cout << "Output Histgram:   "
              << Join(d_histogram, d_histogram + num_levels - 1) << std::endl;
    std::cout << "Expected Histgram: " << Join(expect_histgram) << std::endl;
    std::cout << "cub::DeviceHistogram::HistogramRange ROI test failed."
              << std::endl;
    return false;
  }

  std::cout << "cub::DeviceHistogram::HistogramRange ROI test pass."
            << std::endl;
  return true;
}

bool multi_histgram_range() {
  int num_pixels = 5;
  unsigned char *d_samples = Init<unsigned char>(
      {/*(*/ 2, 6, 7, 5 /*)*/, /*(*/ 3, 0, 2, 1 /*)*/,
       /*(*/ 7, 0, 6, 2 /*)*/,
       /*(*/ 0, 6, 7, 5 /*)*/, /*(*/ 3, 0, 2, 6 /*)*/});
  unsigned int *d_histogram[3] = {Init(0U, 4), Init(0U, 4), Init(0U, 4)};
  int num_levels[3] = {5, 5, 5};
  unsigned int *d_levels[3] = {Init<unsigned int>({0, 2, 4, 6, 8}),
                               Init<unsigned int>({0, 2, 4, 6, 8}),
                               Init<unsigned int>({0, 2, 4, 6, 8})};

  unsigned int expect_histgram[3][4] = {
      {1, 3, 0, 1}, {3, 0, 0, 2}, {0, 2, 0, 3}};

  // Determine temporary device storage requirements
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::MultiHistogramRange<4, 3>(
      d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels,
      d_levels, num_pixels);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Compute histograms
  cub::DeviceHistogram::MultiHistogramRange<4, 3>(
      d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels,
      d_levels, num_pixels);
  cudaDeviceSynchronize();
  FreeOnExit Clean{d_temp_storage, d_samples,   d_histogram[0], d_histogram[1],
                   d_histogram[2], d_levels[0], d_levels[1],    d_levels[2]};

  for (int i = 0; i < 3; ++i) {
    if (!std::equal(d_histogram[i], d_histogram[i] + 4, expect_histgram[i])) {
      std::cout << "Output Histgram:   "
                << Join(d_histogram[i], d_histogram[i] + 4) << std::endl;
      std::cout << "Expected Histgram: " << Join(expect_histgram[i])
                << std::endl;
      std::cout << "cub::DeviceHistogram::MultiHistogramRange test failed."
                << std::endl;
      return false;
    }
  }

  std::cout << "cub::DeviceHistogram::MultiHistogramRange test pass."
            << std::endl;

  return true;
}

bool multi_histgram_range_roi() {
  int num_row_pixels = 3;
  int num_rows = 2;
  size_t row_stride_bytes = 4 * sizeof(unsigned char) * NUM_CHANNELS;
  // clang-format off
  unsigned char *d_samples = Init<unsigned char>(
      {/*(*/ 2, 6, 7, 5 /*)*/, /*(*/ 3, 0, 2, 1 /*)*/, /*(*/ 1, 1, 1, 1 /*)*/, /*(*/ 0, 0, 0, 0 /*)*/,
       /*(*/ 7, 0, 6, 2 /*)*/, /*(*/ 0, 6, 7, 5 /*)*/, /*(*/ 3, 0, 2, 6 /*)*/, /*(*/ 0, 0, 0, 0 /*)*/});
  // clang-format on
  int *d_histogram[3] = {Init(0, 4), Init(0, 4), Init(0, 4)};
  int num_levels[3] = {5, 5, 5};
  unsigned int *d_levels[3] = {Init<unsigned int>({0, 2, 4, 6, 8}),
                               Init<unsigned int>({0, 2, 4, 6, 8}),
                               Init<unsigned int>({0, 2, 4, 6, 8})};

  unsigned int expect_histgram[3][4] = {
      {2, 3, 0, 1}, {4, 0, 0, 2}, {1, 2, 0, 3}};

  // Determine temporary device storage requirements
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::MultiHistogramRange<4, 3>(
      d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels,
      d_levels, num_row_pixels, num_rows, row_stride_bytes);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Compute histograms
  cub::DeviceHistogram::MultiHistogramRange<4, 3>(
      d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels,
      d_levels, num_row_pixels, num_rows, row_stride_bytes);
  cudaDeviceSynchronize();
  FreeOnExit Clean{d_temp_storage, d_samples,   d_histogram[0], d_histogram[1],
                   d_histogram[2], d_levels[0], d_levels[1],    d_levels[2]};
  for (int i = 0; i < 3; ++i) {
    if (!std::equal(d_histogram[i], d_histogram[i] + 4, expect_histgram[i])) {
      std::cout << "Output Histgram:   "
                << Join(d_histogram[i], d_histogram[i] + 4) << std::endl;
      std::cout << "Expected Histgram: " << Join(expect_histgram[i])
                << std::endl;
      std::cout << "cub::DeviceHistogram::MultiHistogramRange ROI test failed."
                << std::endl;
      return false;
    }
  }

  std::cout << "cub::DeviceHistogram::MultiHistogramRange ROI test pass."
            << std::endl;
  return true;
}

bool histgram_even_stream() {
  int num_samples = 10;
  float *d_samples =
      Init<float>({2.2, 6.1, 7.1, 2.9, 3.5, 0.3, 2.9, 2.1, 6.1, 999.5});
  int *d_histogram = Init<int>({0, 0, 0, 0, 0, 0});
  int num_levels = 7;       // (seven level boundaries for six bins)
  float lower_level = 0.0;  // (lower sample value boundary of lowest bin)
  float upper_level = 12.0; // (upper sample value boundary of upper bin)

  int expect_histgram[] = {1, 5, 0, 3, 0, 0};
  cudaStream_t S;
  cudaStreamCreate(&S);

  // Determine temporary device storage requirements
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
                                      d_samples, d_histogram, num_levels,
                                      lower_level, upper_level, num_samples, S);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Compute histograms
  cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
                                      d_samples, d_histogram, num_levels,
                                      lower_level, upper_level, num_samples, S);

  cudaDeviceSynchronize();
  cudaStreamDestroy(S);
  FreeOnExit Clean{d_temp_storage, d_samples, d_histogram};

  if (!std::equal(d_histogram, d_histogram + num_levels - 1, expect_histgram)) {
    std::cout << "Output Histgram:   "
              << Join(d_histogram, d_histogram + num_levels - 1) << std::endl;
    std::cout << "Expected Histgram: " << Join(expect_histgram) << std::endl;
    std::cout << "cub::DeviceHistogram::HistogramEven(With custom cuda stream) "
                 "test failed."
              << std::endl;
    return false;
  }

  std::cout << "cub::DeviceHistogram::HistogramEven(With custom cuda stream) "
               "test pass."
            << std::endl;
  return true;
}

bool histgram_even_roi_stream() {
  int num_row_samples = 5;
  int num_rows = 2;
  size_t row_stride_bytes = 7 * sizeof(float);
  float *d_samples = Init<float>(
      {2.2, 6.1, 7.1, 2.9, 3.5, 0, 0, 0.3, 2.9, 2.1, 6.1, 999.5, 0, 0});
  int *d_histogram = Init({0, 0, 0, 0, 0, 0});
  int num_levels = 7;       // (seven level boundaries for six bins)
  float lower_level = 0.0;  // (lower sample value boundary of lowest bin)
  float upper_level = 12.0; // (upper sample value boundary of upper bin)

  int expect_histgram[] = {1, 5, 0, 3, 0, 0};
  cudaStream_t S;
  cudaStreamCreate(&S);
  // Determine temporary device storage requirements
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::HistogramEven(
      d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels,
      lower_level, upper_level, num_row_samples, num_rows, row_stride_bytes, S);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Compute histograms
  cub::DeviceHistogram::HistogramEven(
      d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels,
      lower_level, upper_level, num_row_samples, num_rows, row_stride_bytes, S);
  cudaDeviceSynchronize();
  cudaStreamDestroy(S);
  FreeOnExit Clean{d_temp_storage, d_samples, d_histogram};

  if (!std::equal(d_histogram, d_histogram + num_levels - 1, expect_histgram)) {
    std::cout << "Output Histgram:   "
              << Join(d_histogram, d_histogram + num_levels - 1) << std::endl;
    std::cout << "Expected Histgram: " << Join(expect_histgram) << std::endl;
    std::cout << "cub::DeviceHistogram::HistogramEven ROI(With custom cuda "
                 "stream) test failed."
              << std::endl;
    return false;
  }

  std::cout << "cub::DeviceHistogram::HistogramEven ROI(With custom cuda "
               "stream) test pass."
            << std::endl;
  return true;
}

bool multi_histgram_even_stream() {
  int num_pixels = 5;
  unsigned char *d_samples = Init<unsigned char>(
      {/*(*/ 2, 6, 7, 5 /*)*/, /*(*/ 3, 0, 2, 1 /*)*/,
       /*(*/ 7, 0, 6, 2 /*)*/,
       /*(*/ 0, 6, 7, 5 /*)*/, /*(*/ 3, 0, 2, 6 /*)*/});
  int *d_histogram[3] = {
      Init(0, 256), Init(0, 256),
      Init(0, 256)}; // e.g., three device pointers to three device buffers,
                     //       each allocated with 256 integer counters
  int num_levels[3] = {257, 257, 257};
  unsigned int lower_level[3] = {0, 0, 0};
  unsigned int upper_level[3] = {256, 256, 256};

  int expect_histgram[3][256] = {{1, 0, 1, 2, 0, 0, 0, 1},
                                 {3, 0, 0, 0, 0, 0, 2, 0},
                                 {0, 0, 2, 0, 0, 0, 1, 2}};
  cudaStream_t S;
  cudaStreamCreate(&S);

  // Determine temporary device storage requirements
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::MultiHistogramEven<4, 3>(
      d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels,
      lower_level, upper_level, num_pixels, S);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Compute histograms
  cub::DeviceHistogram::MultiHistogramEven<4, 3>(
      d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels,
      lower_level, upper_level, num_pixels, S);
  cudaDeviceSynchronize();
  cudaStreamDestroy(S);
  FreeOnExit Clean{d_temp_storage, d_samples, d_histogram[0], d_histogram[1],
                   d_histogram[2]};

  for (int i = 0; i < 3; ++i) {
    if (!std::equal(d_histogram[i], d_histogram[i] + 256, expect_histgram[i])) {
      std::cout << "Output Histgram:   "
                << Join(d_histogram[i], d_histogram[i] + 256) << std::endl;
      std::cout << "Expected Histgram: " << Join(expect_histgram[i])
                << std::endl;
      std::cout << "cub::DeviceHistogram::MultiHistogramEven(With custom cuda "
                   "stream) test failed."
                << std::endl;
      return false;
    }
  }

  std::cout << "cub::DeviceHistogram::MultiHistogramEven(With custom cuda "
               "stream) test pass."
            << std::endl;

  return true;
}

bool multi_histgram_even_roi_stream() {
  int num_row_pixels = 3;
  int num_rows = 2;
  size_t row_stride_bytes = 4 * sizeof(unsigned char) * NUM_CHANNELS;
  unsigned char *d_samples = Init<unsigned char>(
      {/*(*/ 2, 6, 7, 5 /*)*/, /*(*/ 3, 0, 2, 1 /*)*/,
       /*(*/ 7, 0, 6, 2 /*)*/, /*(*/ 0, 0, 0, 0 /*)*/,
       /*(*/ 0, 6, 7, 5 /*)*/, /*(*/ 3, 0, 2, 6 /*)*/,
       /*(*/ 1, 1, 1, 1 /*)*/, /*(*/ 0, 0, 0, 0 /*)*/});
  int *d_histogram[3] = {
      Init(0, 256), Init(0, 256),
      Init(0, 256)}; // e.g., three device pointers to three device buffers,
                     //       each allocated with 256 integer counters
  int num_levels[3] = {257, 257, 257};
  unsigned int lower_level[3] = {0, 0, 0};
  unsigned int upper_level[3] = {256, 256, 256};

  int expect_histgram[3][256] = {{1, 1, 1, 2, 0, 0, 0, 1},
                                 {3, 1, 0, 0, 0, 0, 2, 0},
                                 {0, 1, 2, 0, 0, 0, 1, 2}};
  cudaStream_t S;
  cudaStreamCreate(&S);
  // Determine temporary device storage requirements
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::MultiHistogramEven<4, 3>(
      d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels,
      lower_level, upper_level, num_row_pixels, num_rows, row_stride_bytes, S);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Compute histograms
  cub::DeviceHistogram::MultiHistogramEven<4, 3>(
      d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels,
      lower_level, upper_level, num_row_pixels, num_rows, row_stride_bytes, S);
  cudaDeviceSynchronize();
  cudaStreamDestroy(S);
  FreeOnExit Clean{d_temp_storage, d_samples, d_histogram[0], d_histogram[1],
                   d_histogram[2]};

  for (int i = 0; i < 3; ++i) {
    if (!std::equal(d_histogram[i], d_histogram[i] + 256, expect_histgram[i])) {
      std::cout << "Output Histgram:   "
                << Join(d_histogram[i], d_histogram[i] + 256) << std::endl;
      std::cout << "Expected Histgram: " << Join(expect_histgram[i])
                << std::endl;
      std::cout << "cub::DeviceHistogram::MultiHistogramEven ROI(With custom "
                   "cuda stream) test failed."
                << std::endl;
      return false;
    }
  }

  std::cout << "cub::DeviceHistogram::MultiHistogramEven ROI(With custom cuda "
               "stream) test pass."
            << std::endl;

  return true;
}

bool histgram_range_stream() {
  int num_samples = 10;
  float *d_samples =
      Init<float>({2.2, 6.0, 7.1, 2.9, 3.5, 0.3, 2.9, 2.0, 6.1, 999.5});
  int *d_histogram = Init({0, 0, 0, 0, 0, 0});
  int num_levels = 7; // (seven level boundaries for six bins)
  float *d_levels = Init<float>({0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0});
  int expect_histgram[] = {1, 5, 0, 3, 0, 0};
  cudaStream_t S;
  cudaStreamCreate(&S);

  // Determine temporary device storage requirements
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::HistogramRange(d_temp_storage, temp_storage_bytes,
                                       d_samples, d_histogram, num_levels,
                                       d_levels, num_samples, S);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Compute histograms
  cub::DeviceHistogram::HistogramRange(d_temp_storage, temp_storage_bytes,
                                       d_samples, d_histogram, num_levels,
                                       d_levels, num_samples, S);
  cudaDeviceSynchronize();
  cudaStreamDestroy(S);
  FreeOnExit Clean{d_temp_storage, d_samples, d_histogram, d_levels};

  if (!std::equal(d_histogram, d_histogram + num_levels - 1, expect_histgram)) {
    std::cout << "Output Histgram:   "
              << Join(d_histogram, d_histogram + num_levels - 1) << std::endl;
    std::cout << "Expected Histgram: " << Join(expect_histgram) << std::endl;
    std::cout << "cub::DeviceHistogram::HistogramRange ROI(With custom cuda "
                 "stream) test failed."
              << std::endl;
    return false;
  }

  std::cout << "cub::DeviceHistogram::HistogramRange ROI(With custom cuda "
               "stream) test pass."
            << std::endl;
  return true;
}

bool histgram_range_roi_stream() {
  int num_row_samples = 5;
  int num_rows = 2;
  int row_stride_bytes = 7 * sizeof(float);
  float *d_samples = Init<float>(
      {2.2, 6.0, 7.1, 2.9, 3.5, 0, 0, 0.3, 2.9, 2.0, 6.1, 999.5, 0, 0});
  int *d_histogram = Init({0, 0, 0, 0, 0, 0});
  int num_levels = 7; // (seven level boundaries for six bins)
  float *d_levels = Init<float>({0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0});
  int expect_histgram[] = {1, 5, 0, 3, 0, 0};
  cudaStream_t S;
  cudaStreamCreate(&S);
  // Determine temporary device storage requirements
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::HistogramRange(
      d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels,
      d_levels, num_row_samples, num_rows, row_stride_bytes, S);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Compute histograms
  cub::DeviceHistogram::HistogramRange(
      d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels,
      d_levels, num_row_samples, num_rows, row_stride_bytes, S);
  cudaDeviceSynchronize();
  cudaStreamDestroy(S);
  FreeOnExit Clean{d_temp_storage, d_samples, d_histogram, d_levels};

  if (!std::equal(d_histogram, d_histogram + num_levels - 1, expect_histgram)) {
    std::cout << "Output Histgram:   "
              << Join(d_histogram, d_histogram + num_levels - 1) << std::endl;
    std::cout << "Expected Histgram: " << Join(expect_histgram) << std::endl;
    std::cout << "cub::DeviceHistogram::HistogramRange ROI(With custom cuda "
                 "stream) test failed."
              << std::endl;
    return false;
  }

  std::cout << "cub::DeviceHistogram::HistogramRange ROI(With custom cuda "
               "stream) test pass."
            << std::endl;
  return true;
}

bool multi_histgram_range_stream() {
  int num_pixels = 5;
  unsigned char *d_samples = Init<unsigned char>(
      {/*(*/ 2, 6, 7, 5 /*)*/, /*(*/ 3, 0, 2, 1 /*)*/,
       /*(*/ 7, 0, 6, 2 /*)*/,
       /*(*/ 0, 6, 7, 5 /*)*/, /*(*/ 3, 0, 2, 6 /*)*/});
  unsigned int *d_histogram[3] = {Init(0U, 4), Init(0U, 4), Init(0U, 4)};
  int num_levels[3] = {5, 5, 5};
  unsigned int *d_levels[3] = {Init<unsigned int>({0, 2, 4, 6, 8}),
                               Init<unsigned int>({0, 2, 4, 6, 8}),
                               Init<unsigned int>({0, 2, 4, 6, 8})};

  unsigned int expect_histgram[3][4] = {
      {1, 3, 0, 1}, {3, 0, 0, 2}, {0, 2, 0, 3}};
  cudaStream_t S;
  cudaStreamCreate(&S);
  // Determine temporary device storage requirements
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::MultiHistogramRange<4, 3>(
      d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels,
      d_levels, num_pixels, S);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Compute histograms
  cub::DeviceHistogram::MultiHistogramRange<4, 3>(
      d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels,
      d_levels, num_pixels, S);
  cudaDeviceSynchronize();
  cudaStreamDestroy(S);
  FreeOnExit Clean{d_temp_storage, d_samples,   d_histogram[0], d_histogram[1],
                   d_histogram[2], d_levels[0], d_levels[1],    d_levels[2]};

  for (int i = 0; i < 3; ++i) {
    if (!std::equal(d_histogram[i], d_histogram[i] + 4, expect_histgram[i])) {
      std::cout << "Output Histgram:   "
                << Join(d_histogram[i], d_histogram[i] + 4) << std::endl;
      std::cout << "Expected Histgram: " << Join(expect_histgram[i])
                << std::endl;
      std::cout << "cub::DeviceHistogram::MultiHistogramRange(With custom cuda "
                   "stream) test failed."
                << std::endl;
      return false;
    }
  }

  std::cout << "cub::DeviceHistogram::MultiHistogramRange(With custom cuda "
               "stream) test pass."
            << std::endl;

  return true;
}

bool multi_histgram_range_roi_stream() {
  int num_row_pixels = 3;
  int num_rows = 2;
  size_t row_stride_bytes = 4 * sizeof(unsigned char) * NUM_CHANNELS;
  // clang-format off
  unsigned char *d_samples = Init<unsigned char>(
      {/*(*/ 2, 6, 7, 5 /*)*/, /*(*/ 3, 0, 2, 1 /*)*/, /*(*/ 1, 1, 1, 1 /*)*/, /*(*/ 0, 0, 0, 0 /*)*/,
       /*(*/ 7, 0, 6, 2 /*)*/, /*(*/ 0, 6, 7, 5 /*)*/, /*(*/ 3, 0, 2, 6 /*)*/, /*(*/ 0, 0, 0, 0 /*)*/});
  // clang-format on
  int *d_histogram[3] = {Init(0, 4), Init(0, 4), Init(0, 4)};
  int num_levels[3] = {5, 5, 5};
  unsigned int *d_levels[3] = {Init<unsigned int>({0, 2, 4, 6, 8}),
                               Init<unsigned int>({0, 2, 4, 6, 8}),
                               Init<unsigned int>({0, 2, 4, 6, 8})};

  unsigned int expect_histgram[3][4] = {
      {2, 3, 0, 1}, {4, 0, 0, 2}, {1, 2, 0, 3}};
  cudaStream_t S;
  cudaStreamCreate(&S);
  // Determine temporary device storage requirements
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::MultiHistogramRange<4, 3>(
      d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels,
      d_levels, num_row_pixels, num_rows, row_stride_bytes, S);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Compute histograms
  cub::DeviceHistogram::MultiHistogramRange<4, 3>(
      d_temp_storage, temp_storage_bytes, d_samples, d_histogram, num_levels,
      d_levels, num_row_pixels, num_rows, row_stride_bytes, S);
  cudaDeviceSynchronize();
  cudaStreamDestroy(S);
  FreeOnExit Clean{d_temp_storage, d_samples,   d_histogram[0], d_histogram[1],
                   d_histogram[2], d_levels[0], d_levels[1],    d_levels[2]};

  for (int i = 0; i < 3; ++i) {
    if (!std::equal(d_histogram[i], d_histogram[i] + 4, expect_histgram[i])) {
      std::cout << "Output Histgram:   "
                << Join(d_histogram[i], d_histogram[i] + 4) << std::endl;
      std::cout << "Expected Histgram: " << Join(expect_histgram[i])
                << std::endl;
      std::cout << "cub::DeviceHistogram::MultiHistogramRange(With custom cuda "
                   "stream) ROI test failed."
                << std::endl;
      return false;
    }
  }

  std::cout << "cub::DeviceHistogram::MultiHistogramRange(With custom cuda "
               "stream) ROI test pass."
            << std::endl;
  return true;
}

int main() {
  if (!histgram_even())
    return 1;
  if (!histgram_even_roi())
    return 1;
  if (!multi_histgram_even())
    return 1;
  if (!multi_histgram_even_roi())
    return 1;
  if (!histgram_range())
    return 1;
  if (!histgram_range_roi())
    return 1;
  if (!multi_histgram_range())
    return 1;
  if (!multi_histgram_range_roi())
    return 1;
  if (!histgram_even_stream())
    return 1;
  if (!histgram_even_roi_stream())
    return 1;
  if (!multi_histgram_even_stream())
    return 1;
  if (!multi_histgram_even_roi_stream())
    return 1;
  if (!histgram_range_stream())
    return 1;
  if (!histgram_range_roi_stream())
    return 1;
  if (!multi_histgram_range_stream())
    return 1;
  if (!multi_histgram_range_roi_stream())
    return 1;
  return 0;
}
