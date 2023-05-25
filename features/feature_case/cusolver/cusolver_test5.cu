// ===------ cusolver_test5.cu ------------------------------*- CUDA -*-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include "cusolverDn.h"

#include <cmath>
#include <vector>
#include <cstdio>
#include <complex>

template<class d_data_t>
struct Data {
  float *h_data;
  d_data_t *d_data;
  int element_num;
  Data(int element_num) : element_num(element_num) {
    h_data = (float*)malloc(sizeof(float) * element_num);
    memset(h_data, 0, sizeof(float) * element_num);
    cudaMalloc(&d_data, sizeof(d_data_t) * element_num);
    cudaMemset(d_data, 0, sizeof(d_data_t) * element_num);
  }
  Data(float* input_data, int element_num) : element_num(element_num) {
    h_data = (float*)malloc(sizeof(float) * element_num);
    cudaMalloc(&d_data, sizeof(d_data_t) * element_num);
    cudaMemset(d_data, 0, sizeof(d_data_t) * element_num);
    memcpy(h_data, input_data, sizeof(float) * element_num);
  }
  ~Data() {
    free(h_data);
    cudaFree(d_data);
  }
  void H2D() {
    d_data_t* h_temp = (d_data_t*)malloc(sizeof(d_data_t) * element_num);
    memset(h_temp, 0, sizeof(d_data_t) * element_num);
    from_float_convert(h_data, h_temp);
    cudaMemcpy(d_data, h_temp, sizeof(d_data_t) * element_num, cudaMemcpyHostToDevice);
    free(h_temp);
  }
  void D2H() {
    d_data_t* h_temp = (d_data_t*)malloc(sizeof(d_data_t) * element_num);
    memset(h_temp, 0, sizeof(d_data_t) * element_num);
    cudaMemcpy(h_temp, d_data, sizeof(d_data_t) * element_num, cudaMemcpyDeviceToHost);
    to_float_convert(h_temp, h_data);
    free(h_temp);
  }
private:
  inline void from_float_convert(float* in, d_data_t* out) {
    for (int i = 0; i < element_num; i++)
      out[i] = in[i];
  }
  inline void to_float_convert(d_data_t* in, float* out) {
    for (int i = 0; i < element_num; i++)
      out[i] = in[i];
  }
};
template <>
inline void Data<float2>::from_float_convert(float* in, float2* out) {
  for (int i = 0; i < element_num; i++)
    out[i].x = in[i];
}
template <>
inline void Data<double2>::from_float_convert(float* in, double2* out) {
  for (int i = 0; i < element_num; i++)
    out[i].x = in[i];
}

template <>
inline void Data<float2>::to_float_convert(float2* in, float* out) {
  for (int i = 0; i < element_num; i++)
    out[i] = in[i].x;
}
template <>
inline void Data<double2>::to_float_convert(double2* in, float* out) {
  for (int i = 0; i < element_num; i++)
    out[i] = in[i].x;
}

bool compare_result(float* expect, float* result, int element_num) {
  for (int i = 0; i < element_num; i++) {
    if (std::abs(result[i]-expect[i]) >= 0.05) {
      return false;
    }
  }
  return true;
}

bool compare_result(int* expect, int* result, int element_num) {
  for (int i = 0; i < element_num; i++) {
    if (result[i] != expect[i]) {
      return false;
    }
  }
  return true;
}

bool compare_result(int64_t* expect, int64_t* result, int element_num) {
  for (int i = 0; i < element_num; i++) {
    if (result[i] != expect[i]) {
      return false;
    }
  }
  return true;
}

bool compare_result(float* expect, float* result, std::vector<int> indices) {
  for (int i = 0; i < indices.size(); i++) {
    if (std::abs(result[indices[i]]-expect[indices[i]]) >= 0.05) {
      return false;
    }
  }
  return true;
}

bool test_passed = true;

void test_cusolverDnXtrtri() {
  std::vector<float> a = {1, 2, 2, 4};
  Data<float> a_s(a.data(), 4);
  Data<double> a_d(a.data(), 4);
  Data<float2> a_c(a.data(), 4);
  Data<double2> a_z(a.data(), 4);

  cusolverDnHandle_t handle;
  cusolverDnCreate(&handle);

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();

  size_t lwork_s;
  size_t lwork_d;
  size_t lwork_c;
  size_t lwork_z;
  size_t lwork_host_s;
  size_t lwork_host_d;
  size_t lwork_host_c;
  size_t lwork_host_z;

  cusolverDnXtrtri_bufferSize(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_NON_UNIT, 2, CUDA_R_32F, a_s.d_data, 2, &lwork_s, &lwork_host_s);
  cusolverDnXtrtri_bufferSize(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_NON_UNIT, 2, CUDA_R_64F, a_d.d_data, 2, &lwork_d, &lwork_host_d);
  cusolverDnXtrtri_bufferSize(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_NON_UNIT, 2, CUDA_C_32F, a_c.d_data, 2, &lwork_c, &lwork_host_c);
  cusolverDnXtrtri_bufferSize(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_NON_UNIT, 2, CUDA_C_64F, a_z.d_data, 2, &lwork_z, &lwork_host_z);

  void* device_ws_s;
  void* device_ws_d;
  void* device_ws_c;
  void* device_ws_z;
  cudaMalloc(&device_ws_s, lwork_s);
  cudaMalloc(&device_ws_d, lwork_d);
  cudaMalloc(&device_ws_c, lwork_c);
  cudaMalloc(&device_ws_z, lwork_z);
  void* host_ws_s;
  void* host_ws_d;
  void* host_ws_c;
  void* host_ws_z;
  host_ws_s = malloc(lwork_host_s);
  host_ws_d = malloc(lwork_host_d);
  host_ws_c = malloc(lwork_host_c);
  host_ws_z = malloc(lwork_host_z);

  int *info;
  cudaMalloc(&info, sizeof(int));

  cusolverDnXtrtri(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_NON_UNIT, 2, CUDA_R_32F, a_s.d_data, 2, device_ws_s, lwork_s, host_ws_s, lwork_host_s, info);
  cusolverDnXtrtri(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_NON_UNIT, 2, CUDA_R_64F, a_d.d_data, 2, device_ws_d, lwork_d, host_ws_d, lwork_host_d, info);
  cusolverDnXtrtri(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_NON_UNIT, 2, CUDA_C_32F, a_c.d_data, 2, device_ws_c, lwork_c, host_ws_c, lwork_host_c, info);
  cusolverDnXtrtri(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_NON_UNIT, 2, CUDA_C_64F, a_z.d_data, 2, device_ws_z, lwork_z, host_ws_z, lwork_host_z, info);

  a_s.D2H();
  a_d.D2H();
  a_c.D2H();
  a_z.D2H();

  cudaStreamSynchronize(0);

  cusolverDnDestroy(handle);
  cudaFree(device_ws_s);
  cudaFree(device_ws_d);
  cudaFree(device_ws_c);
  cudaFree(device_ws_z);
  free(host_ws_s);
  free(host_ws_d);
  free(host_ws_c);
  free(host_ws_z);
  cudaFree(info);

  printf("a_s:%f,%f,%f,%f\n", a_s.h_data[0], a_s.h_data[1], a_s.h_data[2], a_s.h_data[3]);

  float expect_a[4] = {1.000000,2.000000,-0.500000,0.250000};
  if (compare_result(expect_a, a_s.h_data, 4) &&
      compare_result(expect_a, a_d.h_data, 4) &&
      compare_result(expect_a, a_c.h_data, 4) &&
      compare_result(expect_a, a_z.h_data, 4))
    printf("DnXtrtri pass\n");
  else {
    printf("DnXtrtri fail\n");
    test_passed = false;
  }
}

int main() {
#ifndef DPCT_USM_LEVEL_NONE
  test_cusolverDnXtrtri();
#endif

  if (test_passed)
    return 0;
  return -1;
}
