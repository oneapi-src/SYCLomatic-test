// ===------ cusolver_test1.cu ------------------------------*- CUDA -*-----===//
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

bool compare_result(float* expect, float* result, std::vector<int> indices) {
  for (int i = 0; i < indices.size(); i++) {
    if (std::abs(result[indices[i]]-expect[indices[i]]) >= 0.05) {
      return false;
    }
  }
  return true;
}

bool test_passed = true;

void test_cusolverDnTsygvd() {
  std::vector<float> a = {1, 2, 3, 2, 1, 2, 3, 2, 1};
  std::vector<float> b = {2, -1, 0, -1, 2, -1, 0, -1, 2};
  Data<float> a_s(a.data(), 9);
  Data<double> a_d(a.data(), 9);
  Data<float> b_s(b.data(), 9);
  Data<double> b_d(b.data(), 9);
  Data<float> w_s(3);
  Data<double> w_d(3);

  cusolverDnHandle_t handle;
  cusolverDnCreate(&handle);

  a_s.H2D();
  a_d.H2D();
  b_s.H2D();
  b_d.H2D();

  int lwork_s;
  int lwork_d;
  cusolverDnSsygvd_bufferSize(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_s.d_data, 3, b_s.d_data, 3, w_s.d_data, &lwork_s);
  cusolverDnDsygvd_bufferSize(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_d.d_data, 3, b_d.d_data, 3, w_d.d_data, &lwork_d);

  float* work_s;
  double* work_d;
  int *devInfo;
  cudaMalloc(&work_s, sizeof(float) * lwork_s);
  cudaMalloc(&work_d, sizeof(double) * lwork_d);
  cudaMalloc(&devInfo, sizeof(int));

  cusolverDnSsygvd(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_s.d_data, 3, b_s.d_data, 3, w_s.d_data, work_s, lwork_s, devInfo);
  cusolverDnDsygvd(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_d.d_data, 3, b_d.d_data, 3, w_d.d_data, work_d, lwork_d, devInfo);

  a_s.D2H();
  a_d.D2H();
  b_s.D2H();
  b_d.D2H();
  w_s.D2H();
  w_d.D2H();

  cudaStreamSynchronize(0);

  cusolverDnDestroy(handle);
  cudaFree(work_s);
  cudaFree(work_d);
  cudaFree(devInfo);

  float expect_a[9] = {0.500000,-0.000000,-0.500000,0.194937,-0.484769,0.194937,0.679705,0.874642,0.679705};
  float expect_b[9] = {1.414214,-1.000000,0.000000,-0.707107,1.224745,-1.000000,0.000000,-0.816497,1.154701};
  float expect_w[3] = {-1.000000,-0.216991,9.216990};
  if (compare_result(expect_a, a_s.h_data, 9)
      && compare_result(expect_b, b_s.h_data, 9)
      && compare_result(expect_w, w_s.h_data, 3)
      && compare_result(expect_a, a_d.h_data, 9)
      && compare_result(expect_b, b_d.h_data, 9)
      && compare_result(expect_w, w_d.h_data, 3))
    printf("DnTsygvd pass\n");
  else {
    printf("DnTsygvd fail\n");
    test_passed = false;
  }
}

void test_cusolverDnThegvd() {
  std::vector<float> a = {1, 2, 3, 2, 1, 2, 3, 2, 1};
  std::vector<float> b = {2, -1, 0, -1, 2, -1, 0, -1, 2};
  Data<float2> a_s(a.data(), 9);
  Data<double2> a_d(a.data(), 9);
  Data<float2> b_s(b.data(), 9);
  Data<double2> b_d(b.data(), 9);
  Data<float> w_s(3);
  Data<double> w_d(3);

  cusolverDnHandle_t handle;
  cusolverDnCreate(&handle);

  a_s.H2D();
  a_d.H2D();
  b_s.H2D();
  b_d.H2D();

  int lwork_s;
  int lwork_d;
  cusolverDnChegvd_bufferSize(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_s.d_data, 3, b_s.d_data, 3, w_s.d_data, &lwork_s);
  cusolverDnZhegvd_bufferSize(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_d.d_data, 3, b_d.d_data, 3, w_d.d_data, &lwork_d);

  float2* work_s;
  double2* work_d;
  int *devInfo;
  cudaMalloc(&work_s, sizeof(float2) * lwork_s);
  cudaMalloc(&work_d, sizeof(double2) * lwork_d);
  cudaMalloc(&devInfo, sizeof(int));

  cusolverDnChegvd(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_s.d_data, 3, b_s.d_data, 3, w_s.d_data, work_s, lwork_s, devInfo);
  cusolverDnZhegvd(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_d.d_data, 3, b_d.d_data, 3, w_d.d_data, work_d, lwork_d, devInfo);

  a_s.D2H();
  a_d.D2H();
  b_s.D2H();
  b_d.D2H();
  w_s.D2H();
  w_d.D2H();

  cudaStreamSynchronize(0);

  cusolverDnDestroy(handle);
  cudaFree(work_s);
  cudaFree(work_d);
  cudaFree(devInfo);

  float expect_a[9] = {0.500000,-0.000000,-0.500000,0.194937,-0.484769,0.194937,0.679705,0.874642,0.679705};
  float expect_b[9] = {1.414214,-1.000000,0.000000,-0.707107,1.224745,-1.000000,0.000000,-0.816497,1.154701};
  float expect_w[3] = {-1.000000,-0.216991,9.216990};
  if (compare_result(expect_a, a_s.h_data, 9)
      && compare_result(expect_b, b_s.h_data, 9)
      && compare_result(expect_w, w_s.h_data, 3)
      && compare_result(expect_a, a_d.h_data, 9)
      && compare_result(expect_b, b_d.h_data, 9)
      && compare_result(expect_w, w_d.h_data, 3))
    printf("DnThegvd pass\n");
  else {
    printf("DnThegvd fail\n");
    test_passed = false;
  }
}

void test_helper() {
  cusolverDnHandle_t handle;
  cusolverDnCreate(&handle);
  cudaStream_t stream;
  cusolverDnGetStream(handle, &stream);
  cusolverDnSetStream(handle, stream);
}

int main() {
  test_helper();
  test_cusolverDnTsygvd();
  test_cusolverDnThegvd();

  if (test_passed)
    return 0;
  return -1;
}
