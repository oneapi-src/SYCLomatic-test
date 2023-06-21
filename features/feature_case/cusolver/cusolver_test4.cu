// ===------ cusolver_test4.cu ------------------------------*- CUDA -*-----===//
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

void test_cusolverDnTsyevdx_cusolverDnTheevdx() {
  std::vector<float> a = {1, 2, 2, 4};
  Data<float> a_s(a.data(), 4);
  Data<double> a_d(a.data(), 4);
  Data<float2> a_c(a.data(), 4);
  Data<double2> a_z(a.data(), 4);
  Data<float> w_s(2);
  Data<double> w_d(2);
  Data<float> w_c(2);
  Data<double> w_z(2);

  cusolverDnHandle_t handle;
  cusolverDnCreate(&handle);

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();

  int lwork_s;
  int lwork_d;
  int lwork_c;
  int lwork_z;

  int h_meig_s;
  int h_meig_d;
  int h_meig_c;
  int h_meig_z;

  cusolverDnSsyevdx_bufferSize(handle, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, a_s.d_data, 2, 0, 0, 0, 0, &h_meig_s, w_s.d_data, &lwork_s);
  cusolverDnDsyevdx_bufferSize(handle, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, a_d.d_data, 2, 0, 0, 0, 0, &h_meig_d, w_d.d_data, &lwork_d);
  cusolverDnCheevdx_bufferSize(handle, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, a_c.d_data, 2, 0, 0, 0, 0, &h_meig_c, w_c.d_data, &lwork_c);
  cusolverDnZheevdx_bufferSize(handle, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, a_z.d_data, 2, 0, 0, 0, 0, &h_meig_z, w_z.d_data, &lwork_z);

  float* device_ws_s;
  double* device_ws_d;
  float2* device_ws_c;
  double2* device_ws_z;
  cudaMalloc(&device_ws_s, lwork_s * sizeof(float));
  cudaMalloc(&device_ws_d, lwork_d * sizeof(double));
  cudaMalloc(&device_ws_c, lwork_c * sizeof(float2));
  cudaMalloc(&device_ws_z, lwork_z * sizeof(double2));

  int *info;
  cudaMalloc(&info, sizeof(int));

  cusolverDnSsyevdx(handle, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, a_s.d_data, 2, 0, 0, 0, 0, &h_meig_s, w_s.d_data, device_ws_s, lwork_s, info);
  cusolverDnDsyevdx(handle, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, a_d.d_data, 2, 0, 0, 0, 0, &h_meig_d, w_d.d_data, device_ws_d, lwork_d, info);
  cusolverDnCheevdx(handle, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, a_c.d_data, 2, 0, 0, 0, 0, &h_meig_c, w_c.d_data, device_ws_c, lwork_c, info);
  cusolverDnZheevdx(handle, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, a_z.d_data, 2, 0, 0, 0, 0, &h_meig_z, w_z.d_data, device_ws_z, lwork_z, info);

  a_s.D2H();
  a_d.D2H();
  a_c.D2H();
  a_z.D2H();
  w_s.D2H();
  w_d.D2H();
  w_c.D2H();
  w_z.D2H();

  cudaStreamSynchronize(0);

  cusolverDnDestroy(handle);
  cudaFree(device_ws_s);
  cudaFree(device_ws_d);
  cudaFree(device_ws_c);
  cudaFree(device_ws_z);
  cudaFree(info);

  float expect_a[4] = {0.894427,-0.447214,0.447214,0.894427};
  int expect_h_meig = 2;
  float expect_w[2] = {0.000000,5.000000};
  if (compare_result(expect_a, a_s.h_data, 4) &&
      compare_result(expect_a, a_d.h_data, 4) &&
      compare_result(expect_a, a_c.h_data, 4) &&
      compare_result(expect_a, a_z.h_data, 4) &&
      compare_result(&expect_h_meig, &h_meig_s, 1) &&
      compare_result(&expect_h_meig, &h_meig_d, 1) &&
      compare_result(&expect_h_meig, &h_meig_c, 1) &&
      compare_result(&expect_h_meig, &h_meig_z, 1) &&
      compare_result(expect_w, w_s.h_data, 2) &&
      compare_result(expect_w, w_d.h_data, 2) &&
      compare_result(expect_w, w_c.h_data, 2) &&
      compare_result(expect_w, w_z.h_data, 2))
    printf("DnTsyevdx/DnCheevdx pass\n");
  else {
    printf("DnTsyevdx/DnCheevdx fail\n");
    test_passed = false;
  }
}

void test_cusolverDnTsygvdx_cusolverDnThegvdx() {
  std::vector<float> a = {1, 2, 2, 4};
  Data<float> a_s(a.data(), 4);
  Data<double> a_d(a.data(), 4);
  Data<float2> a_c(a.data(), 4);
  Data<double2> a_z(a.data(), 4);
  std::vector<float> b = {1, 0, 0, 1};
  Data<float> b_s(b.data(), 4);
  Data<double> b_d(b.data(), 4);
  Data<float2> b_c(b.data(), 4);
  Data<double2> b_z(b.data(), 4);
  Data<float> w_s(2);
  Data<double> w_d(2);
  Data<float> w_c(2);
  Data<double> w_z(2);

  cusolverDnHandle_t handle;
  cusolverDnCreate(&handle);

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();
  b_s.H2D();
  b_d.H2D();
  b_c.H2D();
  b_z.H2D();

  int lwork_s;
  int lwork_d;
  int lwork_c;
  int lwork_z;

  int h_meig_s;
  int h_meig_d;
  int h_meig_c;
  int h_meig_z;

  cusolverDnSsygvdx_bufferSize(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, a_s.d_data, 2, b_s.d_data, 2, 0, 0, 0, 0, &h_meig_s, w_s.d_data, &lwork_s);
  cusolverDnDsygvdx_bufferSize(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, a_d.d_data, 2, b_d.d_data, 2, 0, 0, 0, 0, &h_meig_d, w_d.d_data, &lwork_d);
  cusolverDnChegvdx_bufferSize(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, a_c.d_data, 2, b_c.d_data, 2, 0, 0, 0, 0, &h_meig_c, w_c.d_data, &lwork_c);
  cusolverDnZhegvdx_bufferSize(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, a_z.d_data, 2, b_z.d_data, 2, 0, 0, 0, 0, &h_meig_z, w_z.d_data, &lwork_z);

  float* device_ws_s;
  double* device_ws_d;
  float2* device_ws_c;
  double2* device_ws_z;
  cudaMalloc(&device_ws_s, lwork_s * sizeof(float));
  cudaMalloc(&device_ws_d, lwork_d * sizeof(double));
  cudaMalloc(&device_ws_c, lwork_c * sizeof(float2));
  cudaMalloc(&device_ws_z, lwork_z * sizeof(double2));

  int *info;
  cudaMalloc(&info, sizeof(int));

  cusolverDnSsygvdx(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, a_s.d_data, 2, b_s.d_data, 2, 0, 0, 0, 0, &h_meig_s, w_s.d_data, device_ws_s, lwork_s, info);
  cusolverDnDsygvdx(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, a_d.d_data, 2, b_d.d_data, 2, 0, 0, 0, 0, &h_meig_d, w_d.d_data, device_ws_d, lwork_d, info);
  cusolverDnChegvdx(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, a_c.d_data, 2, b_c.d_data, 2, 0, 0, 0, 0, &h_meig_c, w_c.d_data, device_ws_c, lwork_c, info);
  cusolverDnZhegvdx(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, a_z.d_data, 2, b_z.d_data, 2, 0, 0, 0, 0, &h_meig_z, w_z.d_data, device_ws_z, lwork_z, info);

  a_s.D2H();
  a_d.D2H();
  a_c.D2H();
  a_z.D2H();
  b_s.D2H();
  b_d.D2H();
  b_c.D2H();
  b_z.D2H();
  w_s.D2H();
  w_d.D2H();
  w_c.D2H();
  w_z.D2H();

  cudaStreamSynchronize(0);

  cusolverDnDestroy(handle);
  cudaFree(device_ws_s);
  cudaFree(device_ws_d);
  cudaFree(device_ws_c);
  cudaFree(device_ws_z);
  cudaFree(info);

  float expect_a[4] = {0.894427,-0.447214,0.447214,0.894427};
  float expect_b[4] = {1,0,0,1};
  int expect_h_meig = 2;
  float expect_w[2] = {0.000000,5.000000};
  if (compare_result(expect_a, a_s.h_data, 4) &&
      compare_result(expect_a, a_d.h_data, 4) &&
      compare_result(expect_a, a_c.h_data, 4) &&
      compare_result(expect_a, a_z.h_data, 4) &&
      compare_result(expect_b, b_s.h_data, 4) &&
      compare_result(expect_b, b_d.h_data, 4) &&
      compare_result(expect_b, b_c.h_data, 4) &&
      compare_result(expect_b, b_z.h_data, 4) &&
      compare_result(&expect_h_meig, &h_meig_s, 1) &&
      compare_result(&expect_h_meig, &h_meig_d, 1) &&
      compare_result(&expect_h_meig, &h_meig_c, 1) &&
      compare_result(&expect_h_meig, &h_meig_z, 1) &&
      compare_result(expect_w, w_s.h_data, 2) &&
      compare_result(expect_w, w_d.h_data, 2) &&
      compare_result(expect_w, w_c.h_data, 2) &&
      compare_result(expect_w, w_z.h_data, 2))
    printf("DnTsygvdx/DnChegvdx pass\n");
  else {
    printf("DnTsygvdx/DnChegvdx fail\n");
    test_passed = false;
  }
}

void test_cusolverDnTsygvj_cusolverDnThegvj() {
  std::vector<float> a = {1, 2, 2, 4};
  Data<float> a_s(a.data(), 4);
  Data<double> a_d(a.data(), 4);
  Data<float2> a_c(a.data(), 4);
  Data<double2> a_z(a.data(), 4);
  std::vector<float> b = {1, 0, 0, 1};
  Data<float> b_s(b.data(), 4);
  Data<double> b_d(b.data(), 4);
  Data<float2> b_c(b.data(), 4);
  Data<double2> b_z(b.data(), 4);
  Data<float> w_s(2);
  Data<double> w_d(2);
  Data<float> w_c(2);
  Data<double> w_z(2);

  cusolverDnHandle_t handle;
  cusolverDnCreate(&handle);

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();
  b_s.H2D();
  b_d.H2D();
  b_c.H2D();
  b_z.H2D();

  syevjInfo_t params;
  cusolverDnCreateSyevjInfo(&params);

  int lwork_s;
  int lwork_d;
  int lwork_c;
  int lwork_z;

  cusolverDnSsygvj_bufferSize(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, a_s.d_data, 2, b_s.d_data, 2, w_s.d_data, &lwork_s, params);
  cusolverDnDsygvj_bufferSize(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, a_d.d_data, 2, b_d.d_data, 2, w_d.d_data, &lwork_d, params);
  cusolverDnChegvj_bufferSize(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, a_c.d_data, 2, b_c.d_data, 2, w_c.d_data, &lwork_c, params);
  cusolverDnZhegvj_bufferSize(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, a_z.d_data, 2, b_z.d_data, 2, w_z.d_data, &lwork_z, params);

  float* device_ws_s;
  double* device_ws_d;
  float2* device_ws_c;
  double2* device_ws_z;
  cudaMalloc(&device_ws_s, lwork_s * sizeof(float));
  cudaMalloc(&device_ws_d, lwork_d * sizeof(double));
  cudaMalloc(&device_ws_c, lwork_c * sizeof(float2));
  cudaMalloc(&device_ws_z, lwork_z * sizeof(double2));

  int *info;
  cudaMalloc(&info, sizeof(int));

  cusolverDnSsygvj(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, a_s.d_data, 2, b_s.d_data, 2, w_s.d_data, device_ws_s, lwork_s, info, params);
  cusolverDnDsygvj(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, a_d.d_data, 2, b_d.d_data, 2, w_d.d_data, device_ws_d, lwork_d, info, params);
  cusolverDnChegvj(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, a_c.d_data, 2, b_c.d_data, 2, w_c.d_data, device_ws_c, lwork_c, info, params);
  cusolverDnZhegvj(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, a_z.d_data, 2, b_z.d_data, 2, w_z.d_data, device_ws_z, lwork_z, info, params);

  a_s.D2H();
  a_d.D2H();
  a_c.D2H();
  a_z.D2H();
  b_s.D2H();
  b_d.D2H();
  b_c.D2H();
  b_z.D2H();
  w_s.D2H();
  w_d.D2H();
  w_c.D2H();
  w_z.D2H();

  cudaStreamSynchronize(0);

  cusolverDnDestroySyevjInfo(params);
  cusolverDnDestroy(handle);
  cudaFree(device_ws_s);
  cudaFree(device_ws_d);
  cudaFree(device_ws_c);
  cudaFree(device_ws_z);
  cudaFree(info);

  float expect_a[4] = {-0.894427,0.447214,0.447214,0.894427};
  float expect_b[4] = {1,0,0,1};
  float expect_w[2] = {0.000000,5.000000};
  if (compare_result(expect_a, a_s.h_data, 4) &&
      compare_result(expect_a, a_d.h_data, 4) &&
      compare_result(expect_a, a_c.h_data, 4) &&
      compare_result(expect_a, a_z.h_data, 4) &&
      compare_result(expect_b, b_s.h_data, 4) &&
      compare_result(expect_b, b_d.h_data, 4) &&
      compare_result(expect_b, b_c.h_data, 4) &&
      compare_result(expect_b, b_z.h_data, 4) &&
      compare_result(expect_w, w_s.h_data, 2) &&
      compare_result(expect_w, w_d.h_data, 2) &&
      compare_result(expect_w, w_c.h_data, 2) &&
      compare_result(expect_w, w_z.h_data, 2))
    printf("DnTsygvj/DnChegvj pass\n");
  else {
    printf("DnTsygvj/DnChegvj fail\n");
    test_passed = false;
  }
}

void test_cusolverDnTsyevj_cusolverDnTheevj() {
  std::vector<float> a = {1, 2, 2, 4};
  Data<float> a_s(a.data(), 4);
  Data<double> a_d(a.data(), 4);
  Data<float2> a_c(a.data(), 4);
  Data<double2> a_z(a.data(), 4);
  Data<float> w_s(2);
  Data<double> w_d(2);
  Data<float> w_c(2);
  Data<double> w_z(2);

  cusolverDnHandle_t handle;
  cusolverDnCreate(&handle);

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();

  syevjInfo_t params;
  cusolverDnCreateSyevjInfo(&params);

  int lwork_s;
  int lwork_d;
  int lwork_c;
  int lwork_z;

  cusolverDnSsyevj_bufferSize(handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, a_s.d_data, 2, w_s.d_data, &lwork_s, params);
  cusolverDnDsyevj_bufferSize(handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, a_d.d_data, 2, w_d.d_data, &lwork_d, params);
  cusolverDnCheevj_bufferSize(handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, a_c.d_data, 2, w_c.d_data, &lwork_c, params);
  cusolverDnZheevj_bufferSize(handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, a_z.d_data, 2, w_z.d_data, &lwork_z, params);

  float* device_ws_s;
  double* device_ws_d;
  float2* device_ws_c;
  double2* device_ws_z;
  cudaMalloc(&device_ws_s, lwork_s * sizeof(float));
  cudaMalloc(&device_ws_d, lwork_d * sizeof(double));
  cudaMalloc(&device_ws_c, lwork_c * sizeof(float2));
  cudaMalloc(&device_ws_z, lwork_z * sizeof(double2));

  int *info;
  cudaMalloc(&info, sizeof(int));

  cusolverDnSsyevj(handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, a_s.d_data, 2, w_s.d_data, device_ws_s, lwork_s, info, params);
  cusolverDnDsyevj(handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, a_d.d_data, 2, w_d.d_data, device_ws_d, lwork_d, info, params);
  cusolverDnCheevj(handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, a_c.d_data, 2, w_c.d_data, device_ws_c, lwork_c, info, params);
  cusolverDnZheevj(handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, a_z.d_data, 2, w_z.d_data, device_ws_z, lwork_z, info, params);

  a_s.D2H();
  a_d.D2H();
  a_c.D2H();
  a_z.D2H();
  w_s.D2H();
  w_d.D2H();
  w_c.D2H();
  w_z.D2H();

  cudaStreamSynchronize(0);

  cusolverDnDestroySyevjInfo(params);
  cusolverDnDestroy(handle);
  cudaFree(device_ws_s);
  cudaFree(device_ws_d);
  cudaFree(device_ws_c);
  cudaFree(device_ws_z);
  cudaFree(info);

  float expect_a[4] = {-0.894427,0.447214,0.447214,0.894427};
  float expect_w[2] = {0.000000,5.000000};
  if (compare_result(expect_a, a_s.h_data, 4) &&
      compare_result(expect_a, a_d.h_data, 4) &&
      compare_result(expect_a, a_c.h_data, 4) &&
      compare_result(expect_a, a_z.h_data, 4) &&
      compare_result(expect_w, w_s.h_data, 2) &&
      compare_result(expect_w, w_d.h_data, 2) &&
      compare_result(expect_w, w_c.h_data, 2) &&
      compare_result(expect_w, w_z.h_data, 2))
    printf("DnTsyevj/DnCheevj pass\n");
  else {
    printf("DnTsyevj/DnCheevj fail\n");
    test_passed = false;
  }
}

void test_cusolverDnXsyevd() {
  std::vector<float> a = {1, 2, 2, 4};
  Data<float> a_s(a.data(), 4);
  Data<double> a_d(a.data(), 4);
  Data<float2> a_c(a.data(), 4);
  Data<double2> a_z(a.data(), 4);
  Data<float> w_s(2);
  Data<double> w_d(2);
  Data<float> w_c(2);
  Data<double> w_z(2);

  cusolverDnHandle_t handle;
  cusolverDnCreate(&handle);

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();

  cusolverDnParams_t params;
  cusolverDnCreateParams(&params);

  size_t lwork_s;
  size_t lwork_d;
  size_t lwork_c;
  size_t lwork_z;
  size_t lwork_host_s;
  size_t lwork_host_d;
  size_t lwork_host_c;
  size_t lwork_host_z;

  cusolverDnXsyevd_bufferSize(handle, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, CUDA_R_32F, a_s.d_data, 2, CUDA_R_32F, w_s.d_data, CUDA_R_32F, &lwork_s, &lwork_host_s);
  cusolverDnXsyevd_bufferSize(handle, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, CUDA_R_64F, a_d.d_data, 2, CUDA_R_64F, w_d.d_data, CUDA_R_64F, &lwork_d, &lwork_host_d);
  cusolverDnXsyevd_bufferSize(handle, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, CUDA_C_32F, a_c.d_data, 2, CUDA_R_32F, w_c.d_data, CUDA_C_32F, &lwork_c, &lwork_host_c);
  cusolverDnXsyevd_bufferSize(handle, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, CUDA_C_64F, a_z.d_data, 2, CUDA_R_64F, w_z.d_data, CUDA_C_64F, &lwork_z, &lwork_host_z);

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

  cusolverDnXsyevd(handle, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, CUDA_R_32F, a_s.d_data, 2, CUDA_R_32F, w_s.d_data, CUDA_R_32F, device_ws_s, lwork_s, host_ws_s, lwork_host_s, info);
  cusolverDnXsyevd(handle, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, CUDA_R_64F, a_d.d_data, 2, CUDA_R_64F, w_d.d_data, CUDA_R_64F, device_ws_d, lwork_d, host_ws_d, lwork_host_d, info);
  cusolverDnXsyevd(handle, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, CUDA_C_32F, a_c.d_data, 2, CUDA_R_32F, w_c.d_data, CUDA_C_32F, device_ws_c, lwork_c, host_ws_c, lwork_host_c, info);
  cusolverDnXsyevd(handle, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, CUDA_C_64F, a_z.d_data, 2, CUDA_R_64F, w_z.d_data, CUDA_C_64F, device_ws_z, lwork_z, host_ws_z, lwork_host_z, info);

  a_s.D2H();
  a_d.D2H();
  a_c.D2H();
  a_z.D2H();
  w_s.D2H();
  w_d.D2H();
  w_c.D2H();
  w_z.D2H();

  cudaStreamSynchronize(0);

  cusolverDnDestroyParams(params);
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

  float expect_a[4] = {-0.894427,0.447214,0.447214,0.894427};
  float expect_w[2] = {0.000000,5.000000};
  if (compare_result(expect_a, a_s.h_data, 4) &&
      compare_result(expect_a, a_d.h_data, 4) &&
      compare_result(expect_a, a_c.h_data, 4) &&
      compare_result(expect_a, a_z.h_data, 4) &&
      compare_result(expect_w, w_s.h_data, 2) &&
      compare_result(expect_w, w_d.h_data, 2) &&
      compare_result(expect_w, w_c.h_data, 2) &&
      compare_result(expect_w, w_z.h_data, 2))
    printf("DnXsyevd pass\n");
  else {
    printf("DnXsyevd fail\n");
    test_passed = false;
  }
}

void test_cusolverDnSyevd() {
  std::vector<float> a = {1, 2, 2, 4};
  Data<float> a_s(a.data(), 4);
  Data<double> a_d(a.data(), 4);
  Data<float2> a_c(a.data(), 4);
  Data<double2> a_z(a.data(), 4);
  Data<float> w_s(2);
  Data<double> w_d(2);
  Data<float> w_c(2);
  Data<double> w_z(2);

  cusolverDnHandle_t handle;
  cusolverDnCreate(&handle);

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();

  cusolverDnParams_t params;
  cusolverDnCreateParams(&params);

  size_t lwork_s;
  size_t lwork_d;
  size_t lwork_c;
  size_t lwork_z;

  cusolverDnSyevd_bufferSize(handle, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, CUDA_R_32F, a_s.d_data, 2, CUDA_R_32F, w_s.d_data, CUDA_R_32F, &lwork_s);
  cusolverDnSyevd_bufferSize(handle, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, CUDA_R_64F, a_d.d_data, 2, CUDA_R_64F, w_d.d_data, CUDA_R_64F, &lwork_d);
  cusolverDnSyevd_bufferSize(handle, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, CUDA_C_32F, a_c.d_data, 2, CUDA_R_32F, w_c.d_data, CUDA_C_32F, &lwork_c);
  cusolverDnSyevd_bufferSize(handle, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, CUDA_C_64F, a_z.d_data, 2, CUDA_R_64F, w_z.d_data, CUDA_C_64F, &lwork_z);

  void* device_ws_s;
  void* device_ws_d;
  void* device_ws_c;
  void* device_ws_z;
  cudaMalloc(&device_ws_s, lwork_s);
  cudaMalloc(&device_ws_d, lwork_d);
  cudaMalloc(&device_ws_c, lwork_c);
  cudaMalloc(&device_ws_z, lwork_z);

  int *info;
  cudaMalloc(&info, sizeof(int));

  cusolverDnSyevd(handle, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, CUDA_R_32F, a_s.d_data, 2, CUDA_R_32F, w_s.d_data, CUDA_R_32F, device_ws_s, lwork_s, info);
  cusolverDnSyevd(handle, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, CUDA_R_64F, a_d.d_data, 2, CUDA_R_64F, w_d.d_data, CUDA_R_64F, device_ws_d, lwork_d, info);
  cusolverDnSyevd(handle, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, CUDA_C_32F, a_c.d_data, 2, CUDA_R_32F, w_c.d_data, CUDA_C_32F, device_ws_c, lwork_c, info);
  cusolverDnSyevd(handle, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, CUDA_C_64F, a_z.d_data, 2, CUDA_R_64F, w_z.d_data, CUDA_C_64F, device_ws_z, lwork_z, info);

  a_s.D2H();
  a_d.D2H();
  a_c.D2H();
  a_z.D2H();
  w_s.D2H();
  w_d.D2H();
  w_c.D2H();
  w_z.D2H();

  cudaStreamSynchronize(0);

  cusolverDnDestroyParams(params);
  cusolverDnDestroy(handle);
  cudaFree(device_ws_s);
  cudaFree(device_ws_d);
  cudaFree(device_ws_c);
  cudaFree(device_ws_z);
  cudaFree(info);

  float expect_a[4] = {-0.894427,0.447214,0.447214,0.894427};
  float expect_w[2] = {0.000000,5.000000};
  if (compare_result(expect_a, a_s.h_data, 4) &&
      compare_result(expect_a, a_d.h_data, 4) &&
      compare_result(expect_a, a_c.h_data, 4) &&
      compare_result(expect_a, a_z.h_data, 4) &&
      compare_result(expect_w, w_s.h_data, 2) &&
      compare_result(expect_w, w_d.h_data, 2) &&
      compare_result(expect_w, w_c.h_data, 2) &&
      compare_result(expect_w, w_z.h_data, 2))
    printf("DnSyevd pass\n");
  else {
    printf("DnSyevd fail\n");
    test_passed = false;
  }
}

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
  test_cusolverDnTsyevdx_cusolverDnTheevdx();
  test_cusolverDnTsygvdx_cusolverDnThegvdx();
  test_cusolverDnTsygvj_cusolverDnThegvj();
  test_cusolverDnTsyevj_cusolverDnTheevj();
  test_cusolverDnXsyevd();
  test_cusolverDnSyevd();
  test_cusolverDnXtrtri();

  if (test_passed)
    return 0;
  return -1;
}
