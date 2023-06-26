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

int main() {
  test_cusolverDnTsyevdx_cusolverDnTheevdx();
  test_cusolverDnTsygvdx_cusolverDnThegvdx();
  test_cusolverDnTsygvj_cusolverDnThegvj();

  if (test_passed)
    return 0;
  return -1;
}
