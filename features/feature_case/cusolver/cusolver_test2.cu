// ===------ cusolver_test2.cu ------------------------------*- CUDA -*-----===//
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

struct Ptr_Data {
  int group_num;
  void** h_data;
  void** d_data;
  Ptr_Data(int group_num) : group_num(group_num) {
    h_data = (void**)malloc(group_num * sizeof(void*));
    memset(h_data, 0, group_num * sizeof(void*));
    cudaMalloc(&d_data, group_num * sizeof(void*));
    cudaMemset(d_data, 0, group_num * sizeof(void*));
  }
  ~Ptr_Data() {
    free(h_data);
    cudaFree(d_data);
  }
  void H2D() {
    cudaMemcpy(d_data, h_data, group_num * sizeof(void*), cudaMemcpyHostToDevice);
  }
};

#ifndef DPCT_USM_LEVEL_NONE
void test_cusolverDnTpotrfBatched() {
  std::vector<float> a = {2, -1, 0, -1, 2, -1, 0, -1, 2,
                          2, -1, 0, -1, 2, -1, 0, -1, 2};
  Data<float> a_s(a.data(), 18);
  Data<double> a_d(a.data(), 18);
  Data<float2> a_c(a.data(), 18);
  Data<double2> a_z(a.data(), 18);

  Ptr_Data a_s_ptrs(2); a_s_ptrs.h_data[0] = a_s.d_data; a_s_ptrs.h_data[1] = a_s.d_data + 9;
  Ptr_Data a_d_ptrs(2); a_d_ptrs.h_data[0] = a_d.d_data; a_d_ptrs.h_data[1] = a_d.d_data + 9;
  Ptr_Data a_c_ptrs(2); a_c_ptrs.h_data[0] = a_c.d_data; a_c_ptrs.h_data[1] = a_c.d_data + 9;
  Ptr_Data a_z_ptrs(2); a_z_ptrs.h_data[0] = a_z.d_data; a_z_ptrs.h_data[1] = a_z.d_data + 9;

  cusolverDnHandle_t handle;
  cusolverDnCreate(&handle);

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();

  a_s_ptrs.H2D();
  a_d_ptrs.H2D();
  a_c_ptrs.H2D();
  a_z_ptrs.H2D();

  int *infoArray;
  cudaMalloc(&infoArray, 2 * sizeof(int));

  cusolverDnSpotrfBatched(handle, CUBLAS_FILL_MODE_UPPER, 3, (float **)a_s_ptrs.d_data, 3, infoArray, 2);
  cusolverDnDpotrfBatched(handle, CUBLAS_FILL_MODE_UPPER, 3, (double **)a_d_ptrs.d_data, 3, infoArray, 2);
  cusolverDnCpotrfBatched(handle, CUBLAS_FILL_MODE_UPPER, 3, (float2 **)a_c_ptrs.d_data, 3, infoArray, 2);
  cusolverDnZpotrfBatched(handle, CUBLAS_FILL_MODE_UPPER, 3, (double2 **)a_z_ptrs.d_data, 3, infoArray, 2);

  a_s.D2H();
  a_d.D2H();
  a_c.D2H();
  a_z.D2H();

  cudaStreamSynchronize(0);

  cusolverDnDestroy(handle);

  std::vector<int> indeces = {0, 3, 4, 6, 7, 8,
                              9,12,13,15,16,17 };
  float expect[18] = { 1.414214,-0.707107,0.000000,-0.707107,1.224745,-0.816497,0.000000,-0.816497,1.154701,
                       1.414214,-0.707107,0.000000,-0.707107,1.224745,-0.816497,0.000000,-0.816497,1.154701 };
  if (compare_result(expect, a_s.h_data, indeces) &&
      compare_result(expect, a_d.h_data, indeces) &&
      compare_result(expect, a_c.h_data, indeces) &&
      compare_result(expect, a_z.h_data, indeces))
    printf("DnTpotrfBatched pass\n");
  else {
    printf("DnTpotrfBatched fail\n");
    test_passed = false;
  }
}

void test_cusolverDnTpotrsBatched() {
  std::vector<float> a = {1.414214,-0.707107,0.000000,-0.707107,1.224745,-0.816497,0.000000,-0.816497,1.154701,
                          1.414214,-0.707107,0.000000,-0.707107,1.224745,-0.816497,0.000000,-0.816497,1.154701 };
  Data<float> a_s(a.data(), 18);
  Data<double> a_d(a.data(), 18);
  Data<float2> a_c(a.data(), 18);
  Data<double2> a_z(a.data(), 18);

  Ptr_Data a_s_ptrs(2); a_s_ptrs.h_data[0] = a_s.d_data; a_s_ptrs.h_data[1] = a_s.d_data + 9;
  Ptr_Data a_d_ptrs(2); a_d_ptrs.h_data[0] = a_d.d_data; a_d_ptrs.h_data[1] = a_d.d_data + 9;
  Ptr_Data a_c_ptrs(2); a_c_ptrs.h_data[0] = a_c.d_data; a_c_ptrs.h_data[1] = a_c.d_data + 9;
  Ptr_Data a_z_ptrs(2); a_z_ptrs.h_data[0] = a_z.d_data; a_z_ptrs.h_data[1] = a_z.d_data + 9;

  std::vector<float> b = {0, 0, 4,
                          0, 0, 4};
  Data<float> b_s(b.data(), 6);
  Data<double> b_d(b.data(), 6);
  Data<float2> b_c(b.data(), 6);
  Data<double2> b_z(b.data(), 6);

  Ptr_Data b_s_ptrs(2); b_s_ptrs.h_data[0] = b_s.d_data; b_s_ptrs.h_data[1] = b_s.d_data + 3;
  Ptr_Data b_d_ptrs(2); b_d_ptrs.h_data[0] = b_d.d_data; b_d_ptrs.h_data[1] = b_d.d_data + 3;
  Ptr_Data b_c_ptrs(2); b_c_ptrs.h_data[0] = b_c.d_data; b_c_ptrs.h_data[1] = b_c.d_data + 3;
  Ptr_Data b_z_ptrs(2); b_z_ptrs.h_data[0] = b_z.d_data; b_z_ptrs.h_data[1] = b_z.d_data + 3;

  cusolverDnHandle_t handle;
  cusolverDnCreate(&handle);

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();

  a_s_ptrs.H2D();
  a_d_ptrs.H2D();
  a_c_ptrs.H2D();
  a_z_ptrs.H2D();

  b_s.H2D();
  b_d.H2D();
  b_c.H2D();
  b_z.H2D();

  b_s_ptrs.H2D();
  b_d_ptrs.H2D();
  b_c_ptrs.H2D();
  b_z_ptrs.H2D();

  int *infoArray;
  cudaMalloc(&infoArray, 2 * sizeof(int));

  cusolverDnSpotrsBatched(handle, CUBLAS_FILL_MODE_UPPER, 3, 1, (float **)a_s_ptrs.d_data, 3, (float **)b_s_ptrs.d_data, 3, infoArray, 2);
  cusolverDnDpotrsBatched(handle, CUBLAS_FILL_MODE_UPPER, 3, 1, (double **)a_d_ptrs.d_data, 3, (double **)b_d_ptrs.d_data, 3, infoArray, 2);
  cusolverDnCpotrsBatched(handle, CUBLAS_FILL_MODE_UPPER, 3, 1, (float2 **)a_c_ptrs.d_data, 3, (float2 **)b_c_ptrs.d_data, 3, infoArray, 2);
  cusolverDnZpotrsBatched(handle, CUBLAS_FILL_MODE_UPPER, 3, 1, (double2 **)a_z_ptrs.d_data, 3, (double2 **)b_z_ptrs.d_data, 3, infoArray, 2);

  a_s.D2H();
  a_d.D2H();
  a_c.D2H();
  a_z.D2H();

  b_s.D2H();
  b_d.D2H();
  b_c.D2H();
  b_z.D2H();

  cudaStreamSynchronize(0);

  cusolverDnDestroy(handle);

  float expect[6] = { 1,2,3,
                      1,2,3 };
  if (compare_result(expect, b_s.h_data, 6) &&
      compare_result(expect, b_d.h_data, 6) &&
      compare_result(expect, b_c.h_data, 6) &&
      compare_result(expect, b_z.h_data, 6))
    printf("DnTpotrsBatched pass\n");
  else {
    printf("DnTpotrsBatched fail\n");
    test_passed = false;
  }
}
#endif

int main() {
#ifndef DPCT_USM_LEVEL_NONE
  test_cusolverDnTpotrfBatched();
  test_cusolverDnTpotrsBatched();
#endif

  if (test_passed)
    return 0;
  return -1;
}
