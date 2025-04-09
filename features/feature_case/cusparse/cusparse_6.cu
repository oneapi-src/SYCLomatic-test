// ===------- cusparse_6.cu -------------------------------- *- CUDA -* ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include "cusparse.h"

#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

template <class d_data_t>
struct Data {
  float *h_data;
  d_data_t *d_data;
  int element_num;
  Data(int element_num) : element_num(element_num) {
    h_data = (float *)malloc(sizeof(float) * element_num);
    memset(h_data, 0, sizeof(float) * element_num);
    cudaMalloc(&d_data, sizeof(d_data_t) * element_num);
    cudaMemset(d_data, 0, sizeof(d_data_t) * element_num);
  }
  Data(float *input_data, int element_num) : element_num(element_num) {
    h_data = (float *)malloc(sizeof(float) * element_num);
    cudaMalloc(&d_data, sizeof(d_data_t) * element_num);
    cudaMemset(d_data, 0, sizeof(d_data_t) * element_num);
    memcpy(h_data, input_data, sizeof(float) * element_num);
  }
  ~Data() {
    free(h_data);
    cudaFree(d_data);
  }
  void H2D() {
    d_data_t *h_temp = (d_data_t *)malloc(sizeof(d_data_t) * element_num);
    memset(h_temp, 0, sizeof(d_data_t) * element_num);
    from_float_convert(h_data, h_temp);
    cudaMemcpy(d_data, h_temp, sizeof(d_data_t) * element_num,
               cudaMemcpyHostToDevice);
    free(h_temp);
  }
  void D2H() {
    d_data_t *h_temp = (d_data_t *)malloc(sizeof(d_data_t) * element_num);
    memset(h_temp, 0, sizeof(d_data_t) * element_num);
    cudaMemcpy(h_temp, d_data, sizeof(d_data_t) * element_num,
               cudaMemcpyDeviceToHost);
    to_float_convert(h_temp, h_data);
    free(h_temp);
  }

private:
  inline void from_float_convert(float *in, d_data_t *out) {
    for (int i = 0; i < element_num; i++)
      out[i] = in[i];
  }
  inline void to_float_convert(d_data_t *in, float *out) {
    for (int i = 0; i < element_num; i++)
      out[i] = in[i];
  }
};
template <>
inline void Data<float2>::from_float_convert(float *in, float2 *out) {
  for (int i = 0; i < element_num; i++)
    out[i].x = in[i];
}
template <>
inline void Data<double2>::from_float_convert(float *in, double2 *out) {
  for (int i = 0; i < element_num; i++)
    out[i].x = in[i];
}

template <>
inline void Data<float2>::to_float_convert(float2 *in, float *out) {
  for (int i = 0; i < element_num; i++)
    out[i] = in[i].x;
}
template <>
inline void Data<double2>::to_float_convert(double2 *in, float *out) {
  for (int i = 0; i < element_num; i++)
    out[i] = in[i].x;
}

bool compare_result(float *expect, float *result, int element_num) {
  for (int i = 0; i < element_num; i++) {
    if (std::abs(result[i] - expect[i]) >= 0.05) {
      return false;
    }
  }
  return true;
}

bool compare_result(float *expect, float *result, std::vector<int> indices) {
  for (int i = 0; i < indices.size(); i++) {
    if (std::abs(result[indices[i]] - expect[indices[i]]) >= 0.05) {
      return false;
    }
  }
  return true;
}

bool test_passed = true;

// A * x = f
//
// | 1 1 2 |   | 1 |   | 9  |  
// | 0 1 3 | * | 2 | = | 11 |
// | 0 0 1 |   | 3 |   | 3  |
void test_cusparseTcsrsv2() {
  std::vector<float> a_val_vec = {1, 1, 2, 1, 3, 1};
  Data<float> a_s_val(a_val_vec.data(), 6);
  Data<double> a_d_val(a_val_vec.data(), 6);
  Data<float2> a_c_val(a_val_vec.data(), 6);
  Data<double2> a_z_val(a_val_vec.data(), 6);
  std::vector<float> a_row_ptr_vec = {0, 3, 5, 6};
  Data<int> a_row_ptr_s(a_row_ptr_vec.data(), 4);
  Data<int> a_row_ptr_d(a_row_ptr_vec.data(), 4);
  Data<int> a_row_ptr_c(a_row_ptr_vec.data(), 4);
  Data<int> a_row_ptr_z(a_row_ptr_vec.data(), 4);
  std::vector<float> a_col_ind_vec = {0, 1, 2, 1, 2, 2};
  Data<int> a_col_ind_s(a_col_ind_vec.data(), 6);
  Data<int> a_col_ind_d(a_col_ind_vec.data(), 6);
  Data<int> a_col_ind_c(a_col_ind_vec.data(), 6);
  Data<int> a_col_ind_z(a_col_ind_vec.data(), 6);

  std::vector<float> f_vec = {9, 11, 3};
  Data<float> f_s(f_vec.data(), 3);
  Data<double> f_d(f_vec.data(), 3);
  Data<float2> f_c(f_vec.data(), 3);
  Data<double2> f_z(f_vec.data(), 3);

  Data<float> x_s(3);
  Data<double> x_d(3);
  Data<float2> x_c(3);
  Data<double2> x_z(3);

  cusparseHandle_t handle;
  cusparseCreate(&handle);
  csrsv2Info_t info_s;
  csrsv2Info_t info_d;
  csrsv2Info_t info_c;
  csrsv2Info_t info_z;
  cusparseCreateCsrsv2Info(&info_s);
  cusparseCreateCsrsv2Info(&info_d);
  cusparseCreateCsrsv2Info(&info_c);
  cusparseCreateCsrsv2Info(&info_z);
  cusparseSolvePolicy_t policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
  policy = CUSPARSE_SOLVE_POLICY_NO_LEVEL;


  cusparseMatDescr_t descrA;
  cusparseCreateMatDescr(&descrA);
  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_TRIANGULAR);
  cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_UNIT);

  a_s_val.H2D();
  a_d_val.H2D();
  a_c_val.H2D();
  a_z_val.H2D();
  a_row_ptr_s.H2D();
  a_row_ptr_d.H2D();
  a_row_ptr_c.H2D();
  a_row_ptr_z.H2D();
  a_col_ind_s.H2D();
  a_col_ind_d.H2D();
  a_col_ind_c.H2D();
  a_col_ind_z.H2D();
  f_s.H2D();
  f_d.H2D();
  f_c.H2D();
  f_z.H2D();

  int buffer_size_s0;
  int buffer_size_d0;
  int buffer_size_c0;
  int buffer_size_z0;
  cusparseScsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA,   (float *)a_s_val.d_data, (int *)a_row_ptr_s.d_data, (int *)a_col_ind_s.d_data, info_s, &buffer_size_s0);
  cusparseDcsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA,  (double *)a_d_val.d_data, (int *)a_row_ptr_d.d_data, (int *)a_col_ind_d.d_data, info_d, &buffer_size_d0);
  cusparseCcsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA,  (float2 *)a_c_val.d_data, (int *)a_row_ptr_c.d_data, (int *)a_col_ind_c.d_data, info_c, &buffer_size_c0);
  cusparseZcsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA, (double2 *)a_z_val.d_data, (int *)a_row_ptr_z.d_data, (int *)a_col_ind_z.d_data, info_z, &buffer_size_z0);

  size_t buffer_size_s;
  size_t buffer_size_d;
  size_t buffer_size_c;
  size_t buffer_size_z;
  cusparseScsrsv2_bufferSizeExt(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA,   (float *)a_s_val.d_data, (int *)a_row_ptr_s.d_data, (int *)a_col_ind_s.d_data, info_s, &buffer_size_s);
  cusparseDcsrsv2_bufferSizeExt(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA,  (double *)a_d_val.d_data, (int *)a_row_ptr_d.d_data, (int *)a_col_ind_d.d_data, info_d, &buffer_size_d);
  cusparseCcsrsv2_bufferSizeExt(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA,  (float2 *)a_c_val.d_data, (int *)a_row_ptr_c.d_data, (int *)a_col_ind_c.d_data, info_c, &buffer_size_c);
  cusparseZcsrsv2_bufferSizeExt(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA, (double2 *)a_z_val.d_data, (int *)a_row_ptr_z.d_data, (int *)a_col_ind_z.d_data, info_z, &buffer_size_z);

  void* buffer_s;
  void* buffer_d;
  void* buffer_c;
  void* buffer_z;
  cudaMalloc(&buffer_s, buffer_size_s);
  cudaMalloc(&buffer_d, buffer_size_d);
  cudaMalloc(&buffer_c, buffer_size_c);
  cudaMalloc(&buffer_z, buffer_size_z);

  cusparseScsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA,   (float *)a_s_val.d_data, (int *)a_row_ptr_s.d_data, (int *)a_col_ind_s.d_data, info_s, policy, buffer_s);
  cusparseDcsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA,  (double *)a_d_val.d_data, (int *)a_row_ptr_d.d_data, (int *)a_col_ind_d.d_data, info_d, policy, buffer_d);
  cusparseCcsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA,  (float2 *)a_c_val.d_data, (int *)a_row_ptr_c.d_data, (int *)a_col_ind_c.d_data, info_c, policy, buffer_c);
  cusparseZcsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA, (double2 *)a_z_val.d_data, (int *)a_row_ptr_z.d_data, (int *)a_col_ind_z.d_data, info_z, policy, buffer_z);

  float alpha_s = 1;
  double alpha_d = 1;
  float2 alpha_c = float2{1, 0};
  double2 alpha_z = double2{1, 0};

  cusparseScsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, &alpha_s, descrA,   (float *)a_s_val.d_data, (int *)a_row_ptr_s.d_data, (int *)a_col_ind_s.d_data, info_s, f_s.d_data, x_s.d_data, policy, buffer_s);
  cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, &alpha_d, descrA,  (double *)a_d_val.d_data, (int *)a_row_ptr_d.d_data, (int *)a_col_ind_d.d_data, info_d, f_d.d_data, x_d.d_data, policy, buffer_d);
  cusparseCcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, &alpha_c, descrA,  (float2 *)a_c_val.d_data, (int *)a_row_ptr_c.d_data, (int *)a_col_ind_c.d_data, info_c, f_c.d_data, x_c.d_data, policy, buffer_c);
  cusparseZcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, &alpha_z, descrA, (double2 *)a_z_val.d_data, (int *)a_row_ptr_z.d_data, (int *)a_col_ind_z.d_data, info_z, f_z.d_data, x_z.d_data, policy, buffer_z);

  x_s.D2H();
  x_d.D2H();
  x_c.D2H();
  x_z.D2H();

  cudaStreamSynchronize(0);
  cusparseDestroyCsrsv2Info(info_s);
  cusparseDestroyCsrsv2Info(info_d);
  cusparseDestroyCsrsv2Info(info_c);
  cusparseDestroyCsrsv2Info(info_z);
  cusparseDestroyMatDescr(descrA);
  cusparseDestroy(handle);
  cudaFree(buffer_s);
  cudaFree(buffer_d);
  cudaFree(buffer_c);
  cudaFree(buffer_z);

  float expect_x[4] = {1, 2, 3};
  if (compare_result(expect_x, x_s.h_data, 3) &&
      compare_result(expect_x, x_d.h_data, 3) &&
      compare_result(expect_x, x_c.h_data, 3) &&
      compare_result(expect_x, x_z.h_data, 3))
    printf("Tcsrsv2 pass\n");
  else {
    printf("Tcsrsv2 fail\n");
    test_passed = false;
  }
}

// 2*A*B + 3*D = C
//
// 2 * | 0 1 2 |   | 1 0 0 0 | + 3 * | 1 0 0 0 |   | 4 6 20 24 | + | 3  0  0 0  | = | 7  6  20 24 |
//     | 0 0 3 | * | 2 3 0 0 |       | 5 6 0 0 | = | 0 0 30 36 |   | 15 18 0 0  |   | 15 18 30 36 |
//     | 4 0 0 |   | 0 0 5 6 |       | 0 0 0 7 |   | 8 0 0  0  |   | 0  0  0 21 |   | 8  0  0  21 |
void test_cusparseTcsrgemm2() {
  std::vector<float> a_val_vec = {1, 2, 3, 4};
  Data<float> a_s_val(a_val_vec.data(), 4);
  Data<double> a_d_val(a_val_vec.data(), 4);
  Data<float2> a_c_val(a_val_vec.data(), 4);
  Data<double2> a_z_val(a_val_vec.data(), 4);
  std::vector<float> a_row_ptr_vec = {0, 2, 3, 4};
  Data<int> a_row_ptr(a_row_ptr_vec.data(), 4);
  std::vector<float> a_col_ind_vec = {1, 2, 2, 0};
  Data<int> a_col_ind(a_col_ind_vec.data(), 4);

  std::vector<float> b_val_vec = {1, 2, 3, 5, 6};
  Data<float> b_s_val(b_val_vec.data(), 5);
  Data<double> b_d_val(b_val_vec.data(), 5);
  Data<float2> b_c_val(b_val_vec.data(), 5);
  Data<double2> b_z_val(b_val_vec.data(), 5);
  std::vector<float> b_row_ptr_vec = {0, 1, 3, 5};
  Data<int> b_row_ptr(b_row_ptr_vec.data(), 4);
  std::vector<float> b_col_ind_vec = {0, 0, 1, 2, 3};
  Data<int> b_col_ind(b_col_ind_vec.data(), 5);

  std::vector<float> d_val_vec = {1, 5, 6, 7};
  Data<float> d_s_val(d_val_vec.data(), 4);
  Data<double> d_d_val(d_val_vec.data(), 4);
  Data<float2> d_c_val(d_val_vec.data(), 4);
  Data<double2> d_z_val(d_val_vec.data(), 4);
  std::vector<float> d_row_ptr_vec = {0, 1, 3, 4};
  Data<int> d_row_ptr(d_row_ptr_vec.data(), 4);
  std::vector<float> d_col_ind_vec = {0, 0, 1, 3};
  Data<int> d_col_ind(d_col_ind_vec.data(), 4);

  float alpha = 2;
  Data<float> alpha_s(&alpha, 1);
  Data<double> alpha_d(&alpha, 1);
  Data<float2> alpha_c(&alpha, 1);
  Data<double2> alpha_z(&alpha, 1);

  float beta = 3;
  Data<float> beta_s(&beta, 1);
  Data<double> beta_d(&beta, 1);
  Data<float2> beta_c(&beta, 1);
  Data<double2> beta_z(&beta, 1);

  cusparseHandle_t handle;
  cusparseCreate(&handle);

  cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE);

  a_s_val.H2D();
  a_d_val.H2D();
  a_c_val.H2D();
  a_z_val.H2D();
  a_row_ptr.H2D();
  a_col_ind.H2D();
  b_s_val.H2D();
  b_d_val.H2D();
  b_c_val.H2D();
  b_z_val.H2D();
  b_row_ptr.H2D();
  b_col_ind.H2D();
  d_s_val.H2D();
  d_d_val.H2D();
  d_c_val.H2D();
  d_z_val.H2D();
  d_row_ptr.H2D();
  d_col_ind.H2D();
  alpha_s.H2D();
  alpha_d.H2D();
  alpha_c.H2D();
  alpha_z.H2D();
  beta_s.H2D();
  beta_d.H2D();
  beta_c.H2D();
  beta_z.H2D();

  csrgemm2Info_t info_s;
  csrgemm2Info_t info_d;
  csrgemm2Info_t info_c;
  csrgemm2Info_t info_z;
  cusparseCreateCsrgemm2Info(&info_s);
  cusparseCreateCsrgemm2Info(&info_d);
  cusparseCreateCsrgemm2Info(&info_c);
  cusparseCreateCsrgemm2Info(&info_z);

  const int m = 3;
  const int n = 4;
  const int k = 3;
  const int nnzA = 4;
  const int nnzB = 5;
  const int nnzD = 4;

  cusparseMatDescr_t descrA;
  cusparseCreateMatDescr(&descrA);
  cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
  cusparseMatDescr_t descrB;
  cusparseCreateMatDescr(&descrB);
  cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO);
  cusparseMatDescr_t descrC;
  cusparseCreateMatDescr(&descrC);
  cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO);
  cusparseMatDescr_t descrD;
  cusparseCreateMatDescr(&descrD);
  cusparseSetMatType(descrD, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descrD, CUSPARSE_INDEX_BASE_ZERO);

  size_t ws_1_size_s = 0;
  size_t ws_1_size_d = 0;
  size_t ws_1_size_c = 0;
  size_t ws_1_size_z = 0;
  cusparseScsrgemm2_bufferSizeExt(handle, m, n, k, alpha_s.d_data, descrA, nnzA, a_row_ptr.d_data, a_col_ind.d_data, descrB, nnzB, b_row_ptr.d_data, b_col_ind.d_data, beta_s.d_data, descrD, nnzD, d_row_ptr.d_data, d_col_ind.d_data, info_s, &ws_1_size_s);
  cusparseDcsrgemm2_bufferSizeExt(handle, m, n, k, alpha_d.d_data, descrA, nnzA, a_row_ptr.d_data, a_col_ind.d_data, descrB, nnzB, b_row_ptr.d_data, b_col_ind.d_data, beta_d.d_data, descrD, nnzD, d_row_ptr.d_data, d_col_ind.d_data, info_d, &ws_1_size_d);
  cusparseCcsrgemm2_bufferSizeExt(handle, m, n, k, alpha_c.d_data, descrA, nnzA, a_row_ptr.d_data, a_col_ind.d_data, descrB, nnzB, b_row_ptr.d_data, b_col_ind.d_data, beta_c.d_data, descrD, nnzD, d_row_ptr.d_data, d_col_ind.d_data, info_c, &ws_1_size_c);
  cusparseZcsrgemm2_bufferSizeExt(handle, m, n, k, alpha_z.d_data, descrA, nnzA, a_row_ptr.d_data, a_col_ind.d_data, descrB, nnzB, b_row_ptr.d_data, b_col_ind.d_data, beta_z.d_data, descrD, nnzD, d_row_ptr.d_data, d_col_ind.d_data, info_z, &ws_1_size_z);

  void *ws_1_s = nullptr;
  void *ws_1_d = nullptr;
  void *ws_1_c = nullptr;
  void *ws_1_z = nullptr;

  cudaMalloc(&ws_1_s, ws_1_size_s);
  cudaMalloc(&ws_1_d, ws_1_size_d);
  cudaMalloc(&ws_1_c, ws_1_size_c);
  cudaMalloc(&ws_1_z, ws_1_size_z);

  Data<int> c_s_row_ptr(m + 1);
  Data<int> c_d_row_ptr(m + 1);
  Data<int> c_c_row_ptr(m + 1);
  Data<int> c_z_row_ptr(m + 1);

  Data<int> nnzC_s(1);
  Data<int> nnzC_d(1);
  Data<int> nnzC_c(1);
  Data<int> nnzC_z(1);
  cusparseXcsrgemm2Nnz(handle, m, n, k, descrA, nnzA, a_row_ptr.d_data, a_col_ind.d_data, descrB, nnzB, b_row_ptr.d_data, b_col_ind.d_data, descrD, nnzD, d_row_ptr.d_data, d_col_ind.d_data, descrC, c_s_row_ptr.d_data, nnzC_s.d_data, info_s, ws_1_s);
  cusparseXcsrgemm2Nnz(handle, m, n, k, descrA, nnzA, a_row_ptr.d_data, a_col_ind.d_data, descrB, nnzB, b_row_ptr.d_data, b_col_ind.d_data, descrD, nnzD, d_row_ptr.d_data, d_col_ind.d_data, descrC, c_d_row_ptr.d_data, nnzC_d.d_data, info_d, ws_1_d);
  cusparseXcsrgemm2Nnz(handle, m, n, k, descrA, nnzA, a_row_ptr.d_data, a_col_ind.d_data, descrB, nnzB, b_row_ptr.d_data, b_col_ind.d_data, descrD, nnzD, d_row_ptr.d_data, d_col_ind.d_data, descrC, c_c_row_ptr.d_data, nnzC_c.d_data, info_c, ws_1_c);
  cusparseXcsrgemm2Nnz(handle, m, n, k, descrA, nnzA, a_row_ptr.d_data, a_col_ind.d_data, descrB, nnzB, b_row_ptr.d_data, b_col_ind.d_data, descrD, nnzD, d_row_ptr.d_data, d_col_ind.d_data, descrC, c_z_row_ptr.d_data, nnzC_z.d_data, info_z, ws_1_z);

  cudaStreamSynchronize(0);

  nnzC_s.D2H();
  nnzC_d.D2H();
  nnzC_c.D2H();
  nnzC_z.D2H();

  int nnzC_s_int = *(nnzC_s.h_data);
  int nnzC_d_int = *(nnzC_d.h_data);
  int nnzC_c_int = *(nnzC_c.h_data);
  int nnzC_z_int = *(nnzC_z.h_data);

  Data<float> c_s_val(nnzC_s_int);
  Data<double> c_d_val(nnzC_d_int);
  Data<float2> c_c_val(nnzC_c_int);
  Data<double2> c_z_val(nnzC_z_int);
  Data<int> c_s_col_ind(nnzC_s_int);
  Data<int> c_d_col_ind(nnzC_d_int);
  Data<int> c_c_col_ind(nnzC_c_int);
  Data<int> c_z_col_ind(nnzC_z_int);

  cusparseScsrgemm2(handle, m, n, k, alpha_s.d_data, descrA, nnzA, a_s_val.d_data, a_row_ptr.d_data, a_col_ind.d_data, descrB, nnzB, b_s_val.d_data, b_row_ptr.d_data, b_col_ind.d_data, beta_s.d_data, descrD, nnzD, d_s_val.d_data, d_row_ptr.d_data, d_col_ind.d_data, descrC, c_s_val.d_data, c_s_row_ptr.d_data, c_s_col_ind.d_data, info_s, ws_1_s);
  cusparseDcsrgemm2(handle, m, n, k, alpha_d.d_data, descrA, nnzA, a_d_val.d_data, a_row_ptr.d_data, a_col_ind.d_data, descrB, nnzB, b_d_val.d_data, b_row_ptr.d_data, b_col_ind.d_data, beta_d.d_data, descrD, nnzD, d_d_val.d_data, d_row_ptr.d_data, d_col_ind.d_data, descrC, c_d_val.d_data, c_d_row_ptr.d_data, c_d_col_ind.d_data, info_d, ws_1_d);
  cusparseCcsrgemm2(handle, m, n, k, alpha_c.d_data, descrA, nnzA, a_c_val.d_data, a_row_ptr.d_data, a_col_ind.d_data, descrB, nnzB, b_c_val.d_data, b_row_ptr.d_data, b_col_ind.d_data, beta_c.d_data, descrD, nnzD, d_c_val.d_data, d_row_ptr.d_data, d_col_ind.d_data, descrC, c_c_val.d_data, c_c_row_ptr.d_data, c_c_col_ind.d_data, info_c, ws_1_c);
  cusparseZcsrgemm2(handle, m, n, k, alpha_z.d_data, descrA, nnzA, a_z_val.d_data, a_row_ptr.d_data, a_col_ind.d_data, descrB, nnzB, b_z_val.d_data, b_row_ptr.d_data, b_col_ind.d_data, beta_z.d_data, descrD, nnzD, d_z_val.d_data, d_row_ptr.d_data, d_col_ind.d_data, descrC, c_z_val.d_data, c_z_row_ptr.d_data, c_z_col_ind.d_data, info_z, ws_1_z);

  cudaStreamSynchronize(0);

  cudaFree(ws_1_s);
  cudaFree(ws_1_d);
  cudaFree(ws_1_c);
  cudaFree(ws_1_z);
  cusparseDestroyCsrgemm2Info(info_s);
  cusparseDestroyCsrgemm2Info(info_d);
  cusparseDestroyCsrgemm2Info(info_c);
  cusparseDestroyCsrgemm2Info(info_z);
  cusparseDestroy(handle);

  c_s_val.D2H();
  c_d_val.D2H();
  c_c_val.D2H();
  c_z_val.D2H();
  c_s_row_ptr.D2H();
  c_d_row_ptr.D2H();
  c_c_row_ptr.D2H();
  c_z_row_ptr.D2H();
  c_s_col_ind.D2H();
  c_d_col_ind.D2H();
  c_c_col_ind.D2H();
  c_z_col_ind.D2H();

  float expect_c_val[10] = {7, 6, 20, 24, 15, 18, 30, 36, 8, 21};
  float expect_c_row_ptr[4] = {0, 4, 8, 10};
  float expect_c_col_ind[10] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 3};
  if (compare_result(expect_c_val, c_s_val.h_data, 10) &&
      compare_result(expect_c_val, c_d_val.h_data, 10) &&
      compare_result(expect_c_val, c_c_val.h_data, 10) &&
      compare_result(expect_c_val, c_z_val.h_data, 10) &&
      compare_result(expect_c_row_ptr, c_s_row_ptr.h_data, 4) &&
      compare_result(expect_c_row_ptr, c_d_row_ptr.h_data, 4) &&
      compare_result(expect_c_row_ptr, c_c_row_ptr.h_data, 4) &&
      compare_result(expect_c_row_ptr, c_z_row_ptr.h_data, 4) &&
      compare_result(expect_c_col_ind, c_s_col_ind.h_data, 10) &&
      compare_result(expect_c_col_ind, c_d_col_ind.h_data, 10) &&
      compare_result(expect_c_col_ind, c_c_col_ind.h_data, 10) &&
      compare_result(expect_c_col_ind, c_z_col_ind.h_data, 10)
    )
    printf("Tcsrgemm2 pass\n");
  else {
    printf("Tcsrgemm2 fail\n");
    test_passed = false;
  }
}

int main() {
  test_cusparseTcsrsv2();
  test_cusparseTcsrgemm2();

  if (test_passed)
    return 0;
  return -1;
}
