// ===------- cusparse_11.cu ------------------------------- *- CUDA -* ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include "cusparse.h"
#include <iostream>
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

// 3*3     3*2             3*2
// op(A) * op(X) = alpha * op(B)
// 1 0 0   1 4       1   * 1  4
// 0 2 0   2 5             4  10
// 0 4 3   3 6             17 38
void test_cusparseTcsrsm2() {
  const int nrhs = 2;
  const int m = 3;
  const int nnz = 4;

  std::vector<float> a_val_vec = {1, 2, 4, 3};
  Data<float> a_s_val(a_val_vec.data(), 4);
  Data<double> a_d_val(a_val_vec.data(), 4);
  Data<float2> a_c_val(a_val_vec.data(), 4);
  Data<double2> a_z_val(a_val_vec.data(), 4);
  std::vector<float> a_row_ptr_vec = {0, 1, 2, 4};
  Data<int> a_row_ptr_s(a_row_ptr_vec.data(), 4);
  Data<int> a_row_ptr_d(a_row_ptr_vec.data(), 4);
  Data<int> a_row_ptr_c(a_row_ptr_vec.data(), 4);
  Data<int> a_row_ptr_z(a_row_ptr_vec.data(), 4);
  std::vector<float> a_col_ind_vec = {0, 1, 1, 2};
  Data<int> a_col_ind_s(a_col_ind_vec.data(), 4);
  Data<int> a_col_ind_d(a_col_ind_vec.data(), 4);
  Data<int> a_col_ind_c(a_col_ind_vec.data(), 4);
  Data<int> a_col_ind_z(a_col_ind_vec.data(), 4);

  std::vector<float> b_vec = {1, 4, 4, 10, 17, 38};
  Data<float> b_s(b_vec.data(), m * nrhs);
  Data<double> b_d(b_vec.data(), m * nrhs);
  Data<float2> b_c(b_vec.data(), m * nrhs);
  Data<double2> b_z(b_vec.data(), m * nrhs);

  cusparseHandle_t handle;
  cusparseCreate(&handle);
  csrsm2Info_t info_s;
  csrsm2Info_t info_d;
  csrsm2Info_t info_c;
  csrsm2Info_t info_z;
  cusparseCreateCsrsm2Info(&info_s);
  cusparseCreateCsrsm2Info(&info_d);
  cusparseCreateCsrsm2Info(&info_c);
  cusparseCreateCsrsm2Info(&info_z);
  cusparseSolvePolicy_t policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
  policy = CUSPARSE_SOLVE_POLICY_NO_LEVEL;

  cusparseMatDescr_t descrA;
  cusparseCreateMatDescr(&descrA);
  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_TRIANGULAR);
  cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT);
  cusparseSetMatFillMode(descrA, CUSPARSE_FILL_MODE_LOWER);

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
  b_s.H2D();
  b_d.H2D();
  b_c.H2D();
  b_z.H2D();

  float alpha_s = 1;
  double alpha_d = 1;
  float2 alpha_c = float2{1, 0};
  double2 alpha_z = double2{1, 0};

  size_t buffer_size_s;
  size_t buffer_size_d;
  size_t buffer_size_c;
  size_t buffer_size_z;
  cusparseScsrsm2_bufferSizeExt(handle, 0, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, m, nrhs, nnz, &alpha_s, descrA, (float *)a_s_val.d_data, (int *)a_row_ptr_s.d_data, (int *)a_col_ind_s.d_data, (float *)b_s.d_data, nrhs, info_s, policy, &buffer_size_s);
  cusparseDcsrsm2_bufferSizeExt(handle, 0, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, m, nrhs, nnz, &alpha_d, descrA, (double *)a_d_val.d_data, (int *)a_row_ptr_d.d_data, (int *)a_col_ind_d.d_data, (double *)b_d.d_data, nrhs, info_d, policy, &buffer_size_d);
  cusparseCcsrsm2_bufferSizeExt(handle, 0, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, m, nrhs, nnz, &alpha_c, descrA, (float2 *)a_c_val.d_data, (int *)a_row_ptr_c.d_data, (int *)a_col_ind_c.d_data, (float2 *)b_c.d_data, nrhs, info_c, policy, &buffer_size_c);
  cusparseZcsrsm2_bufferSizeExt(handle, 0, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, m, nrhs, nnz, &alpha_z, descrA, (double2 *)a_z_val.d_data, (int *)a_row_ptr_z.d_data, (int *)a_col_ind_z.d_data, (double2 *)b_z.d_data, nrhs, info_z, policy, &buffer_size_z);

  void* buffer_s;
  void* buffer_d;
  void* buffer_c;
  void* buffer_z;
  cudaMalloc(&buffer_s, buffer_size_s);
  cudaMalloc(&buffer_d, buffer_size_d);
  cudaMalloc(&buffer_c, buffer_size_c);
  cudaMalloc(&buffer_z, buffer_size_z);

  cusparseScsrsm2_analysis(handle, 0, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, m, nrhs, nnz, &alpha_s, descrA, (float *)a_s_val.d_data, (int *)a_row_ptr_s.d_data, (int *)a_col_ind_s.d_data, (float *)b_s.d_data, nrhs, info_s, policy, buffer_s);
  cusparseDcsrsm2_analysis(handle, 0, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, m, nrhs, nnz, &alpha_d, descrA, (double *)a_d_val.d_data, (int *)a_row_ptr_d.d_data, (int *)a_col_ind_d.d_data, (double *)b_d.d_data, nrhs, info_d, policy, buffer_d);
  cusparseCcsrsm2_analysis(handle, 0, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, m, nrhs, nnz, &alpha_c, descrA, (float2 *)a_c_val.d_data, (int *)a_row_ptr_c.d_data, (int *)a_col_ind_c.d_data, (float2 *)b_c.d_data, nrhs, info_c, policy, buffer_c);
  cusparseZcsrsm2_analysis(handle, 0, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, m, nrhs, nnz, &alpha_z, descrA, (double2 *)a_z_val.d_data, (int *)a_row_ptr_z.d_data, (int *)a_col_ind_z.d_data, (double2 *)b_z.d_data, nrhs, info_z, policy, buffer_z);

  cusparseScsrsm2_solve(handle, 0, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, m, nrhs, nnz, &alpha_s, descrA, (float *)a_s_val.d_data, (int *)a_row_ptr_s.d_data, (int *)a_col_ind_s.d_data, (float *)b_s.d_data, nrhs, info_s, policy, buffer_s);
  cusparseDcsrsm2_solve(handle, 0, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, m, nrhs, nnz, &alpha_d, descrA, (double *)a_d_val.d_data, (int *)a_row_ptr_d.d_data, (int *)a_col_ind_d.d_data, (double *)b_d.d_data, nrhs, info_d, policy, buffer_d);
  cusparseCcsrsm2_solve(handle, 0, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, m, nrhs, nnz, &alpha_c, descrA, (float2 *)a_c_val.d_data, (int *)a_row_ptr_c.d_data, (int *)a_col_ind_c.d_data, (float2 *)b_c.d_data, nrhs, info_c, policy, buffer_c);
  cusparseZcsrsm2_solve(handle, 0, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, m, nrhs, nnz, &alpha_z, descrA, (double2 *)a_z_val.d_data, (int *)a_row_ptr_z.d_data, (int *)a_col_ind_z.d_data, (double2 *)b_z.d_data, nrhs, info_z, policy, buffer_z);

  cudaStreamSynchronize(0);

  b_s.D2H();
  b_d.D2H();
  b_c.D2H();
  b_z.D2H();

  cusparseDestroyCsrsm2Info(info_s);
  cusparseDestroyCsrsm2Info(info_d);
  cusparseDestroyCsrsm2Info(info_c);
  cusparseDestroyCsrsm2Info(info_z);
  cusparseDestroyMatDescr(descrA);
  cusparseDestroy(handle);
  cudaFree(buffer_s);
  cudaFree(buffer_d);
  cudaFree(buffer_c);
  cudaFree(buffer_z);

  std::cout << "Solution x: ";
  for (int i = 0; i < m * nrhs; ++i) {
    std::cout << b_s.h_data[i] << " ";
  }
  std::cout << std::endl;

  float expect_x[6] = {1, 4, 2, 5, 3, 6};
  if (compare_result(expect_x, b_s.h_data, 6) &&
      compare_result(expect_x, b_d.h_data, 6) &&
      compare_result(expect_x, b_c.h_data, 6) &&
      compare_result(expect_x, b_z.h_data, 6))
    printf("Tcsrsm2 pass\n");
  else {
    printf("Tcsrsm2 fail\n");
    test_passed = false;
  }
}

int main() {
  test_cusparseTcsrsm2();

  if (test_passed)
    return 0;
  return -1;
}
