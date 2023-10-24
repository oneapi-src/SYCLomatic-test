// ===------- cusparse_2.cu -------------------------------- *- CUDA -* ----===//
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

const bool run_complex_datatype = true;

void test_cusparseSetGetStream() {
  cusparseHandle_t handle;
  cusparseCreate(&handle);
  cudaStream_t stream;
  cusparseGetStream(handle, &stream);
  cusparseSetStream(handle, stream);
  cusparseDestroy(handle);
  printf("SetGetStream pass\n");
  test_passed = true;
}

void test_cusparseTcsrmv_ge() {
  std::vector<float> a_val_vec = {1, 4, 2, 3, 5, 7, 8, 9, 6};
  Data<float> a_s_val(a_val_vec.data(), 9);
  Data<double> a_d_val(a_val_vec.data(), 9);
  Data<float2> a_c_val(a_val_vec.data(), 9);
  Data<double2> a_z_val(a_val_vec.data(), 9);
  std::vector<float> a_row_ptr_vec = {0, 2, 4, 7, 9};
  Data<int> a_row_ptr(a_row_ptr_vec.data(), 5);
  std::vector<float> a_col_ind_vec = {0, 1, 1, 2, 0, 3, 4, 2, 4};
  Data<int> a_col_ind(a_col_ind_vec.data(), 9);

  std::vector<float> b_vec = {1, 2, 3, 4, 5};
  Data<float> b_s(b_vec.data(), 5);
  Data<double> b_d(b_vec.data(), 5);
  Data<float2> b_c(b_vec.data(), 5);
  Data<double2> b_z(b_vec.data(), 5);

  Data<float> c_s(4);
  Data<double> c_d(4);
  Data<float2> c_c(4);
  Data<double2> c_z(4);

  float alpha = 10;
  Data<float> alpha_s(&alpha, 1);
  Data<double> alpha_d(&alpha, 1);
  Data<float2> alpha_c(&alpha, 1);
  Data<double2> alpha_z(&alpha, 1);

  float beta = 0;
  Data<float> beta_s(&beta, 1);
  Data<double> beta_d(&beta, 1);
  Data<float2> beta_c(&beta, 1);
  Data<double2> beta_z(&beta, 1);

  cusparseHandle_t handle;
  cusparseCreate(&handle);

  cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE);

  cusparseMatDescr_t descrA;
  cusparseCreateMatDescr(&descrA);
  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);

  a_s_val.H2D();
  a_d_val.H2D();
  a_c_val.H2D();
  a_z_val.H2D();
  a_row_ptr.H2D();
  a_col_ind.H2D();
  b_s.H2D();
  b_d.H2D();
  b_c.H2D();
  b_z.H2D();
  alpha_s.H2D();
  alpha_d.H2D();
  alpha_c.H2D();
  alpha_z.H2D();
  beta_s.H2D();
  beta_d.H2D();
  beta_c.H2D();
  beta_z.H2D();

  cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 4, 5, 9, (float *)alpha_s.d_data, descrA, (float *)a_s_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, (float *)b_s.d_data, (float *)beta_s.d_data, (float *)c_s.d_data);
  cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 4, 5, 9, (double *)alpha_d.d_data, descrA, (double *)a_d_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, (double *)b_d.d_data, (double *)beta_d.d_data, (double *)c_d.d_data);
  if (run_complex_datatype) {
    cusparseCcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 4, 5, 9, (float2 *)alpha_c.d_data, descrA, (float2 *)a_c_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, (float2 *)b_c.d_data, (float2 *)beta_c.d_data, (float2 *)c_c.d_data);
    cusparseZcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 4, 5, 9, (double2 *)alpha_z.d_data, descrA, (double2 *)a_z_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, (double2 *)b_z.d_data, (double2 *)beta_z.d_data, (double2 *)c_z.d_data);
  }

  c_s.D2H();
  c_d.D2H();
  c_c.D2H();
  c_z.D2H();

  cudaStreamSynchronize(0);
  cusparseDestroyMatDescr(descrA);
  cusparseDestroy(handle);

  float expect_c[4] = {90, 130, 730, 570};
  if (compare_result(expect_c, c_s.h_data, 4) &&
      compare_result(expect_c, c_d.h_data, 4) &&
      compare_result(expect_c, c_c.h_data, 4) &&
      compare_result(expect_c, c_z.h_data, 4))
    printf("Tcsrmv_ge pass\n");
  else {
    printf("Tcsrmv_ge fail\n");
    test_passed = false;
  }
}


//  alpha  *  A          *  B      =  C
//     10  * | 1 4 0 1 |    | 1 |    | 130 |
//           | 4 2 3 0 |    | 2 |    | 170 |
//           | 0 3 0 7 |    | 3 |    | 340 |
//           | 1 0 7 0 |    | 4 |    | 220 |
void test_cusparseTcsrmv_sy() {
  std::vector<float> a_val_vec = {1, 4, 1, 4, 2, 3, 3, 7, 1, 7};
  Data<float> a_s_val(a_val_vec.data(), 10);
  Data<double> a_d_val(a_val_vec.data(), 10);
  Data<float2> a_c_val(a_val_vec.data(), 10);
  Data<double2> a_z_val(a_val_vec.data(), 10);
  std::vector<float> a_row_ptr_vec = {0, 3, 6, 8, 10};
  Data<int> a_row_ptr(a_row_ptr_vec.data(), 5);
  std::vector<float> a_col_ind_vec = {0, 1, 3, 0, 1, 2, 1, 3, 0, 2};
  Data<int> a_col_ind(a_col_ind_vec.data(), 10);

  std::vector<float> b_vec = {1, 2, 3, 4};
  Data<float> b_s(b_vec.data(), 4);
  Data<double> b_d(b_vec.data(), 4);
  Data<float2> b_c(b_vec.data(), 4);
  Data<double2> b_z(b_vec.data(), 4);

  Data<float> c_s(4);
  Data<double> c_d(4);
  Data<float2> c_c(4);
  Data<double2> c_z(4);

  float alpha = 10;
  Data<float> alpha_s(&alpha, 1);
  Data<double> alpha_d(&alpha, 1);
  Data<float2> alpha_c(&alpha, 1);
  Data<double2> alpha_z(&alpha, 1);

  float beta = 0;
  Data<float> beta_s(&beta, 1);
  Data<double> beta_d(&beta, 1);
  Data<float2> beta_c(&beta, 1);
  Data<double2> beta_z(&beta, 1);

  cusparseHandle_t handle;
  cusparseCreate(&handle);

  cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE);

  cusparseMatDescr_t descrA;
  cusparseCreateMatDescr(&descrA);
  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_SYMMETRIC);

  a_s_val.H2D();
  a_d_val.H2D();
  a_c_val.H2D();
  a_z_val.H2D();
  a_row_ptr.H2D();
  a_col_ind.H2D();
  b_s.H2D();
  b_d.H2D();
  b_c.H2D();
  b_z.H2D();
  alpha_s.H2D();
  alpha_d.H2D();
  alpha_c.H2D();
  alpha_z.H2D();
  beta_s.H2D();
  beta_d.H2D();
  beta_c.H2D();
  beta_z.H2D();

  cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 4, 4, 10, (float *)alpha_s.d_data, descrA, (float *)a_s_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, (float *)b_s.d_data, (float *)beta_s.d_data, (float *)c_s.d_data);
  cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 4, 4, 10, (double *)alpha_d.d_data, descrA, (double *)a_d_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, (double *)b_d.d_data, (double *)beta_d.d_data, (double *)c_d.d_data);
  if (run_complex_datatype) {
    cusparseCcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 4, 4, 10, (float2 *)alpha_c.d_data, descrA, (float2 *)a_c_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, (float2 *)b_c.d_data, (float2 *)beta_c.d_data, (float2 *)c_c.d_data);
    cusparseZcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 4, 4, 10, (double2 *)alpha_z.d_data, descrA, (double2 *)a_z_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, (double2 *)b_z.d_data, (double2 *)beta_z.d_data, (double2 *)c_z.d_data);
  }

  c_s.D2H();
  c_d.D2H();
  c_c.D2H();
  c_z.D2H();

  cudaStreamSynchronize(0);
  cusparseDestroyMatDescr(descrA);
  cusparseDestroy(handle);

  float expect_c[4] = {130, 170, 340, 220};
  if (compare_result(expect_c, c_s.h_data, 4) &&
      compare_result(expect_c, c_d.h_data, 4) &&
      compare_result(expect_c, c_c.h_data, 4) &&
      compare_result(expect_c, c_z.h_data, 4))
    printf("Tcsrmv_sy pass\n");
  else {
    printf("Tcsrmv_sy fail\n");
    test_passed = false;
  }
}

//  alpha  *  A          *  B      =  C
//     10  * | 1 4 0 1 |    | 1 |    | 130 |
//           | 0 2 3 0 |    | 2 |    | 130 |
//           | 0 0 0 7 |    | 3 |    | 280 |
//           | 0 0 0 1 |    | 4 |    | 40  |

// Note: this matrix type is not supported in CUDA but supported in oneMKL
void test_cusparseTcsrmv_tr() {
  std::vector<float> a_val_vec = {1, 4, 1, 2, 3, 7, 1};
  Data<float> a_s_val(a_val_vec.data(), 7);
  Data<double> a_d_val(a_val_vec.data(), 7);
  Data<float2> a_c_val(a_val_vec.data(), 7);
  Data<double2> a_z_val(a_val_vec.data(), 7);
  std::vector<float> a_row_ptr_vec = {0, 3, 5, 6, 7};
  Data<int> a_row_ptr(a_row_ptr_vec.data(), 5);
  std::vector<float> a_col_ind_vec = {0, 1, 3, 1, 2, 3, 3};
  Data<int> a_col_ind(a_col_ind_vec.data(), 7);

  std::vector<float> b_vec = {1, 2, 3, 4};
  Data<float> b_s(b_vec.data(), 4);
  Data<double> b_d(b_vec.data(), 4);
  Data<float2> b_c(b_vec.data(), 4);
  Data<double2> b_z(b_vec.data(), 4);

  Data<float> c_s(4);
  Data<double> c_d(4);
  Data<float2> c_c(4);
  Data<double2> c_z(4);

  float alpha = 10;
  Data<float> alpha_s(&alpha, 1);
  Data<double> alpha_d(&alpha, 1);
  Data<float2> alpha_c(&alpha, 1);
  Data<double2> alpha_z(&alpha, 1);

  float beta = 0;
  Data<float> beta_s(&beta, 1);
  Data<double> beta_d(&beta, 1);
  Data<float2> beta_c(&beta, 1);
  Data<double2> beta_z(&beta, 1);

  cusparseHandle_t handle;
  cusparseCreate(&handle);

  cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE);

  cusparseMatDescr_t descrA;
  cusparseCreateMatDescr(&descrA);
  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_TRIANGULAR);
  cusparseSetMatFillMode(descrA, CUSPARSE_FILL_MODE_UPPER);

  a_s_val.H2D();
  a_d_val.H2D();
  a_c_val.H2D();
  a_z_val.H2D();
  a_row_ptr.H2D();
  a_col_ind.H2D();
  b_s.H2D();
  b_d.H2D();
  b_c.H2D();
  b_z.H2D();
  alpha_s.H2D();
  alpha_d.H2D();
  alpha_c.H2D();
  alpha_z.H2D();
  beta_s.H2D();
  beta_d.H2D();
  beta_c.H2D();
  beta_z.H2D();

  cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 4, 4, 7, (float *)alpha_s.d_data, descrA, (float *)a_s_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, (float *)b_s.d_data, (float *)beta_s.d_data, (float *)c_s.d_data);
  cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 4, 4, 7, (double *)alpha_d.d_data, descrA, (double *)a_d_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, (double *)b_d.d_data, (double *)beta_d.d_data, (double *)c_d.d_data);
  if (run_complex_datatype) {
    cusparseCcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 4, 4, 7, (float2 *)alpha_c.d_data, descrA, (float2 *)a_c_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, (float2 *)b_c.d_data, (float2 *)beta_c.d_data, (float2 *)c_c.d_data);
    cusparseZcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 4, 4, 7, (double2 *)alpha_z.d_data, descrA, (double2 *)a_z_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, (double2 *)b_z.d_data, (double2 *)beta_z.d_data, (double2 *)c_z.d_data);
  }

  c_s.D2H();
  c_d.D2H();
  c_c.D2H();
  c_z.D2H();

  cudaStreamSynchronize(0);
  cusparseDestroyMatDescr(descrA);
  cusparseDestroy(handle);

  float expect_c[4] = {130, 130, 280, 40};
  if (compare_result(expect_c, c_s.h_data, 4) &&
      compare_result(expect_c, c_d.h_data, 4) &&
      compare_result(expect_c, c_c.h_data, 4) &&
      compare_result(expect_c, c_z.h_data, 4))
    printf("Tcsrmv_tr pass\n");
  else {
    printf("Tcsrmv_tr fail\n");
    test_passed = false;
  }
}

void test_cusparseTcsrmm() {
  std::vector<float> a_val_vec = {1, 4, 2, 3, 5, 7, 8, 9, 6};
  Data<float> a_s_val(a_val_vec.data(), 9);
  Data<double> a_d_val(a_val_vec.data(), 9);
  Data<float2> a_c_val(a_val_vec.data(), 9);
  Data<double2> a_z_val(a_val_vec.data(), 9);
  std::vector<float> a_row_ptr_vec = {0, 2, 4, 7, 9};
  Data<int> a_row_ptr(a_row_ptr_vec.data(), 5);
  std::vector<float> a_col_ind_vec = {0, 1, 1, 2, 0, 3, 4, 2, 4};
  Data<int> a_col_ind(a_col_ind_vec.data(), 9);

  std::vector<float> b_vec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  Data<float> b_s(b_vec.data(), 10);
  Data<double> b_d(b_vec.data(), 10);
  Data<float2> b_c(b_vec.data(), 10);
  Data<double2> b_z(b_vec.data(), 10);

  Data<float> c_s(8);
  Data<double> c_d(8);
  Data<float2> c_c(8);
  Data<double2> c_z(8);

  float alpha = 10;
  Data<float> alpha_s(&alpha, 1);
  Data<double> alpha_d(&alpha, 1);
  Data<float2> alpha_c(&alpha, 1);
  Data<double2> alpha_z(&alpha, 1);

  float beta = 0;
  Data<float> beta_s(&beta, 1);
  Data<double> beta_d(&beta, 1);
  Data<float2> beta_c(&beta, 1);
  Data<double2> beta_z(&beta, 1);

  cusparseHandle_t handle;
  cusparseCreate(&handle);

  cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE);

  cusparseMatDescr_t descrA;
  cusparseCreateMatDescr(&descrA);
  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);

  a_s_val.H2D();
  a_d_val.H2D();
  a_c_val.H2D();
  a_z_val.H2D();
  a_row_ptr.H2D();
  a_col_ind.H2D();
  b_s.H2D();
  b_d.H2D();
  b_c.H2D();
  b_z.H2D();
  alpha_s.H2D();
  alpha_d.H2D();
  alpha_c.H2D();
  alpha_z.H2D();
  beta_s.H2D();
  beta_d.H2D();
  beta_c.H2D();
  beta_z.H2D();

  cusparseScsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 4, 2, 5, 9, (float *)alpha_s.d_data, descrA, (float *)a_s_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, (float *)b_s.d_data, 5, (float *)beta_s.d_data, (float *)c_s.d_data, 4);
  cusparseDcsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 4, 2, 5, 9, (double *)alpha_d.d_data, descrA, (double *)a_d_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, (double *)b_d.d_data, 5, (double *)beta_d.d_data, (double *)c_d.d_data, 4);
  if (run_complex_datatype) {
    cusparseCcsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 4, 2, 5, 9, (float2 *)alpha_c.d_data, descrA, (float2 *)a_c_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, (float2 *)b_c.d_data, 5, (float2 *)beta_c.d_data, (float2 *)c_c.d_data, 4);
    cusparseZcsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 4, 2, 5, 9, (double2 *)alpha_z.d_data, descrA, (double2 *)a_z_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, (double2 *)b_z.d_data, 5, (double2 *)beta_z.d_data, (double2 *)c_z.d_data, 4);
  }

  c_s.D2H();
  c_d.D2H();
  c_c.D2H();
  c_z.D2H();

  cudaStreamSynchronize(0);
  cusparseDestroyMatDescr(descrA);
  cusparseDestroy(handle);

  float expect_c[8] = {90, 130, 730, 570, 340, 380, 1730, 1320};
  if (compare_result(expect_c, c_s.h_data, 8) &&
      compare_result(expect_c, c_d.h_data, 8) &&
      compare_result(expect_c, c_c.h_data, 8) &&
      compare_result(expect_c, c_z.h_data, 8))
    printf("Tcsrmm pass\n");
  else {
    printf("Tcsrmm fail\n");
    test_passed = false;
  }
}

// A * x = f
//
// | 1 1 2 |   | 1 |   | 9  |  
// | 0 1 3 | * | 2 | = | 11 |
// | 0 0 1 |   | 3 |   | 3  |
void test_cusparseTcsrsv() {
  std::vector<float> a_val_vec = {1, 1, 2, 1, 3, 1};
  Data<float> a_s_val(a_val_vec.data(), 6);
  Data<double> a_d_val(a_val_vec.data(), 6);
  Data<float2> a_c_val(a_val_vec.data(), 6);
  Data<double2> a_z_val(a_val_vec.data(), 6);
  std::vector<float> a_row_ptr_vec = {0, 3, 5, 6};
  Data<int> a_row_ptr(a_row_ptr_vec.data(), 4);
  std::vector<float> a_col_ind_vec = {0, 1, 2, 1, 2, 3};
  Data<int> a_col_ind(a_col_ind_vec.data(), 6);

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
  cusparseSolveAnalysisInfo_t info;
  cusparseCreateSolveAnalysisInfo(&info);

  cusparseMatDescr_t descrA;
  cusparseCreateMatDescr(&descrA);
  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_TRIANGULAR);

  a_s_val.H2D();
  a_d_val.H2D();
  a_c_val.H2D();
  a_z_val.H2D();
  a_row_ptr.H2D();
  a_col_ind.H2D();
  f_s.H2D();
  f_d.H2D();
  f_c.H2D();
  f_z.H2D();

  cusparseScsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA, (float *)a_s_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, info);
  cusparseDcsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA, (double *)a_d_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, info);
  if (run_complex_datatype) {
    cusparseCcsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA, (float2 *)a_c_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, info);
    cusparseZcsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA, (double2 *)a_z_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, info);
  }

  float alpha_s = 1;
  double alpha_d = 1;
  float2 alpha_c = float2{1, 0};
  double2 alpha_z = double2{1, 0};

  cusparseScsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, &alpha_s, descrA, (float *)a_s_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, info, f_s.d_data, x_s.d_data);
  cusparseDcsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, &alpha_d, descrA, (double *)a_d_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, info, f_d.d_data, x_d.d_data);
  if (run_complex_datatype) {
    cusparseCcsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, &alpha_c, descrA, (float2 *)a_c_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, info, f_c.d_data, x_c.d_data);
    cusparseZcsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, &alpha_z, descrA, (double2 *)a_z_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, info, f_z.d_data, x_z.d_data);
  }

  x_s.D2H();
  x_d.D2H();
  x_c.D2H();
  x_z.D2H();

  cudaStreamSynchronize(0);
  cusparseDestroySolveAnalysisInfo(info);
  cusparseDestroyMatDescr(descrA);
  cusparseDestroy(handle);

  float expect_x[4] = {1, 2, 3};
  if (compare_result(expect_x, x_s.h_data, 3) &&
      compare_result(expect_x, x_d.h_data, 3) &&
      compare_result(expect_x, x_c.h_data, 3) &&
      compare_result(expect_x, x_z.h_data, 3))
    printf("Tcsrsv pass\n");
  else {
    printf("Tcsrsv fail\n");
    test_passed = false;
  }
  test_passed = true;
}

void test_cusparseTcsrmv_mp() {
  std::vector<float> a_val_vec = {1, 4, 2, 3, 5, 7, 8, 9, 6};
  Data<float> a_s_val(a_val_vec.data(), 9);
  Data<double> a_d_val(a_val_vec.data(), 9);
  Data<float2> a_c_val(a_val_vec.data(), 9);
  Data<double2> a_z_val(a_val_vec.data(), 9);
  std::vector<float> a_row_ptr_vec = {0, 2, 4, 7, 9};
  Data<int> a_row_ptr(a_row_ptr_vec.data(), 5);
  std::vector<float> a_col_ind_vec = {0, 1, 1, 2, 0, 3, 4, 2, 4};
  Data<int> a_col_ind(a_col_ind_vec.data(), 9);

  std::vector<float> b_vec = {1, 2, 3, 4, 5};
  Data<float> b_s(b_vec.data(), 5);
  Data<double> b_d(b_vec.data(), 5);
  Data<float2> b_c(b_vec.data(), 5);
  Data<double2> b_z(b_vec.data(), 5);

  Data<float> c_s(4);
  Data<double> c_d(4);
  Data<float2> c_c(4);
  Data<double2> c_z(4);

  float alpha = 10;
  Data<float> alpha_s(&alpha, 1);
  Data<double> alpha_d(&alpha, 1);
  Data<float2> alpha_c(&alpha, 1);
  Data<double2> alpha_z(&alpha, 1);

  float beta = 0;
  Data<float> beta_s(&beta, 1);
  Data<double> beta_d(&beta, 1);
  Data<float2> beta_c(&beta, 1);
  Data<double2> beta_z(&beta, 1);

  cusparseHandle_t handle;
  cusparseCreate(&handle);

  cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE);

  cusparseMatDescr_t descrA;
  cusparseCreateMatDescr(&descrA);
  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);

  a_s_val.H2D();
  a_d_val.H2D();
  a_c_val.H2D();
  a_z_val.H2D();
  a_row_ptr.H2D();
  a_col_ind.H2D();
  b_s.H2D();
  b_d.H2D();
  b_c.H2D();
  b_z.H2D();
  alpha_s.H2D();
  alpha_d.H2D();
  alpha_c.H2D();
  alpha_z.H2D();
  beta_s.H2D();
  beta_d.H2D();
  beta_c.H2D();
  beta_z.H2D();

  cusparseScsrmv_mp(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 4, 5, 9, (float *)alpha_s.d_data, descrA, (float *)a_s_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, (float *)b_s.d_data, (float *)beta_s.d_data, (float *)c_s.d_data);
  cusparseDcsrmv_mp(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 4, 5, 9, (double *)alpha_d.d_data, descrA, (double *)a_d_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, (double *)b_d.d_data, (double *)beta_d.d_data, (double *)c_d.d_data);
  if (run_complex_datatype) {
    cusparseCcsrmv_mp(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 4, 5, 9, (float2 *)alpha_c.d_data, descrA, (float2 *)a_c_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, (float2 *)b_c.d_data, (float2 *)beta_c.d_data, (float2 *)c_c.d_data);
    cusparseZcsrmv_mp(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 4, 5, 9, (double2 *)alpha_z.d_data, descrA, (double2 *)a_z_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, (double2 *)b_z.d_data, (double2 *)beta_z.d_data, (double2 *)c_z.d_data);
  }

  c_s.D2H();
  c_d.D2H();
  c_c.D2H();
  c_z.D2H();

  cudaStreamSynchronize(0);
  cusparseDestroyMatDescr(descrA);
  cusparseDestroy(handle);

  float expect_c[4] = {90, 130, 730, 570};
  if (compare_result(expect_c, c_s.h_data, 4) &&
      compare_result(expect_c, c_d.h_data, 4) &&
      compare_result(expect_c, c_c.h_data, 4) &&
      compare_result(expect_c, c_z.h_data, 4))
    printf("Tcsrmv_mp pass\n");
  else {
    printf("Tcsrmv_mp fail\n");
    test_passed = false;
  }
}

void test_cusparseCsrmvEx() {
  std::vector<float> a_val_vec = {1, 4, 2, 3, 5, 7, 8, 9, 6};
  Data<float> a_s_val(a_val_vec.data(), 9);
  Data<double> a_d_val(a_val_vec.data(), 9);
  Data<float2> a_c_val(a_val_vec.data(), 9);
  Data<double2> a_z_val(a_val_vec.data(), 9);
  std::vector<float> a_row_ptr_vec = {0, 2, 4, 7, 9};
  Data<int> a_row_ptr(a_row_ptr_vec.data(), 5);
  std::vector<float> a_col_ind_vec = {0, 1, 1, 2, 0, 3, 4, 2, 4};
  Data<int> a_col_ind(a_col_ind_vec.data(), 9);

  std::vector<float> b_vec = {1, 2, 3, 4, 5};
  Data<float> b_s(b_vec.data(), 5);
  Data<double> b_d(b_vec.data(), 5);
  Data<float2> b_c(b_vec.data(), 5);
  Data<double2> b_z(b_vec.data(), 5);

  Data<float> c_s(4);
  Data<double> c_d(4);
  Data<float2> c_c(4);
  Data<double2> c_z(4);

  float alpha = 10;
  Data<float> alpha_s(&alpha, 1);
  Data<double> alpha_d(&alpha, 1);
  Data<float2> alpha_c(&alpha, 1);
  Data<double2> alpha_z(&alpha, 1);

  float beta = 0;
  Data<float> beta_s(&beta, 1);
  Data<double> beta_d(&beta, 1);
  Data<float2> beta_c(&beta, 1);
  Data<double2> beta_z(&beta, 1);

  cusparseHandle_t handle;
  cusparseCreate(&handle);

  cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE);

  cusparseMatDescr_t descrA;
  cusparseCreateMatDescr(&descrA);
  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);

  a_s_val.H2D();
  a_d_val.H2D();
  a_c_val.H2D();
  a_z_val.H2D();
  a_row_ptr.H2D();
  a_col_ind.H2D();
  b_s.H2D();
  b_d.H2D();
  b_c.H2D();
  b_z.H2D();
  alpha_s.H2D();
  alpha_d.H2D();
  alpha_c.H2D();
  alpha_z.H2D();
  beta_s.H2D();
  beta_d.H2D();
  beta_c.H2D();
  beta_z.H2D();

  cusparseAlgMode_t alg;

  size_t ws_size_s;
  size_t ws_size_d;
  size_t ws_size_c;
  size_t ws_size_z;
  cusparseCsrmvEx_bufferSize(handle, alg, CUSPARSE_OPERATION_NON_TRANSPOSE, 4, 5, 9, alpha_s.d_data, CUDA_R_32F, descrA, a_s_val.d_data, CUDA_R_32F, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, b_s.d_data, CUDA_R_32F, beta_s.d_data, CUDA_R_32F, c_s.d_data, CUDA_R_32F, CUDA_R_32F, &ws_size_s);
  cusparseCsrmvEx_bufferSize(handle, alg, CUSPARSE_OPERATION_NON_TRANSPOSE, 4, 5, 9, alpha_d.d_data, CUDA_R_64F, descrA, a_d_val.d_data, CUDA_R_64F, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, b_d.d_data, CUDA_R_64F, beta_d.d_data, CUDA_R_64F, c_d.d_data, CUDA_R_64F, CUDA_R_64F, &ws_size_d);
  if (run_complex_datatype) {
    cusparseCsrmvEx_bufferSize(handle, alg, CUSPARSE_OPERATION_NON_TRANSPOSE, 4, 5, 9, alpha_c.d_data, CUDA_C_32F, descrA, a_c_val.d_data, CUDA_C_32F, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, b_c.d_data, CUDA_C_32F, beta_c.d_data, CUDA_C_32F, c_c.d_data, CUDA_C_32F, CUDA_C_32F, &ws_size_c);
    cusparseCsrmvEx_bufferSize(handle, alg, CUSPARSE_OPERATION_NON_TRANSPOSE, 4, 5, 9, alpha_z.d_data, CUDA_C_64F, descrA, a_z_val.d_data, CUDA_C_64F, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, b_z.d_data, CUDA_C_64F, beta_z.d_data, CUDA_C_64F, c_z.d_data, CUDA_C_64F, CUDA_C_64F, &ws_size_z);
  }

  void *ws_s;
  void *ws_d;
  void *ws_c;
  void *ws_z;
  cudaMalloc(&ws_s, ws_size_s);
  cudaMalloc(&ws_d, ws_size_d);
  cudaMalloc(&ws_c, ws_size_c);
  cudaMalloc(&ws_z, ws_size_z);

  cusparseCsrmvEx(handle, alg, CUSPARSE_OPERATION_NON_TRANSPOSE, 4, 5, 9, alpha_s.d_data, CUDA_R_32F, descrA, a_s_val.d_data, CUDA_R_32F, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, b_s.d_data, CUDA_R_32F, beta_s.d_data, CUDA_R_32F, c_s.d_data, CUDA_R_32F, CUDA_R_32F, ws_s);
  cusparseCsrmvEx(handle, alg, CUSPARSE_OPERATION_NON_TRANSPOSE, 4, 5, 9, alpha_d.d_data, CUDA_R_64F, descrA, a_d_val.d_data, CUDA_R_64F, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, b_d.d_data, CUDA_R_64F, beta_d.d_data, CUDA_R_64F, c_d.d_data, CUDA_R_64F, CUDA_R_64F, ws_d);
  if (run_complex_datatype) {
    cusparseCsrmvEx(handle, alg, CUSPARSE_OPERATION_NON_TRANSPOSE, 4, 5, 9, alpha_c.d_data, CUDA_C_32F, descrA, a_c_val.d_data, CUDA_C_32F, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, b_c.d_data, CUDA_C_32F, beta_c.d_data, CUDA_C_32F, c_c.d_data, CUDA_C_32F, CUDA_C_32F, ws_c);
    cusparseCsrmvEx(handle, alg, CUSPARSE_OPERATION_NON_TRANSPOSE, 4, 5, 9, alpha_z.d_data, CUDA_C_64F, descrA, a_z_val.d_data, CUDA_C_64F, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, b_z.d_data, CUDA_C_64F, beta_z.d_data, CUDA_C_64F, c_z.d_data, CUDA_C_64F, CUDA_C_64F, ws_z);
  }

  c_s.D2H();
  c_d.D2H();
  c_c.D2H();
  c_z.D2H();

  cudaFree(ws_s);
  cudaFree(ws_d);
  cudaFree(ws_c);
  cudaFree(ws_z);
  cudaStreamSynchronize(0);
  cusparseDestroyMatDescr(descrA);
  cusparseDestroy(handle);

  float expect_c[4] = {90, 130, 730, 570};
  if (compare_result(expect_c, c_s.h_data, 4) &&
      compare_result(expect_c, c_d.h_data, 4) &&
      compare_result(expect_c, c_c.h_data, 4) &&
      compare_result(expect_c, c_z.h_data, 4))
    printf("CsrmvEx pass\n");
  else {
    printf("CsrmvEx fail\n");
    test_passed = false;
  }
}

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
  Data<int> a_row_ptr(a_row_ptr_vec.data(), 4);
  std::vector<float> a_col_ind_vec = {0, 1, 2, 1, 2, 3};
  Data<int> a_col_ind(a_col_ind_vec.data(), 6);

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
  csrsv2Info_t info;
  cusparseCreateCsrsv2Info(&info);
  cusparseSolvePolicy_t policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
  policy = CUSPARSE_SOLVE_POLICY_NO_LEVEL;


  cusparseMatDescr_t descrA;
  cusparseCreateMatDescr(&descrA);
  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_TRIANGULAR);

  a_s_val.H2D();
  a_d_val.H2D();
  a_c_val.H2D();
  a_z_val.H2D();
  a_row_ptr.H2D();
  a_col_ind.H2D();
  f_s.H2D();
  f_d.H2D();
  f_c.H2D();
  f_z.H2D();

  int buffer_size_s0;
  int buffer_size_d0;
  int buffer_size_c0;
  int buffer_size_z0;
  cusparseScsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA, (float *)a_s_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, info, &buffer_size_s0);
  cusparseDcsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA, (double *)a_d_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, info, &buffer_size_d0);
  cusparseCcsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA, (float2 *)a_c_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, info, &buffer_size_c0);
  cusparseZcsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA, (double2 *)a_z_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, info, &buffer_size_z0);

  size_t buffer_size_s;
  size_t buffer_size_d;
  size_t buffer_size_c;
  size_t buffer_size_z;
  cusparseScsrsv2_bufferSizeExt(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA, (float *)a_s_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, info, &buffer_size_s);
  cusparseDcsrsv2_bufferSizeExt(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA, (double *)a_d_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, info, &buffer_size_d);
  cusparseCcsrsv2_bufferSizeExt(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA, (float2 *)a_c_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, info, &buffer_size_c);
  cusparseZcsrsv2_bufferSizeExt(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA, (double2 *)a_z_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, info, &buffer_size_z);

  void* buffer_s;
  void* buffer_d;
  void* buffer_c;
  void* buffer_z;
  cudaMalloc(&buffer_s, buffer_size_s);
  cudaMalloc(&buffer_d, buffer_size_d);
  cudaMalloc(&buffer_c, buffer_size_c);
  cudaMalloc(&buffer_z, buffer_size_z);

  cusparseScsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA, (float *)a_s_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, info, policy, buffer_s);
  cusparseDcsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA, (double *)a_d_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, info, policy, buffer_d);
  cusparseCcsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA, (float2 *)a_c_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, info, policy, buffer_c);
  cusparseZcsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA, (double2 *)a_z_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, info, policy, buffer_z);

  float alpha_s = 1;
  double alpha_d = 1;
  float2 alpha_c = float2{1, 0};
  double2 alpha_z = double2{1, 0};

  cusparseScsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, &alpha_s, descrA, (float *)a_s_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, info, f_s.d_data, x_s.d_data, policy, buffer_s);
  cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, &alpha_d, descrA, (double *)a_d_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, info, f_d.d_data, x_d.d_data, policy, buffer_d);
  cusparseCcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, &alpha_c, descrA, (float2 *)a_c_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, info, f_c.d_data, x_c.d_data, policy, buffer_c);
  cusparseZcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, &alpha_z, descrA, (double2 *)a_z_val.d_data, (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data, info, f_z.d_data, x_z.d_data, policy, buffer_z);

  x_s.D2H();
  x_d.D2H();
  x_c.D2H();
  x_z.D2H();

  cudaStreamSynchronize(0);
  cusparseDestroyCsrsv2Info(info);
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
  test_passed = true;
}

int main() {
  test_cusparseSetGetStream();
  test_cusparseTcsrmv_ge();
  test_cusparseTcsrmv_sy();
  test_cusparseTcsrmv_tr();
  // test_cusparseTcsrmm(); // Re-enable this test until MKL issue fixed
  test_cusparseTcsrsv();
  test_cusparseTcsrmv_mp();
  test_cusparseCsrmvEx();
  test_cusparseTcsrsv2();

  if (test_passed)
    return 0;
  return -1;
}
