// ===------- sparse_utils_2_buffer.cpp --------------------- *- C++ -* ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#define DPCT_USM_LEVEL_NONE
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/sparse_utils.hpp>
#include <dpct/blas_utils.hpp>

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
    d_data = (d_data_t *)dpct::dpct_malloc(sizeof(d_data_t) * element_num);
    dpct::dpct_memset(d_data, 0, sizeof(d_data_t) * element_num);
  }
  Data(float *input_data, int element_num) : element_num(element_num) {
    h_data = (float *)malloc(sizeof(float) * element_num);
    d_data = (d_data_t *)dpct::dpct_malloc(sizeof(d_data_t) * element_num);
    dpct::dpct_memset(d_data, 0, sizeof(d_data_t) * element_num);
    memcpy(h_data, input_data, sizeof(float) * element_num);
  }
  ~Data() {
    free(h_data);
    dpct::dpct_free(d_data);
  }
  void H2D() {
    d_data_t *h_temp = (d_data_t *)malloc(sizeof(d_data_t) * element_num);
    memset(h_temp, 0, sizeof(d_data_t) * element_num);
    from_float_convert(h_data, h_temp);
    dpct::dpct_memcpy(d_data, h_temp, sizeof(d_data_t) * element_num,
                      dpct::host_to_device);
    free(h_temp);
  }
  void D2H() {
    d_data_t *h_temp = (d_data_t *)malloc(sizeof(d_data_t) * element_num);
    memset(h_temp, 0, sizeof(d_data_t) * element_num);
    dpct::dpct_memcpy(h_temp, d_data, sizeof(d_data_t) * element_num,
                      dpct::device_to_host);
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
inline void Data<sycl::float2>::from_float_convert(float *in,
                                                   sycl::float2 *out) {
  for (int i = 0; i < element_num; i++)
    out[i].x() = in[i];
}
template <>
inline void Data<sycl::double2>::from_float_convert(float *in,
                                                    sycl::double2 *out) {
  for (int i = 0; i < element_num; i++)
    out[i].x() = in[i];
}

template <>
inline void Data<sycl::float2>::to_float_convert(sycl::float2 *in, float *out) {
  for (int i = 0; i < element_num; i++)
    out[i] = in[i].x();
}
template <>
inline void Data<sycl::double2>::to_float_convert(sycl::double2 *in,
                                                  float *out) {
  for (int i = 0; i < element_num; i++)
    out[i] = in[i].x();
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

const bool run_complex_datatype = false;

void test_cusparseSetGetStream() {
  sycl::queue *handle;
  handle = &dpct::get_default_queue();
  dpct::queue_ptr stream;
  stream = handle;
  handle = stream;
  handle = nullptr;
  printf("SetGetStream pass\n");
  test_passed = true;
}

void test_cusparseTcsrmv_ge() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::vector<float> a_val_vec = {1, 4, 2, 3, 5, 7, 8, 9, 6};
  Data<float> a_s_val(a_val_vec.data(), 9);
  Data<double> a_d_val(a_val_vec.data(), 9);
  Data<sycl::float2> a_c_val(a_val_vec.data(), 9);
  Data<sycl::double2> a_z_val(a_val_vec.data(), 9);
  std::vector<float> a_row_ptr_vec = {0, 2, 4, 7, 9};
  Data<int> a_row_ptr(a_row_ptr_vec.data(), 5);
  std::vector<float> a_col_ind_vec = {0, 1, 1, 2, 0, 3, 4, 2, 4};
  Data<int> a_col_ind(a_col_ind_vec.data(), 9);

  std::vector<float> b_vec = {1, 2, 3, 4, 5};
  Data<float> b_s(b_vec.data(), 5);
  Data<double> b_d(b_vec.data(), 5);
  Data<sycl::float2> b_c(b_vec.data(), 5);
  Data<sycl::double2> b_z(b_vec.data(), 5);

  Data<float> c_s(4);
  Data<double> c_d(4);
  Data<sycl::float2> c_c(4);
  Data<sycl::double2> c_z(4);

  float alpha = 10;
  Data<float> alpha_s(&alpha, 1);
  Data<double> alpha_d(&alpha, 1);
  Data<sycl::float2> alpha_c(&alpha, 1);
  Data<sycl::double2> alpha_z(&alpha, 1);

  float beta = 0;
  Data<float> beta_s(&beta, 1);
  Data<double> beta_d(&beta, 1);
  Data<sycl::float2> beta_c(&beta, 1);
  Data<sycl::double2> beta_z(&beta, 1);

  sycl::queue *handle;
  handle = &q_ct1;

  /*
  DPCT1026:0: The call to cusparseSetPointerMode was removed because this call
  is redundant in SYCL.
  */

  std::shared_ptr<dpct::sparse::matrix_info> descrA;
  descrA = std::make_shared<dpct::sparse::matrix_info>();
  descrA->set_index_base(oneapi::mkl::index_base::zero);
  descrA->set_matrix_type(dpct::sparse::matrix_info::matrix_type::ge);

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

  /*
  DPCT1045:1: Migration is only supported for this API for the
  general/symmetric/triangular sparse matrix type. You may need to adjust the
  code.
  */
  dpct::sparse::csrmv(*handle, oneapi::mkl::transpose::nontrans, 4, 5,
                      (float *)alpha_s.d_data, descrA, (float *)a_s_val.d_data,
                      (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data,
                      (float *)b_s.d_data, (float *)beta_s.d_data,
                      (float *)c_s.d_data);
  /*
  DPCT1045:2: Migration is only supported for this API for the
  general/symmetric/triangular sparse matrix type. You may need to adjust the
  code.
  */
  dpct::sparse::csrmv(*handle, oneapi::mkl::transpose::nontrans, 4, 5,
                      (double *)alpha_d.d_data, descrA,
                      (double *)a_d_val.d_data, (int *)a_row_ptr.d_data,
                      (int *)a_col_ind.d_data, (double *)b_d.d_data,
                      (double *)beta_d.d_data, (double *)c_d.d_data);
  if (run_complex_datatype) {
    /*
    DPCT1045:4: Migration is only supported for this API for the
    general/symmetric/triangular sparse matrix type. You may need to adjust the
    code.
    */
    dpct::sparse::csrmv(*handle, oneapi::mkl::transpose::nontrans, 4, 5,
                        (sycl::float2 *)alpha_c.d_data, descrA,
                        (sycl::float2 *)a_c_val.d_data, (int *)a_row_ptr.d_data,
                        (int *)a_col_ind.d_data, (sycl::float2 *)b_c.d_data,
                        (sycl::float2 *)beta_c.d_data,
                        (sycl::float2 *)c_c.d_data);
    /*
    DPCT1045:5: Migration is only supported for this API for the
    general/symmetric/triangular sparse matrix type. You may need to adjust the
    code.
    */
    dpct::sparse::csrmv(
        *handle, oneapi::mkl::transpose::nontrans, 4, 5,
        (sycl::double2 *)alpha_z.d_data, descrA,
        (sycl::double2 *)a_z_val.d_data, (int *)a_row_ptr.d_data,
        (int *)a_col_ind.d_data, (sycl::double2 *)b_z.d_data,
        (sycl::double2 *)beta_z.d_data, (sycl::double2 *)c_z.d_data);
  }

  c_s.D2H();
  c_d.D2H();
  c_c.D2H();
  c_z.D2H();

  q_ct1.wait();
  /*
  DPCT1026:3: The call to cusparseDestroyMatDescr was removed because this call
  is redundant in SYCL.
  */
  handle = nullptr;

  float expect_c[4] = {90, 130, 730, 570};
  if (compare_result(expect_c, c_s.h_data, 4) &&
      compare_result(expect_c, c_d.h_data, 4)/* &&
      compare_result(expect_c, c_c.h_data, 4) &&
      compare_result(expect_c, c_z.h_data, 4)*/)
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
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::vector<float> a_val_vec = {1, 4, 1, 4, 2, 3, 3, 7, 1, 7};
  Data<float> a_s_val(a_val_vec.data(), 10);
  Data<double> a_d_val(a_val_vec.data(), 10);
  Data<sycl::float2> a_c_val(a_val_vec.data(), 10);
  Data<sycl::double2> a_z_val(a_val_vec.data(), 10);
  std::vector<float> a_row_ptr_vec = {0, 3, 6, 8, 10};
  Data<int> a_row_ptr(a_row_ptr_vec.data(), 5);
  std::vector<float> a_col_ind_vec = {0, 1, 3, 0, 1, 2, 1, 3, 0, 2};
  Data<int> a_col_ind(a_col_ind_vec.data(), 10);

  std::vector<float> b_vec = {1, 2, 3, 4};
  Data<float> b_s(b_vec.data(), 4);
  Data<double> b_d(b_vec.data(), 4);
  Data<sycl::float2> b_c(b_vec.data(), 4);
  Data<sycl::double2> b_z(b_vec.data(), 4);

  Data<float> c_s(4);
  Data<double> c_d(4);
  Data<sycl::float2> c_c(4);
  Data<sycl::double2> c_z(4);

  float alpha = 10;
  Data<float> alpha_s(&alpha, 1);
  Data<double> alpha_d(&alpha, 1);
  Data<sycl::float2> alpha_c(&alpha, 1);
  Data<sycl::double2> alpha_z(&alpha, 1);

  float beta = 0;
  Data<float> beta_s(&beta, 1);
  Data<double> beta_d(&beta, 1);
  Data<sycl::float2> beta_c(&beta, 1);
  Data<sycl::double2> beta_z(&beta, 1);

  sycl::queue *handle;
  handle = &q_ct1;

  /*
  DPCT1026:6: The call to cusparseSetPointerMode was removed because this call
  is redundant in SYCL.
  */

  std::shared_ptr<dpct::sparse::matrix_info> descrA;
  descrA = std::make_shared<dpct::sparse::matrix_info>();
  descrA->set_index_base(oneapi::mkl::index_base::zero);
  descrA->set_matrix_type(dpct::sparse::matrix_info::matrix_type::sy);

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

  /*
  DPCT1045:7: Migration is only supported for this API for the
  general/symmetric/triangular sparse matrix type. You may need to adjust the
  code.
  */
  dpct::sparse::csrmv(*handle, oneapi::mkl::transpose::nontrans, 4, 4,
                      (float *)alpha_s.d_data, descrA, (float *)a_s_val.d_data,
                      (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data,
                      (float *)b_s.d_data, (float *)beta_s.d_data,
                      (float *)c_s.d_data);
  /*
  DPCT1045:8: Migration is only supported for this API for the
  general/symmetric/triangular sparse matrix type. You may need to adjust the
  code.
  */
  dpct::sparse::csrmv(*handle, oneapi::mkl::transpose::nontrans, 4, 4,
                      (double *)alpha_d.d_data, descrA,
                      (double *)a_d_val.d_data, (int *)a_row_ptr.d_data,
                      (int *)a_col_ind.d_data, (double *)b_d.d_data,
                      (double *)beta_d.d_data, (double *)c_d.d_data);
  if (run_complex_datatype) {
    /*
    DPCT1045:10: Migration is only supported for this API for the
    general/symmetric/triangular sparse matrix type. You may need to adjust the
    code.
    */
    dpct::sparse::csrmv(*handle, oneapi::mkl::transpose::nontrans, 4, 4,
                        (sycl::float2 *)alpha_c.d_data, descrA,
                        (sycl::float2 *)a_c_val.d_data, (int *)a_row_ptr.d_data,
                        (int *)a_col_ind.d_data, (sycl::float2 *)b_c.d_data,
                        (sycl::float2 *)beta_c.d_data,
                        (sycl::float2 *)c_c.d_data);
    /*
    DPCT1045:11: Migration is only supported for this API for the
    general/symmetric/triangular sparse matrix type. You may need to adjust the
    code.
    */
    dpct::sparse::csrmv(
        *handle, oneapi::mkl::transpose::nontrans, 4, 4,
        (sycl::double2 *)alpha_z.d_data, descrA,
        (sycl::double2 *)a_z_val.d_data, (int *)a_row_ptr.d_data,
        (int *)a_col_ind.d_data, (sycl::double2 *)b_z.d_data,
        (sycl::double2 *)beta_z.d_data, (sycl::double2 *)c_z.d_data);
  }

  c_s.D2H();
  c_d.D2H();
  c_c.D2H();
  c_z.D2H();

  q_ct1.wait();
  /*
  DPCT1026:9: The call to cusparseDestroyMatDescr was removed because this call
  is redundant in SYCL.
  */
  handle = nullptr;

  float expect_c[4] = {130, 170, 340, 220};
  if (compare_result(expect_c, c_s.h_data, 4) &&
      compare_result(expect_c, c_d.h_data, 4)/* &&
      compare_result(expect_c, c_c.h_data, 4) &&
      compare_result(expect_c, c_z.h_data, 4)*/)
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
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::vector<float> a_val_vec = {1, 4, 1, 2, 3, 7, 1};
  Data<float> a_s_val(a_val_vec.data(), 7);
  Data<double> a_d_val(a_val_vec.data(), 7);
  Data<sycl::float2> a_c_val(a_val_vec.data(), 7);
  Data<sycl::double2> a_z_val(a_val_vec.data(), 7);
  std::vector<float> a_row_ptr_vec = {0, 3, 5, 6, 7};
  Data<int> a_row_ptr(a_row_ptr_vec.data(), 5);
  std::vector<float> a_col_ind_vec = {0, 1, 3, 1, 2, 3, 3};
  Data<int> a_col_ind(a_col_ind_vec.data(), 7);

  std::vector<float> b_vec = {1, 2, 3, 4};
  Data<float> b_s(b_vec.data(), 4);
  Data<double> b_d(b_vec.data(), 4);
  Data<sycl::float2> b_c(b_vec.data(), 4);
  Data<sycl::double2> b_z(b_vec.data(), 4);

  Data<float> c_s(4);
  Data<double> c_d(4);
  Data<sycl::float2> c_c(4);
  Data<sycl::double2> c_z(4);

  float alpha = 10;
  Data<float> alpha_s(&alpha, 1);
  Data<double> alpha_d(&alpha, 1);
  Data<sycl::float2> alpha_c(&alpha, 1);
  Data<sycl::double2> alpha_z(&alpha, 1);

  float beta = 0;
  Data<float> beta_s(&beta, 1);
  Data<double> beta_d(&beta, 1);
  Data<sycl::float2> beta_c(&beta, 1);
  Data<sycl::double2> beta_z(&beta, 1);

  sycl::queue *handle;
  handle = &q_ct1;

  /*
  DPCT1026:12: The call to cusparseSetPointerMode was removed because this call
  is redundant in SYCL.
  */

  std::shared_ptr<dpct::sparse::matrix_info> descrA;
  descrA = std::make_shared<dpct::sparse::matrix_info>();
  descrA->set_index_base(oneapi::mkl::index_base::zero);
  descrA->set_matrix_type(dpct::sparse::matrix_info::matrix_type::tr);
  descrA->set_uplo(oneapi::mkl::uplo::upper);

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

  /*
  DPCT1045:13: Migration is only supported for this API for the
  general/symmetric/triangular sparse matrix type. You may need to adjust the
  code.
  */
  dpct::sparse::csrmv(*handle, oneapi::mkl::transpose::nontrans, 4, 4,
                      (float *)alpha_s.d_data, descrA, (float *)a_s_val.d_data,
                      (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data,
                      (float *)b_s.d_data, (float *)beta_s.d_data,
                      (float *)c_s.d_data);
  /*
  DPCT1045:14: Migration is only supported for this API for the
  general/symmetric/triangular sparse matrix type. You may need to adjust the
  code.
  */
  dpct::sparse::csrmv(*handle, oneapi::mkl::transpose::nontrans, 4, 4,
                      (double *)alpha_d.d_data, descrA,
                      (double *)a_d_val.d_data, (int *)a_row_ptr.d_data,
                      (int *)a_col_ind.d_data, (double *)b_d.d_data,
                      (double *)beta_d.d_data, (double *)c_d.d_data);
  if (run_complex_datatype) {
    /*
    DPCT1045:16: Migration is only supported for this API for the
    general/symmetric/triangular sparse matrix type. You may need to adjust the
    code.
    */
    dpct::sparse::csrmv(*handle, oneapi::mkl::transpose::nontrans, 4, 4,
                        (sycl::float2 *)alpha_c.d_data, descrA,
                        (sycl::float2 *)a_c_val.d_data, (int *)a_row_ptr.d_data,
                        (int *)a_col_ind.d_data, (sycl::float2 *)b_c.d_data,
                        (sycl::float2 *)beta_c.d_data,
                        (sycl::float2 *)c_c.d_data);
    /*
    DPCT1045:17: Migration is only supported for this API for the
    general/symmetric/triangular sparse matrix type. You may need to adjust the
    code.
    */
    dpct::sparse::csrmv(
        *handle, oneapi::mkl::transpose::nontrans, 4, 4,
        (sycl::double2 *)alpha_z.d_data, descrA,
        (sycl::double2 *)a_z_val.d_data, (int *)a_row_ptr.d_data,
        (int *)a_col_ind.d_data, (sycl::double2 *)b_z.d_data,
        (sycl::double2 *)beta_z.d_data, (sycl::double2 *)c_z.d_data);
  }

  c_s.D2H();
  c_d.D2H();
  c_c.D2H();
  c_z.D2H();

  q_ct1.wait();
  /*
  DPCT1026:15: The call to cusparseDestroyMatDescr was removed because this call
  is redundant in SYCL.
  */
  handle = nullptr;

  float expect_c[4] = {130, 130, 280, 40};
  if (compare_result(expect_c, c_s.h_data, 4) &&
      compare_result(expect_c, c_d.h_data, 4)/* &&
      compare_result(expect_c, c_c.h_data, 4) &&
      compare_result(expect_c, c_z.h_data, 4)*/)
    printf("Tcsrmv_tr pass\n");
  else {
    printf("Tcsrmv_tr fail\n");
    test_passed = false;
  }
}

void test_cusparseTcsrmm() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::vector<float> a_val_vec = {1, 4, 2, 3, 5, 7, 8, 9, 6};
  Data<float> a_s_val(a_val_vec.data(), 9);
  Data<double> a_d_val(a_val_vec.data(), 9);
  Data<sycl::float2> a_c_val(a_val_vec.data(), 9);
  Data<sycl::double2> a_z_val(a_val_vec.data(), 9);
  std::vector<float> a_row_ptr_vec = {0, 2, 4, 7, 9};
  Data<int> a_row_ptr(a_row_ptr_vec.data(), 5);
  std::vector<float> a_col_ind_vec = {0, 1, 1, 2, 0, 3, 4, 2, 4};
  Data<int> a_col_ind(a_col_ind_vec.data(), 9);

  std::vector<float> b_vec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  Data<float> b_s(b_vec.data(), 10);
  Data<double> b_d(b_vec.data(), 10);
  Data<sycl::float2> b_c(b_vec.data(), 10);
  Data<sycl::double2> b_z(b_vec.data(), 10);

  Data<float> c_s(8);
  Data<double> c_d(8);
  Data<sycl::float2> c_c(8);
  Data<sycl::double2> c_z(8);

  float alpha = 10;
  Data<float> alpha_s(&alpha, 1);
  Data<double> alpha_d(&alpha, 1);
  Data<sycl::float2> alpha_c(&alpha, 1);
  Data<sycl::double2> alpha_z(&alpha, 1);

  float beta = 0;
  Data<float> beta_s(&beta, 1);
  Data<double> beta_d(&beta, 1);
  Data<sycl::float2> beta_c(&beta, 1);
  Data<sycl::double2> beta_z(&beta, 1);

  sycl::queue *handle;
  handle = &q_ct1;

  /*
  DPCT1026:18: The call to cusparseSetPointerMode was removed because this call
  is redundant in SYCL.
  */

  std::shared_ptr<dpct::sparse::matrix_info> descrA;
  descrA = std::make_shared<dpct::sparse::matrix_info>();
  descrA->set_index_base(oneapi::mkl::index_base::zero);
  descrA->set_matrix_type(dpct::sparse::matrix_info::matrix_type::ge);

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

  /*
  DPCT1045:19: Migration is only supported for this API for the general sparse
  matrix type. You may need to adjust the code.
  */
  dpct::sparse::csrmm(*handle, oneapi::mkl::transpose::nontrans, 4, 2, 5,
                      (float *)alpha_s.d_data, descrA, (float *)a_s_val.d_data,
                      (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data,
                      (float *)b_s.d_data, 5, (float *)beta_s.d_data,
                      (float *)c_s.d_data, 4);
  /*
  DPCT1045:20: Migration is only supported for this API for the general sparse
  matrix type. You may need to adjust the code.
  */
  dpct::sparse::csrmm(*handle, oneapi::mkl::transpose::nontrans, 4, 2, 5,
                      (double *)alpha_d.d_data, descrA,
                      (double *)a_d_val.d_data, (int *)a_row_ptr.d_data,
                      (int *)a_col_ind.d_data, (double *)b_d.d_data, 5,
                      (double *)beta_d.d_data, (double *)c_d.d_data, 4);
  if (run_complex_datatype) {
    /*
    DPCT1045:22: Migration is only supported for this API for the general sparse
    matrix type. You may need to adjust the code.
    */
    dpct::sparse::csrmm(*handle, oneapi::mkl::transpose::nontrans, 4, 2, 5,
                        (sycl::float2 *)alpha_c.d_data, descrA,
                        (sycl::float2 *)a_c_val.d_data, (int *)a_row_ptr.d_data,
                        (int *)a_col_ind.d_data, (sycl::float2 *)b_c.d_data, 5,
                        (sycl::float2 *)beta_c.d_data,
                        (sycl::float2 *)c_c.d_data, 4);
    /*
    DPCT1045:23: Migration is only supported for this API for the general sparse
    matrix type. You may need to adjust the code.
    */
    dpct::sparse::csrmm(
        *handle, oneapi::mkl::transpose::nontrans, 4, 2, 5,
        (sycl::double2 *)alpha_z.d_data, descrA,
        (sycl::double2 *)a_z_val.d_data, (int *)a_row_ptr.d_data,
        (int *)a_col_ind.d_data, (sycl::double2 *)b_z.d_data, 5,
        (sycl::double2 *)beta_z.d_data, (sycl::double2 *)c_z.d_data, 4);
  }

  c_s.D2H();
  c_d.D2H();
  c_c.D2H();
  c_z.D2H();

  q_ct1.wait();
  /*
  DPCT1026:21: The call to cusparseDestroyMatDescr was removed because this call
  is redundant in SYCL.
  */
  handle = nullptr;

  float expect_c[8] = {90, 130, 730, 570, 340, 380, 1730, 1320};
  if (compare_result(expect_c, c_s.h_data, 8) &&
      compare_result(expect_c, c_d.h_data, 8)/* &&
      compare_result(expect_c, c_c.h_data, 8) &&
      compare_result(expect_c, c_z.h_data, 8)*/)
    printf("Tcsrmm pass\n");
  else {
    printf("Tcsrmm fail\n");
    test_passed = false;
  }
}

void test_cusparseTcsrsv() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::vector<float> a_val_vec = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  Data<float> a_s_val(a_val_vec.data(), 9);
  Data<double> a_d_val(a_val_vec.data(), 9);
  Data<sycl::float2> a_c_val(a_val_vec.data(), 9);
  Data<sycl::double2> a_z_val(a_val_vec.data(), 9);
  std::vector<float> a_row_ptr_vec = {0, 3, 4, 7, 9};
  Data<int> a_row_ptr(a_row_ptr_vec.data(), 5);
  std::vector<float> a_col_ind_vec = {0, 2, 3, 1, 0, 2, 3, 1, 3};
  Data<int> a_col_ind(a_col_ind_vec.data(), 9);

  sycl::queue *handle;
  handle = &q_ct1;
  std::shared_ptr<dpct::sparse::optimize_info> info;
  info = std::make_shared<dpct::sparse::optimize_info>();

  std::shared_ptr<dpct::sparse::matrix_info> descrA;
  descrA = std::make_shared<dpct::sparse::matrix_info>();
  descrA->set_index_base(oneapi::mkl::index_base::zero);
  descrA->set_matrix_type(dpct::sparse::matrix_info::matrix_type::ge);

  a_s_val.H2D();
  a_d_val.H2D();
  a_c_val.H2D();
  a_z_val.H2D();
  a_row_ptr.H2D();
  a_col_ind.H2D();

  dpct::sparse::optimize_csrsv(*handle, oneapi::mkl::transpose::nontrans, 4,
                               descrA, (float *)a_s_val.d_data,
                               (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data,
                               info);
  dpct::sparse::optimize_csrsv(*handle, oneapi::mkl::transpose::nontrans, 4,
                               descrA, (double *)a_d_val.d_data,
                               (int *)a_row_ptr.d_data, (int *)a_col_ind.d_data,
                               info);
  if (run_complex_datatype) {
    dpct::sparse::optimize_csrsv(*handle, oneapi::mkl::transpose::nontrans, 4,
                                 descrA, (sycl::float2 *)a_c_val.d_data,
                                 (int *)a_row_ptr.d_data,
                                 (int *)a_col_ind.d_data, info);
    dpct::sparse::optimize_csrsv(*handle, oneapi::mkl::transpose::nontrans, 4,
                                 descrA, (sycl::double2 *)a_z_val.d_data,
                                 (int *)a_row_ptr.d_data,
                                 (int *)a_col_ind.d_data, info);
  }

  q_ct1.wait();
  info.reset();
  /*
  DPCT1026:24: The call to cusparseDestroyMatDescr was removed because this call
  is redundant in SYCL.
  */
  handle = nullptr;

  printf("Tcsrsv pass\n");
  test_passed = true;
}

void test_cusparseSpMV() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::vector<float> a_val_vec = {1, 4, 2, 3, 5, 7, 8, 9, 6};
  Data<float> a_s_val(a_val_vec.data(), 9);
  Data<double> a_d_val(a_val_vec.data(), 9);
  Data<sycl::float2> a_c_val(a_val_vec.data(), 9);
  Data<sycl::double2> a_z_val(a_val_vec.data(), 9);
  std::vector<float> a_row_ptr_vec = {0, 2, 4, 7, 9};
  Data<int> a_row_ptr(a_row_ptr_vec.data(), 5);
  std::vector<float> a_col_ind_vec = {0, 1, 1, 2, 0, 3, 4, 2, 4};
  Data<int> a_col_ind(a_col_ind_vec.data(), 9);

  std::vector<float> b_vec = {1, 2, 3, 4, 5};
  Data<float> b_s(b_vec.data(), 5);
  Data<double> b_d(b_vec.data(), 5);
  Data<sycl::float2> b_c(b_vec.data(), 5);
  Data<sycl::double2> b_z(b_vec.data(), 5);

  Data<float> c_s(4);
  Data<double> c_d(4);
  Data<sycl::float2> c_c(4);
  Data<sycl::double2> c_z(4);

  float alpha = 10;
  Data<float> alpha_s(&alpha, 1);
  Data<double> alpha_d(&alpha, 1);
  Data<sycl::float2> alpha_c(&alpha, 1);
  Data<sycl::double2> alpha_z(&alpha, 1);

  float beta = 0;
  Data<float> beta_s(&beta, 1);
  Data<double> beta_d(&beta, 1);
  Data<sycl::float2> beta_c(&beta, 1);
  Data<sycl::double2> beta_z(&beta, 1);

  sycl::queue *handle;
  handle = &q_ct1;

  /*
  DPCT1026:0: The call to cusparseSetPointerMode was removed because this call
  is redundant in SYCL.
  */

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

  dpct::sparse::sparse_matrix_desc_t a_descr_s;
  dpct::sparse::sparse_matrix_desc_t a_descr_d;
  dpct::sparse::sparse_matrix_desc_t a_descr_c;
  dpct::sparse::sparse_matrix_desc_t a_descr_z;
  a_descr_s = std::make_shared<dpct::sparse::sparse_matrix_desc>(
      4, 5, 9, a_row_ptr.d_data, a_col_ind.d_data, a_s_val.d_data,
      dpct::library_data_t::real_int32, dpct::library_data_t::real_int32,
      oneapi::mkl::index_base::zero, dpct::library_data_t::real_float,
      dpct::sparse::matrix_format::csr);
  a_descr_d = std::make_shared<dpct::sparse::sparse_matrix_desc>(
      4, 5, 9, a_row_ptr.d_data, a_col_ind.d_data, a_d_val.d_data,
      dpct::library_data_t::real_int32, dpct::library_data_t::real_int32,
      oneapi::mkl::index_base::zero, dpct::library_data_t::real_double,
      dpct::sparse::matrix_format::csr);
  a_descr_c = std::make_shared<dpct::sparse::sparse_matrix_desc>(
      4, 5, 9, a_row_ptr.d_data, a_col_ind.d_data, a_c_val.d_data,
      dpct::library_data_t::real_int32, dpct::library_data_t::real_int32,
      oneapi::mkl::index_base::zero, dpct::library_data_t::complex_float,
      dpct::sparse::matrix_format::csr);
  a_descr_z = std::make_shared<dpct::sparse::sparse_matrix_desc>(
      4, 5, 9, a_row_ptr.d_data, a_col_ind.d_data, a_z_val.d_data,
      dpct::library_data_t::real_int32, dpct::library_data_t::real_int32,
      oneapi::mkl::index_base::zero, dpct::library_data_t::complex_double,
      dpct::sparse::matrix_format::csr);

  std::shared_ptr<dpct::sparse::dense_vector_desc> b_descr_s;
  std::shared_ptr<dpct::sparse::dense_vector_desc> b_descr_d;
  std::shared_ptr<dpct::sparse::dense_vector_desc> b_descr_c;
  std::shared_ptr<dpct::sparse::dense_vector_desc> b_descr_z;
  b_descr_s = std::make_shared<dpct::sparse::dense_vector_desc>(
      5, b_s.d_data, dpct::library_data_t::real_float);
  b_descr_d = std::make_shared<dpct::sparse::dense_vector_desc>(
      5, b_d.d_data, dpct::library_data_t::real_double);
  b_descr_c = std::make_shared<dpct::sparse::dense_vector_desc>(
      5, b_c.d_data, dpct::library_data_t::complex_float);
  b_descr_z = std::make_shared<dpct::sparse::dense_vector_desc>(
      5, b_z.d_data, dpct::library_data_t::complex_double);

  std::shared_ptr<dpct::sparse::dense_vector_desc> c_descr_s;
  std::shared_ptr<dpct::sparse::dense_vector_desc> c_descr_d;
  std::shared_ptr<dpct::sparse::dense_vector_desc> c_descr_c;
  std::shared_ptr<dpct::sparse::dense_vector_desc> c_descr_z;
  c_descr_s = std::make_shared<dpct::sparse::dense_vector_desc>(
      4, c_s.d_data, dpct::library_data_t::real_float);
  c_descr_d = std::make_shared<dpct::sparse::dense_vector_desc>(
      4, c_d.d_data, dpct::library_data_t::real_double);
  c_descr_c = std::make_shared<dpct::sparse::dense_vector_desc>(
      4, c_c.d_data, dpct::library_data_t::complex_float);
  c_descr_z = std::make_shared<dpct::sparse::dense_vector_desc>(
      4, c_z.d_data, dpct::library_data_t::complex_double);

  size_t ws_size_s;
  size_t ws_size_d;
  size_t ws_size_c;
  size_t ws_size_z;
  ws_size_s = 0;
  ws_size_d = 0;
  ws_size_c = 0;
  ws_size_z = 0;

  void *ws_s;
  void *ws_d;
  void *ws_c;
  void *ws_z;
  ws_s = dpct::dpct_malloc(ws_size_s);
  ws_d = dpct::dpct_malloc(ws_size_d);
  ws_c = dpct::dpct_malloc(ws_size_c);
  ws_z = dpct::dpct_malloc(ws_size_z);

  dpct::sparse::spmv(*handle, oneapi::mkl::transpose::nontrans, alpha_s.d_data,
                     a_descr_s, b_descr_s, beta_s.d_data, c_descr_s,
                     dpct::library_data_t::real_float);
  dpct::sparse::spmv(*handle, oneapi::mkl::transpose::nontrans, alpha_d.d_data,
                     a_descr_d, b_descr_d, beta_d.d_data, c_descr_d,
                     dpct::library_data_t::real_double);
  if (run_complex_datatype) {
    dpct::sparse::spmv(*handle, oneapi::mkl::transpose::nontrans,
                       alpha_c.d_data, a_descr_c, b_descr_c, beta_c.d_data,
                       c_descr_c, dpct::library_data_t::complex_float);
    dpct::sparse::spmv(*handle, oneapi::mkl::transpose::nontrans,
                       alpha_z.d_data, a_descr_z, b_descr_z, beta_z.d_data,
                       c_descr_z, dpct::library_data_t::complex_double);
  }

  c_s.D2H();
  c_d.D2H();
  c_c.D2H();
  c_z.D2H();

  q_ct1.wait();

  dpct::dpct_free(ws_s);
  dpct::dpct_free(ws_d);
  dpct::dpct_free(ws_c);
  dpct::dpct_free(ws_z);
  a_descr_s.reset();
  a_descr_d.reset();
  a_descr_c.reset();
  a_descr_z.reset();
  b_descr_s.reset();
  b_descr_d.reset();
  b_descr_c.reset();
  b_descr_z.reset();
  c_descr_s.reset();
  c_descr_d.reset();
  c_descr_c.reset();
  c_descr_z.reset();
  handle = nullptr;

  float expect_c[4] = {90, 130, 730, 570};
  if (compare_result(expect_c, c_s.h_data, 4) &&
      compare_result(expect_c, c_d.h_data, 4)/*&&
      compare_result(expect_c, c_c.h_data, 4) &&
      compare_result(expect_c, c_z.h_data, 4)*/)
    printf("SpMV pass\n");
  else {
    printf("SpMV fail\n");
    test_passed = false;
  }
}

void test_cusparseSpMM() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::vector<float> a_val_vec = {1, 4, 2, 3, 5, 7, 8, 9, 6};
  Data<float> a_s_val(a_val_vec.data(), 9);
  Data<double> a_d_val(a_val_vec.data(), 9);
  Data<sycl::float2> a_c_val(a_val_vec.data(), 9);
  Data<sycl::double2> a_z_val(a_val_vec.data(), 9);
  std::vector<float> a_row_ptr_vec = {0, 2, 4, 7, 9};
  Data<int> a_row_ptr(a_row_ptr_vec.data(), 5);
  std::vector<float> a_col_ind_vec = {0, 1, 1, 2, 0, 3, 4, 2, 4};
  Data<int> a_col_ind(a_col_ind_vec.data(), 9);

  std::vector<float> b_vec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  Data<float> b_s(b_vec.data(), 10);
  Data<double> b_d(b_vec.data(), 10);
  Data<sycl::float2> b_c(b_vec.data(), 10);
  Data<sycl::double2> b_z(b_vec.data(), 10);

  Data<float> c_s(8);
  Data<double> c_d(8);
  Data<sycl::float2> c_c(8);
  Data<sycl::double2> c_z(8);

  float alpha = 10;
  Data<float> alpha_s(&alpha, 1);
  Data<double> alpha_d(&alpha, 1);
  Data<sycl::float2> alpha_c(&alpha, 1);
  Data<sycl::double2> alpha_z(&alpha, 1);

  float beta = 0;
  Data<float> beta_s(&beta, 1);
  Data<double> beta_d(&beta, 1);
  Data<sycl::float2> beta_c(&beta, 1);
  Data<sycl::double2> beta_z(&beta, 1);

  sycl::queue *handle;
  handle = &q_ct1;

  /*
  DPCT1026:1: The call to cusparseSetPointerMode was removed because this call
  is redundant in SYCL.
  */

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

  dpct::sparse::sparse_matrix_desc_t a_descr_s;
  dpct::sparse::sparse_matrix_desc_t a_descr_d;
  dpct::sparse::sparse_matrix_desc_t a_descr_c;
  dpct::sparse::sparse_matrix_desc_t a_descr_z;
  a_descr_s = std::make_shared<dpct::sparse::sparse_matrix_desc>(
      4, 5, 9, a_row_ptr.d_data, a_col_ind.d_data, a_s_val.d_data,
      dpct::library_data_t::real_int32, dpct::library_data_t::real_int32,
      oneapi::mkl::index_base::zero, dpct::library_data_t::real_float,
      dpct::sparse::matrix_format::csr);
  a_descr_d = std::make_shared<dpct::sparse::sparse_matrix_desc>(
      4, 5, 9, a_row_ptr.d_data, a_col_ind.d_data, a_d_val.d_data,
      dpct::library_data_t::real_int32, dpct::library_data_t::real_int32,
      oneapi::mkl::index_base::zero, dpct::library_data_t::real_double,
      dpct::sparse::matrix_format::csr);
  a_descr_c = std::make_shared<dpct::sparse::sparse_matrix_desc>(
      4, 5, 9, a_row_ptr.d_data, a_col_ind.d_data, a_c_val.d_data,
      dpct::library_data_t::real_int32, dpct::library_data_t::real_int32,
      oneapi::mkl::index_base::zero, dpct::library_data_t::complex_float,
      dpct::sparse::matrix_format::csr);
  a_descr_z = std::make_shared<dpct::sparse::sparse_matrix_desc>(
      4, 5, 9, a_row_ptr.d_data, a_col_ind.d_data, a_z_val.d_data,
      dpct::library_data_t::real_int32, dpct::library_data_t::real_int32,
      oneapi::mkl::index_base::zero, dpct::library_data_t::complex_double,
      dpct::sparse::matrix_format::csr);

  std::shared_ptr<dpct::sparse::dense_matrix_desc> b_descr_s;
  std::shared_ptr<dpct::sparse::dense_matrix_desc> b_descr_d;
  std::shared_ptr<dpct::sparse::dense_matrix_desc> b_descr_c;
  std::shared_ptr<dpct::sparse::dense_matrix_desc> b_descr_z;
  b_descr_s = std::make_shared<dpct::sparse::dense_matrix_desc>(
      5, 2, 5, b_s.d_data, dpct::library_data_t::real_float,
      oneapi::mkl::layout::col_major);
  b_descr_d = std::make_shared<dpct::sparse::dense_matrix_desc>(
      5, 2, 5, b_d.d_data, dpct::library_data_t::real_double,
      oneapi::mkl::layout::col_major);
  b_descr_c = std::make_shared<dpct::sparse::dense_matrix_desc>(
      5, 2, 5, b_c.d_data, dpct::library_data_t::complex_float,
      oneapi::mkl::layout::col_major);
  b_descr_z = std::make_shared<dpct::sparse::dense_matrix_desc>(
      5, 2, 5, b_z.d_data, dpct::library_data_t::complex_double,
      oneapi::mkl::layout::col_major);

  std::shared_ptr<dpct::sparse::dense_matrix_desc> c_descr_s;
  std::shared_ptr<dpct::sparse::dense_matrix_desc> c_descr_d;
  std::shared_ptr<dpct::sparse::dense_matrix_desc> c_descr_c;
  std::shared_ptr<dpct::sparse::dense_matrix_desc> c_descr_z;
  c_descr_s = std::make_shared<dpct::sparse::dense_matrix_desc>(
      4, 2, 4, c_s.d_data, dpct::library_data_t::real_float,
      oneapi::mkl::layout::col_major);
  c_descr_d = std::make_shared<dpct::sparse::dense_matrix_desc>(
      4, 2, 4, c_d.d_data, dpct::library_data_t::real_double,
      oneapi::mkl::layout::col_major);
  c_descr_c = std::make_shared<dpct::sparse::dense_matrix_desc>(
      4, 2, 4, c_c.d_data, dpct::library_data_t::complex_float,
      oneapi::mkl::layout::col_major);
  c_descr_z = std::make_shared<dpct::sparse::dense_matrix_desc>(
      4, 2, 4, c_z.d_data, dpct::library_data_t::complex_double,
      oneapi::mkl::layout::col_major);

  size_t ws_size_s;
  size_t ws_size_d;
  size_t ws_size_c;
  size_t ws_size_z;
  ws_size_s = 0;
  ws_size_d = 0;
  ws_size_c = 0;
  ws_size_z = 0;

  void *ws_s;
  void *ws_d;
  void *ws_c;
  void *ws_z;
  ws_s = dpct::dpct_malloc(ws_size_s);
  ws_d = dpct::dpct_malloc(ws_size_d);
  ws_c = dpct::dpct_malloc(ws_size_c);
  ws_z = dpct::dpct_malloc(ws_size_z);

  /*
  DPCT1026:2: The call to cusparseSpMM_preprocess was removed because this call
  is redundant in SYCL.
  */
  /*
  DPCT1026:3: The call to cusparseSpMM_preprocess was removed because this call
  is redundant in SYCL.
  */
  /*
  DPCT1026:4: The call to cusparseSpMM_preprocess was removed because this call
  is redundant in SYCL.
  */
  /*
  DPCT1026:5: The call to cusparseSpMM_preprocess was removed because this call
  is redundant in SYCL.
  */
  dpct::sparse::spmm(*handle, oneapi::mkl::transpose::nontrans,
                     oneapi::mkl::transpose::nontrans, alpha_s.d_data,
                     a_descr_s, b_descr_s, beta_s.d_data, c_descr_s,
                     dpct::library_data_t::real_float);
  dpct::sparse::spmm(*handle, oneapi::mkl::transpose::nontrans,
                     oneapi::mkl::transpose::nontrans, alpha_d.d_data,
                     a_descr_d, b_descr_d, beta_d.d_data, c_descr_d,
                     dpct::library_data_t::real_double);
  if (run_complex_datatype) {
    dpct::sparse::spmm(*handle, oneapi::mkl::transpose::nontrans,
                       oneapi::mkl::transpose::nontrans, alpha_c.d_data,
                       a_descr_c, b_descr_c, beta_c.d_data, c_descr_c,
                       dpct::library_data_t::complex_float);
    dpct::sparse::spmm(*handle, oneapi::mkl::transpose::nontrans,
                       oneapi::mkl::transpose::nontrans, alpha_z.d_data,
                       a_descr_z, b_descr_z, beta_z.d_data, c_descr_z,
                       dpct::library_data_t::complex_double);
  }

  c_s.D2H();
  c_d.D2H();
  c_c.D2H();
  c_z.D2H();

  q_ct1.wait();

  dpct::dpct_free(ws_s);
  dpct::dpct_free(ws_d);
  dpct::dpct_free(ws_c);
  dpct::dpct_free(ws_z);
  a_descr_s.reset();
  a_descr_d.reset();
  a_descr_c.reset();
  a_descr_z.reset();
  b_descr_s.reset();
  b_descr_d.reset();
  b_descr_c.reset();
  b_descr_z.reset();
  c_descr_s.reset();
  c_descr_d.reset();
  c_descr_c.reset();
  c_descr_z.reset();
  handle = nullptr;

  float expect_c[8] = {90, 130, 730, 570, 340, 380, 1730, 1320};
  if (compare_result(expect_c, c_s.h_data, 8) &&
      compare_result(expect_c, c_d.h_data, 8)/* &&
      compare_result(expect_c, c_c.h_data, 8) &&
      compare_result(expect_c, c_z.h_data, 8)*/)
    printf("SpMM pass\n");
  else {
    printf("SpMM fail\n");
    test_passed = false;
  }
}

int main() {
  test_cusparseSetGetStream();
  test_cusparseTcsrmv_ge();
  test_cusparseTcsrmv_sy();
  test_cusparseTcsrmv_tr();
  // test_cusparseTcsrmm(); // Re-enable this test until MKL issue fixed
  test_cusparseTcsrsv();
  test_cusparseSpMV();
  test_cusparseSpMM();

  if (test_passed)
    return 0;
  return -1;
}
