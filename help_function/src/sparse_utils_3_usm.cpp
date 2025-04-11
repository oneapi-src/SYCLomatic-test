// ===------- sparse_utils_3_usm.cpp ------------------------ *- C++ -* ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

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
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
    h_data = (float *)malloc(sizeof(float) * element_num);
    memset(h_data, 0, sizeof(float) * element_num);
    d_data =
        (d_data_t *)sycl::malloc_device(sizeof(d_data_t) * element_num, q_ct1);
    q_ct1.memset(d_data, 0, sizeof(d_data_t) * element_num).wait();
  }
  Data(float *input_data, int element_num) : element_num(element_num) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
    h_data = (float *)malloc(sizeof(float) * element_num);
    d_data =
        (d_data_t *)sycl::malloc_device(sizeof(d_data_t) * element_num, q_ct1);
    q_ct1.memset(d_data, 0, sizeof(d_data_t) * element_num).wait();
    memcpy(h_data, input_data, sizeof(float) * element_num);
  }
  ~Data() {
    free(h_data);
    sycl::free(d_data, dpct::get_default_queue());
  }
  void H2D() {
    d_data_t *h_temp = (d_data_t *)malloc(sizeof(d_data_t) * element_num);
    memset(h_temp, 0, sizeof(d_data_t) * element_num);
    from_float_convert(h_data, h_temp);
    dpct::get_default_queue()
        .memcpy(d_data, h_temp, sizeof(d_data_t) * element_num)
        .wait();
    free(h_temp);
  }
  void D2H() {
    d_data_t *h_temp = (d_data_t *)malloc(sizeof(d_data_t) * element_num);
    memset(h_temp, 0, sizeof(d_data_t) * element_num);
    dpct::get_default_queue()
        .memcpy(h_temp, d_data, sizeof(d_data_t) * element_num)
        .wait();
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

// A * x = f
//
// | 1 1 2 |   | 1 |   | 9  |  
// | 0 1 3 | * | 2 | = | 11 |
// | 0 0 1 |   | 3 |   | 3  |
void test_cusparseTcsrsv() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  std::vector<float> a_val_vec = {1, 1, 2, 1, 3, 1};
  Data<float> a_s_val(a_val_vec.data(), 6);
  Data<double> a_d_val(a_val_vec.data(), 6);
  Data<sycl::float2> a_c_val(a_val_vec.data(), 6);
  Data<sycl::double2> a_z_val(a_val_vec.data(), 6);
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
  Data<sycl::float2> f_c(f_vec.data(), 3);
  Data<sycl::double2> f_z(f_vec.data(), 3);

  Data<float> x_s(3);
  Data<double> x_d(3);
  Data<sycl::float2> x_c(3);
  Data<sycl::double2> x_z(3);

  sycl::queue *handle;
  handle = &q_ct1;
  std::shared_ptr<dpct::sparse::optimize_info> info_s;
  std::shared_ptr<dpct::sparse::optimize_info> info_d;
  std::shared_ptr<dpct::sparse::optimize_info> info_c;
  std::shared_ptr<dpct::sparse::optimize_info> info_z;
  info_s = std::make_shared<dpct::sparse::optimize_info>();
  info_d = std::make_shared<dpct::sparse::optimize_info>();
  info_c = std::make_shared<dpct::sparse::optimize_info>();
  info_z = std::make_shared<dpct::sparse::optimize_info>();

  std::shared_ptr<dpct::sparse::matrix_info> descrA;
  descrA = std::make_shared<dpct::sparse::matrix_info>();
  descrA->set_index_base(oneapi::mkl::index_base::zero);
  descrA->set_matrix_type(dpct::sparse::matrix_info::matrix_type::tr);
  descrA->set_diag(oneapi::mkl::diag::unit);

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

  dpct::sparse::optimize_csrsv(*handle, oneapi::mkl::transpose::nontrans, 3,
                               descrA, (float *)a_s_val.d_data,
                               (int *)a_row_ptr_s.d_data,
                               (int *)a_col_ind_s.d_data, info_s);
  dpct::sparse::optimize_csrsv(*handle, oneapi::mkl::transpose::nontrans, 3,
                               descrA, (double *)a_d_val.d_data,
                               (int *)a_row_ptr_d.d_data,
                               (int *)a_col_ind_d.d_data, info_d);
  dpct::sparse::optimize_csrsv(*handle, oneapi::mkl::transpose::nontrans, 3,
                               descrA, (sycl::float2 *)a_c_val.d_data,
                               (int *)a_row_ptr_c.d_data,
                               (int *)a_col_ind_c.d_data, info_c);
  dpct::sparse::optimize_csrsv(*handle, oneapi::mkl::transpose::nontrans, 3,
                               descrA, (sycl::double2 *)a_z_val.d_data,
                               (int *)a_row_ptr_z.d_data,
                               (int *)a_col_ind_z.d_data, info_z);

  float alpha_s = 1;
  double alpha_d = 1;
  sycl::float2 alpha_c = sycl::float2{1, 0};
  sycl::double2 alpha_z = sycl::double2{1, 0};

  dpct::sparse::csrsv(*handle, oneapi::mkl::transpose::nontrans, 3, &alpha_s,
                      descrA, (float *)a_s_val.d_data,
                      (int *)a_row_ptr_s.d_data, (int *)a_col_ind_s.d_data,
                      info_s, f_s.d_data, x_s.d_data);
  dpct::sparse::csrsv(*handle, oneapi::mkl::transpose::nontrans, 3, &alpha_d,
                      descrA, (double *)a_d_val.d_data,
                      (int *)a_row_ptr_d.d_data, (int *)a_col_ind_d.d_data,
                      info_d, f_d.d_data, x_d.d_data);
  dpct::sparse::csrsv(*handle, oneapi::mkl::transpose::nontrans, 3, &alpha_c,
                      descrA, (sycl::float2 *)a_c_val.d_data,
                      (int *)a_row_ptr_c.d_data, (int *)a_col_ind_c.d_data,
                      info_c, f_c.d_data, x_c.d_data);
  dpct::sparse::csrsv(*handle, oneapi::mkl::transpose::nontrans, 3, &alpha_z,
                      descrA, (sycl::double2 *)a_z_val.d_data,
                      (int *)a_row_ptr_z.d_data, (int *)a_col_ind_z.d_data,
                      info_z, f_z.d_data, x_z.d_data);

  x_s.D2H();
  x_d.D2H();
  x_c.D2H();
  x_z.D2H();

  q_ct1.wait();
  info_s.reset();
  info_d.reset();
  info_c.reset();
  info_z.reset();
  /*
  DPCT1026:24: The call to cusparseDestroyMatDescr was removed because this call
  is redundant in SYCL.
  */
  handle = nullptr;

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
}

// A * x = f
//
// | 1 1 2 |   | 1 |   | 9  |  
// | 0 1 3 | * | 2 | = | 11 |
// | 0 0 1 |   | 3 |   | 3  |
void test_cusparseTcsrsv2() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  std::vector<float> a_val_vec = {1, 1, 2, 1, 3, 1};
  Data<float> a_s_val(a_val_vec.data(), 6);
  Data<double> a_d_val(a_val_vec.data(), 6);
  Data<sycl::float2> a_c_val(a_val_vec.data(), 6);
  Data<sycl::double2> a_z_val(a_val_vec.data(), 6);
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
  Data<sycl::float2> f_c(f_vec.data(), 3);
  Data<sycl::double2> f_z(f_vec.data(), 3);

  Data<float> x_s(3);
  Data<double> x_d(3);
  Data<sycl::float2> x_c(3);
  Data<sycl::double2> x_z(3);

  sycl::queue *handle;
  handle = &q_ct1;
  std::shared_ptr<dpct::sparse::optimize_info> info_s;
  std::shared_ptr<dpct::sparse::optimize_info> info_d;
  std::shared_ptr<dpct::sparse::optimize_info> info_c;
  std::shared_ptr<dpct::sparse::optimize_info> info_z;
  info_s = std::make_shared<dpct::sparse::optimize_info>();
  info_d = std::make_shared<dpct::sparse::optimize_info>();
  info_c = std::make_shared<dpct::sparse::optimize_info>();
  info_z = std::make_shared<dpct::sparse::optimize_info>();
  int policy = 1;
  policy = 0;

  std::shared_ptr<dpct::sparse::matrix_info> descrA;
  descrA = std::make_shared<dpct::sparse::matrix_info>();
  descrA->set_index_base(oneapi::mkl::index_base::zero);
  descrA->set_matrix_type(dpct::sparse::matrix_info::matrix_type::tr);
  descrA->set_diag(oneapi::mkl::diag::unit);

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
  buffer_size_s0 = 0;
  buffer_size_d0 = 0;
  buffer_size_c0 = 0;
  buffer_size_z0 = 0;

  size_t buffer_size_s;
  size_t buffer_size_d;
  size_t buffer_size_c;
  size_t buffer_size_z;
  buffer_size_s = 0;
  buffer_size_d = 0;
  buffer_size_c = 0;
  buffer_size_z = 0;

  void* buffer_s;
  void* buffer_d;
  void* buffer_c;
  void* buffer_z;
  buffer_s = (void *)sycl::malloc_device(buffer_size_s, q_ct1);
  buffer_d = (void *)sycl::malloc_device(buffer_size_d, q_ct1);
  buffer_c = (void *)sycl::malloc_device(buffer_size_c, q_ct1);
  buffer_z = (void *)sycl::malloc_device(buffer_size_z, q_ct1);

  dpct::sparse::optimize_csrsv(*handle, oneapi::mkl::transpose::nontrans, 3,
                               descrA, (float *)a_s_val.d_data,
                               (int *)a_row_ptr_s.d_data,
                               (int *)a_col_ind_s.d_data, info_s);
  dpct::sparse::optimize_csrsv(*handle, oneapi::mkl::transpose::nontrans, 3,
                               descrA, (double *)a_d_val.d_data,
                               (int *)a_row_ptr_d.d_data,
                               (int *)a_col_ind_d.d_data, info_d);
  dpct::sparse::optimize_csrsv(*handle, oneapi::mkl::transpose::nontrans, 3,
                               descrA, (sycl::float2 *)a_c_val.d_data,
                               (int *)a_row_ptr_c.d_data,
                               (int *)a_col_ind_c.d_data, info_c);
  dpct::sparse::optimize_csrsv(*handle, oneapi::mkl::transpose::nontrans, 3,
                               descrA, (sycl::double2 *)a_z_val.d_data,
                               (int *)a_row_ptr_z.d_data,
                               (int *)a_col_ind_z.d_data, info_z);

  float alpha_s = 1;
  double alpha_d = 1;
  sycl::float2 alpha_c = sycl::float2{1, 0};
  sycl::double2 alpha_z = sycl::double2{1, 0};

  dpct::sparse::csrsv(*handle, oneapi::mkl::transpose::nontrans, 3, &alpha_s,
                      descrA, (float *)a_s_val.d_data,
                      (int *)a_row_ptr_s.d_data, (int *)a_col_ind_s.d_data,
                      info_s, f_s.d_data, x_s.d_data);
  dpct::sparse::csrsv(*handle, oneapi::mkl::transpose::nontrans, 3, &alpha_d,
                      descrA, (double *)a_d_val.d_data,
                      (int *)a_row_ptr_d.d_data, (int *)a_col_ind_d.d_data,
                      info_d, f_d.d_data, x_d.d_data);
  dpct::sparse::csrsv(*handle, oneapi::mkl::transpose::nontrans, 3, &alpha_c,
                      descrA, (sycl::float2 *)a_c_val.d_data,
                      (int *)a_row_ptr_c.d_data, (int *)a_col_ind_c.d_data,
                      info_c, f_c.d_data, x_c.d_data);
  dpct::sparse::csrsv(*handle, oneapi::mkl::transpose::nontrans, 3, &alpha_z,
                      descrA, (sycl::double2 *)a_z_val.d_data,
                      (int *)a_row_ptr_z.d_data, (int *)a_col_ind_z.d_data,
                      info_z, f_z.d_data, x_z.d_data);

  x_s.D2H();
  x_d.D2H();
  x_c.D2H();
  x_z.D2H();

  q_ct1.wait();
  info_s.reset();
  info_d.reset();
  info_c.reset();
  info_z.reset();
  /*
  DPCT1026:33: The call to cusparseDestroyMatDescr was removed because this call
  is redundant in SYCL.
  */
  handle = nullptr;
  sycl::free(buffer_s, q_ct1);
  sycl::free(buffer_d, q_ct1);
  sycl::free(buffer_c, q_ct1);
  sycl::free(buffer_z, q_ct1);

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

void test_cusparseTcsrmm2() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
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
  DPCT1026:32: The call to cusparseSetPointerMode was removed because this call
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

  std::shared_ptr<dpct::sparse::matrix_info> descrA;
  descrA = std::make_shared<dpct::sparse::matrix_info>();
  descrA->set_index_base(oneapi::mkl::index_base::zero);
  descrA->set_matrix_type(dpct::sparse::matrix_info::matrix_type::ge);

  /*
  DPCT1045:33: Migration is only supported for this API for the general sparse
  matrix type. You may need to adjust the code.
  */
  dpct::sparse::csrmm(*handle, oneapi::mkl::transpose::nontrans,
                      oneapi::mkl::transpose::nontrans, 4, 2, 5, alpha_s.d_data,
                      descrA, a_s_val.d_data, a_row_ptr.d_data,
                      a_col_ind.d_data, b_s.d_data, 5, beta_s.d_data,
                      c_s.d_data, 4);
  /*
  DPCT1045:34: Migration is only supported for this API for the general sparse
  matrix type. You may need to adjust the code.
  */
  dpct::sparse::csrmm(*handle, oneapi::mkl::transpose::nontrans,
                      oneapi::mkl::transpose::nontrans, 4, 2, 5, alpha_d.d_data,
                      descrA, a_d_val.d_data, a_row_ptr.d_data,
                      a_col_ind.d_data, b_d.d_data, 5, beta_d.d_data,
                      c_d.d_data, 4);
  /*
  DPCT1045:35: Migration is only supported for this API for the general sparse
  matrix type. You may need to adjust the code.
  */
  dpct::sparse::csrmm(*handle, oneapi::mkl::transpose::nontrans,
                      oneapi::mkl::transpose::nontrans, 4, 2, 5, alpha_c.d_data,
                      descrA, a_c_val.d_data, a_row_ptr.d_data,
                      a_col_ind.d_data, b_c.d_data, 5, beta_c.d_data,
                      c_c.d_data, 4);
  /*
  DPCT1045:36: Migration is only supported for this API for the general sparse
  matrix type. You may need to adjust the code.
  */
  dpct::sparse::csrmm(*handle, oneapi::mkl::transpose::nontrans,
                      oneapi::mkl::transpose::nontrans, 4, 2, 5, alpha_z.d_data,
                      descrA, a_z_val.d_data, a_row_ptr.d_data,
                      a_col_ind.d_data, b_z.d_data, 5, beta_z.d_data,
                      c_z.d_data, 4);

  c_s.D2H();
  c_d.D2H();
  c_c.D2H();
  c_z.D2H();

  q_ct1.wait();

  /*
  DPCT1026:37: The call to cusparseDestroyMatDescr was removed because this call
  is redundant in SYCL.
  */
  handle = nullptr;

  float expect_c[8] = {90, 130, 730, 570, 340, 380, 1730, 1320};
  if (compare_result(expect_c, c_s.h_data, 8) &&
      compare_result(expect_c, c_d.h_data, 8) &&
      compare_result(expect_c, c_c.h_data, 8) &&
      compare_result(expect_c, c_z.h_data, 8))
    printf("Tcsrmm2 pass\n");
  else {
    printf("Tcsrmm2 fail\n");
    test_passed = false;
  }
}

// 2*A*B + 3*D = C
//
// 2 * | 0 1 2 |   | 1 0 0 0 | + 3 * | 1 0 0 0 |   | 4 6 20 24 | + | 3  0  0 0  | = | 7  6  20 24 |
//     | 0 0 3 | * | 2 3 0 0 |       | 5 6 0 0 | = | 0 0 30 36 |   | 15 18 0 0  |   | 15 18 30 36 |
//     | 4 0 0 |   | 0 0 5 6 |       | 0 0 0 7 |   | 8 0 0  0  |   | 0  0  0 21 |   | 8  0  0  21 |
void test_cusparseTcsrgemm2() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  std::vector<float> a_val_vec = {1, 2, 3, 4};
  Data<float> a_s_val(a_val_vec.data(), 4);
  Data<double> a_d_val(a_val_vec.data(), 4);
  Data<sycl::float2> a_c_val(a_val_vec.data(), 4);
  Data<sycl::double2> a_z_val(a_val_vec.data(), 4);
  std::vector<float> a_row_ptr_vec = {0, 2, 3, 4};
  Data<int> a_row_ptr(a_row_ptr_vec.data(), 4);
  std::vector<float> a_col_ind_vec = {1, 2, 2, 0};
  Data<int> a_col_ind(a_col_ind_vec.data(), 4);

  std::vector<float> b_val_vec = {1, 2, 3, 5, 6};
  Data<float> b_s_val(b_val_vec.data(), 5);
  Data<double> b_d_val(b_val_vec.data(), 5);
  Data<sycl::float2> b_c_val(b_val_vec.data(), 5);
  Data<sycl::double2> b_z_val(b_val_vec.data(), 5);
  std::vector<float> b_row_ptr_vec = {0, 1, 3, 5};
  Data<int> b_row_ptr(b_row_ptr_vec.data(), 4);
  std::vector<float> b_col_ind_vec = {0, 0, 1, 2, 3};
  Data<int> b_col_ind(b_col_ind_vec.data(), 5);

  std::vector<float> d_val_vec = {1, 5, 6, 7};
  Data<float> d_s_val(d_val_vec.data(), 4);
  Data<double> d_d_val(d_val_vec.data(), 4);
  Data<sycl::float2> d_c_val(d_val_vec.data(), 4);
  Data<sycl::double2> d_z_val(d_val_vec.data(), 4);
  std::vector<float> d_row_ptr_vec = {0, 1, 3, 4};
  Data<int> d_row_ptr(d_row_ptr_vec.data(), 4);
  std::vector<float> d_col_ind_vec = {0, 0, 1, 3};
  Data<int> d_col_ind(d_col_ind_vec.data(), 4);

  float alpha = 2;
  Data<float> alpha_s(&alpha, 1);
  Data<double> alpha_d(&alpha, 1);
  Data<sycl::float2> alpha_c(&alpha, 1);
  Data<sycl::double2> alpha_z(&alpha, 1);

  float beta = 3;
  Data<float> beta_s(&beta, 1);
  Data<double> beta_d(&beta, 1);
  Data<sycl::float2> beta_c(&beta, 1);
  Data<sycl::double2> beta_z(&beta, 1);

  dpct::sparse::descriptor_ptr handle;
  handle = new dpct::sparse::descriptor();

  /*
  DPCT1026:1: The call to cusparseSetPointerMode was removed because this
  functionality is redundant in SYCL.
  */

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

  std::shared_ptr<dpct::sparse::csrgemm2_info> info_s;
  std::shared_ptr<dpct::sparse::csrgemm2_info> info_d;
  std::shared_ptr<dpct::sparse::csrgemm2_info> info_c;
  std::shared_ptr<dpct::sparse::csrgemm2_info> info_z;
  info_s = std::make_shared<dpct::sparse::csrgemm2_info>();
  info_d = std::make_shared<dpct::sparse::csrgemm2_info>();
  info_c = std::make_shared<dpct::sparse::csrgemm2_info>();
  info_z = std::make_shared<dpct::sparse::csrgemm2_info>();

  const int m = 3;
  const int n = 4;
  const int k = 3;
  const int nnzA = 4;
  const int nnzB = 5;
  const int nnzD = 4;

  std::shared_ptr<dpct::sparse::matrix_info> descrA;
  descrA = std::make_shared<dpct::sparse::matrix_info>();
  descrA->set_matrix_type(dpct::sparse::matrix_info::matrix_type::ge);
  descrA->set_index_base(oneapi::mkl::index_base::zero);
  std::shared_ptr<dpct::sparse::matrix_info> descrB;
  descrB = std::make_shared<dpct::sparse::matrix_info>();
  descrB->set_matrix_type(dpct::sparse::matrix_info::matrix_type::ge);
  descrB->set_index_base(oneapi::mkl::index_base::zero);
  std::shared_ptr<dpct::sparse::matrix_info> descrC;
  descrC = std::make_shared<dpct::sparse::matrix_info>();
  descrC->set_matrix_type(dpct::sparse::matrix_info::matrix_type::ge);
  descrC->set_index_base(oneapi::mkl::index_base::zero);
  std::shared_ptr<dpct::sparse::matrix_info> descrD;
  descrD = std::make_shared<dpct::sparse::matrix_info>();
  descrD->set_matrix_type(dpct::sparse::matrix_info::matrix_type::ge);
  descrD->set_index_base(oneapi::mkl::index_base::zero);

  size_t ws_1_size_s = 0;
  size_t ws_1_size_d = 0;
  size_t ws_1_size_c = 0;
  size_t ws_1_size_z = 0;
  dpct::sparse::csrgemm2_get_buffer_size<float>(
      handle, m, n, k, alpha_s.d_data, descrA, nnzA, a_row_ptr.d_data,
      a_col_ind.d_data, descrB, nnzB, b_row_ptr.d_data, b_col_ind.d_data,
      beta_s.d_data, descrD, nnzD, d_row_ptr.d_data, d_col_ind.d_data, info_s,
      &ws_1_size_s);
  dpct::sparse::csrgemm2_get_buffer_size<double>(
      handle, m, n, k, alpha_d.d_data, descrA, nnzA, a_row_ptr.d_data,
      a_col_ind.d_data, descrB, nnzB, b_row_ptr.d_data, b_col_ind.d_data,
      beta_d.d_data, descrD, nnzD, d_row_ptr.d_data, d_col_ind.d_data, info_d,
      &ws_1_size_d);
  dpct::sparse::csrgemm2_get_buffer_size<sycl::float2>(
      handle, m, n, k, alpha_c.d_data, descrA, nnzA, a_row_ptr.d_data,
      a_col_ind.d_data, descrB, nnzB, b_row_ptr.d_data, b_col_ind.d_data,
      beta_c.d_data, descrD, nnzD, d_row_ptr.d_data, d_col_ind.d_data, info_c,
      &ws_1_size_c);
  dpct::sparse::csrgemm2_get_buffer_size<sycl::double2>(
      handle, m, n, k, alpha_z.d_data, descrA, nnzA, a_row_ptr.d_data,
      a_col_ind.d_data, descrB, nnzB, b_row_ptr.d_data, b_col_ind.d_data,
      beta_z.d_data, descrD, nnzD, d_row_ptr.d_data, d_col_ind.d_data, info_z,
      &ws_1_size_z);

  void *ws_1_s = nullptr;
  void *ws_1_d = nullptr;
  void *ws_1_c = nullptr;
  void *ws_1_z = nullptr;

  ws_1_s = (void *)sycl::malloc_device(ws_1_size_s, q_ct1);
  ws_1_d = (void *)sycl::malloc_device(ws_1_size_d, q_ct1);
  ws_1_c = (void *)sycl::malloc_device(ws_1_size_c, q_ct1);
  ws_1_z = (void *)sycl::malloc_device(ws_1_size_z, q_ct1);

  Data<int> c_s_row_ptr(m + 1);
  Data<int> c_d_row_ptr(m + 1);
  Data<int> c_c_row_ptr(m + 1);
  Data<int> c_z_row_ptr(m + 1);

  Data<int> nnzC_s(1);
  Data<int> nnzC_d(1);
  Data<int> nnzC_c(1);
  Data<int> nnzC_z(1);
  dpct::sparse::csrgemm2_nnz(handle, m, n, k, descrA, nnzA, a_row_ptr.d_data,
                             a_col_ind.d_data, descrB, nnzB, b_row_ptr.d_data,
                             b_col_ind.d_data, descrD, nnzD, d_row_ptr.d_data,
                             d_col_ind.d_data, descrC, c_s_row_ptr.d_data,
                             nnzC_s.d_data, info_s, ws_1_s);
  dpct::sparse::csrgemm2_nnz(handle, m, n, k, descrA, nnzA, a_row_ptr.d_data,
                             a_col_ind.d_data, descrB, nnzB, b_row_ptr.d_data,
                             b_col_ind.d_data, descrD, nnzD, d_row_ptr.d_data,
                             d_col_ind.d_data, descrC, c_d_row_ptr.d_data,
                             nnzC_d.d_data, info_d, ws_1_d);
  dpct::sparse::csrgemm2_nnz(handle, m, n, k, descrA, nnzA, a_row_ptr.d_data,
                             a_col_ind.d_data, descrB, nnzB, b_row_ptr.d_data,
                             b_col_ind.d_data, descrD, nnzD, d_row_ptr.d_data,
                             d_col_ind.d_data, descrC, c_c_row_ptr.d_data,
                             nnzC_c.d_data, info_c, ws_1_c);
  dpct::sparse::csrgemm2_nnz(handle, m, n, k, descrA, nnzA, a_row_ptr.d_data,
                             a_col_ind.d_data, descrB, nnzB, b_row_ptr.d_data,
                             b_col_ind.d_data, descrD, nnzD, d_row_ptr.d_data,
                             d_col_ind.d_data, descrC, c_z_row_ptr.d_data,
                             nnzC_z.d_data, info_z, ws_1_z);

  q_ct1.wait();

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
  Data<sycl::float2> c_c_val(nnzC_c_int);
  Data<sycl::double2> c_z_val(nnzC_z_int);
  Data<int> c_s_col_ind(nnzC_s_int);
  Data<int> c_d_col_ind(nnzC_d_int);
  Data<int> c_c_col_ind(nnzC_c_int);
  Data<int> c_z_col_ind(nnzC_z_int);

  dpct::sparse::csrgemm2<float>(
      handle, m, n, k, alpha_s.d_data, descrA, nnzA, a_s_val.d_data,
      a_row_ptr.d_data, a_col_ind.d_data, descrB, nnzB, b_s_val.d_data,
      b_row_ptr.d_data, b_col_ind.d_data, beta_s.d_data, descrD, nnzD,
      d_s_val.d_data, d_row_ptr.d_data, d_col_ind.d_data, descrC,
      c_s_val.d_data, c_s_row_ptr.d_data, c_s_col_ind.d_data, info_s, ws_1_s);
  dpct::sparse::csrgemm2<double>(
      handle, m, n, k, alpha_d.d_data, descrA, nnzA, a_d_val.d_data,
      a_row_ptr.d_data, a_col_ind.d_data, descrB, nnzB, b_d_val.d_data,
      b_row_ptr.d_data, b_col_ind.d_data, beta_d.d_data, descrD, nnzD,
      d_d_val.d_data, d_row_ptr.d_data, d_col_ind.d_data, descrC,
      c_d_val.d_data, c_d_row_ptr.d_data, c_d_col_ind.d_data, info_d, ws_1_d);
  dpct::sparse::csrgemm2<sycl::float2>(
      handle, m, n, k, alpha_c.d_data, descrA, nnzA, a_c_val.d_data,
      a_row_ptr.d_data, a_col_ind.d_data, descrB, nnzB, b_c_val.d_data,
      b_row_ptr.d_data, b_col_ind.d_data, beta_c.d_data, descrD, nnzD,
      d_c_val.d_data, d_row_ptr.d_data, d_col_ind.d_data, descrC,
      c_c_val.d_data, c_c_row_ptr.d_data, c_c_col_ind.d_data, info_c, ws_1_c);
  dpct::sparse::csrgemm2<sycl::double2>(
      handle, m, n, k, alpha_z.d_data, descrA, nnzA, a_z_val.d_data,
      a_row_ptr.d_data, a_col_ind.d_data, descrB, nnzB, b_z_val.d_data,
      b_row_ptr.d_data, b_col_ind.d_data, beta_z.d_data, descrD, nnzD,
      d_z_val.d_data, d_row_ptr.d_data, d_col_ind.d_data, descrC,
      c_z_val.d_data, c_z_row_ptr.d_data, c_z_col_ind.d_data, info_z, ws_1_z);

  q_ct1.wait();

  dpct::dpct_free(ws_1_s, q_ct1);
  dpct::dpct_free(ws_1_d, q_ct1);
  dpct::dpct_free(ws_1_c, q_ct1);
  dpct::dpct_free(ws_1_z, q_ct1);
  info_s.reset();
  info_d.reset();
  info_c.reset();
  info_z.reset();
  delete (handle);

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

// 3*3     3*2             3*2
// op(A) * op(X) = alpha * op(B)
// 1 0 0   1 4       1   * 1  4
// 0 2 0   2 5             4  10
// 0 4 3   3 6             17 38
void test_cusparseTcsrsm2() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  const int nrhs = 2;
  const int m = 3;
  const int nnz = 4;

  std::vector<float> a_val_vec = {1, 2, 4, 3};
  Data<float> a_s_val(a_val_vec.data(), 4);
  Data<double> a_d_val(a_val_vec.data(), 4);
  Data<sycl::float2> a_c_val(a_val_vec.data(), 4);
  Data<sycl::double2> a_z_val(a_val_vec.data(), 4);
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
  Data<sycl::float2> b_c(b_vec.data(), m * nrhs);
  Data<sycl::double2> b_z(b_vec.data(), m * nrhs);

  dpct::sparse::descriptor_ptr handle;
  handle = new dpct::sparse::descriptor();
  std::shared_ptr<dpct::sparse::optimize_info> info_s;
  std::shared_ptr<dpct::sparse::optimize_info> info_d;
  std::shared_ptr<dpct::sparse::optimize_info> info_c;
  std::shared_ptr<dpct::sparse::optimize_info> info_z;
  info_s = std::make_shared<dpct::sparse::optimize_info>();
  info_d = std::make_shared<dpct::sparse::optimize_info>();
  info_c = std::make_shared<dpct::sparse::optimize_info>();
  info_z = std::make_shared<dpct::sparse::optimize_info>();
  int policy = 1;
  policy = 0;

  std::shared_ptr<dpct::sparse::matrix_info> descrA;
  descrA = std::make_shared<dpct::sparse::matrix_info>();
  descrA->set_index_base(oneapi::mkl::index_base::zero);
  descrA->set_matrix_type(dpct::sparse::matrix_info::matrix_type::tr);
  descrA->set_diag(oneapi::mkl::diag::nonunit);
  descrA->set_uplo(oneapi::mkl::uplo::lower);

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
  sycl::float2 alpha_c = sycl::float2{1, 0};
  sycl::double2 alpha_z = sycl::double2{1, 0};

  size_t buffer_size_s;
  size_t buffer_size_d;
  size_t buffer_size_c;
  size_t buffer_size_z;
  buffer_size_s = 0;
  buffer_size_d = 0;
  buffer_size_c = 0;
  buffer_size_z = 0;

  void* buffer_s;
  void* buffer_d;
  void* buffer_c;
  void* buffer_z;
  buffer_s = (void *)sycl::malloc_device(buffer_size_s, q_ct1);
  buffer_d = (void *)sycl::malloc_device(buffer_size_d, q_ct1);
  buffer_c = (void *)sycl::malloc_device(buffer_size_c, q_ct1);
  buffer_z = (void *)sycl::malloc_device(buffer_size_z, q_ct1);

  dpct::sparse::optimize_csrsm(
      handle->get_queue(), oneapi::mkl::transpose::nontrans,
      oneapi::mkl::transpose::trans, m, nrhs, descrA, (float *)a_s_val.d_data,
      (int *)a_row_ptr_s.d_data, (int *)a_col_ind_s.d_data, info_s);
  dpct::sparse::optimize_csrsm(
      handle->get_queue(), oneapi::mkl::transpose::nontrans,
      oneapi::mkl::transpose::trans, m, nrhs, descrA, (double *)a_d_val.d_data,
      (int *)a_row_ptr_d.d_data, (int *)a_col_ind_d.d_data, info_d);
  dpct::sparse::optimize_csrsm(
      handle->get_queue(), oneapi::mkl::transpose::nontrans,
      oneapi::mkl::transpose::trans, m, nrhs, descrA,
      (sycl::float2 *)a_c_val.d_data, (int *)a_row_ptr_c.d_data,
      (int *)a_col_ind_c.d_data, info_c);
  dpct::sparse::optimize_csrsm(
      handle->get_queue(), oneapi::mkl::transpose::nontrans,
      oneapi::mkl::transpose::trans, m, nrhs, descrA,
      (sycl::double2 *)a_z_val.d_data, (int *)a_row_ptr_z.d_data,
      (int *)a_col_ind_z.d_data, info_z);

  dpct::sparse::csrsm(handle->get_queue(), oneapi::mkl::transpose::nontrans,
                      oneapi::mkl::transpose::trans, m, nrhs, &alpha_s, descrA,
                      (float *)a_s_val.d_data, (int *)a_row_ptr_s.d_data,
                      (int *)a_col_ind_s.d_data, (float *)b_s.d_data, nrhs,
                      info_s);
  dpct::sparse::csrsm(handle->get_queue(), oneapi::mkl::transpose::nontrans,
                      oneapi::mkl::transpose::trans, m, nrhs, &alpha_d, descrA,
                      (double *)a_d_val.d_data, (int *)a_row_ptr_d.d_data,
                      (int *)a_col_ind_d.d_data, (double *)b_d.d_data, nrhs,
                      info_d);
  dpct::sparse::csrsm(handle->get_queue(), oneapi::mkl::transpose::nontrans,
                      oneapi::mkl::transpose::trans, m, nrhs, &alpha_c, descrA,
                      (sycl::float2 *)a_c_val.d_data, (int *)a_row_ptr_c.d_data,
                      (int *)a_col_ind_c.d_data, (sycl::float2 *)b_c.d_data,
                      nrhs, info_c);
  dpct::sparse::csrsm(handle->get_queue(), oneapi::mkl::transpose::nontrans,
                      oneapi::mkl::transpose::trans, m, nrhs, &alpha_z, descrA,
                      (sycl::double2 *)a_z_val.d_data,
                      (int *)a_row_ptr_z.d_data, (int *)a_col_ind_z.d_data,
                      (sycl::double2 *)b_z.d_data, nrhs, info_z);

  q_ct1.wait();

  b_s.D2H();
  b_d.D2H();
  b_c.D2H();
  b_z.D2H();

  info_s.reset();
  info_d.reset();
  info_c.reset();
  info_z.reset();
  /*
  DPCT1026:0: The call to cusparseDestroyMatDescr was removed because this
  functionality is redundant in SYCL.
  */
  delete (handle);
  dpct::dpct_free(buffer_s, q_ct1);
  dpct::dpct_free(buffer_d, q_ct1);
  dpct::dpct_free(buffer_c, q_ct1);
  dpct::dpct_free(buffer_z, q_ct1);

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
  test_cusparseTcsrsv();
  test_cusparseTcsrsv2();
  test_cusparseTcsrmm2();
  test_cusparseTcsrgemm2();
  test_cusparseTcsrsm2();

  if (test_passed)
    return 0;
  return -1;
}
