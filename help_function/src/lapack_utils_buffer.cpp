// ===------ lapack_utils_buffer.cpp -------------------------*- C++ -*-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#define DPCT_USM_LEVEL_NONE
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <oneapi/mkl.hpp>
#include <dpct/lapack_utils.hpp>

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
    d_data = (d_data_t *)dpct::dpct_malloc(sizeof(d_data_t) * element_num);
    dpct::dpct_memset(d_data, 0, sizeof(d_data_t) * element_num);
  }
  Data(float* input_data, int element_num) : element_num(element_num) {
    h_data = (float*)malloc(sizeof(float) * element_num);
    d_data = (d_data_t *)dpct::dpct_malloc(sizeof(d_data_t) * element_num);
    dpct::dpct_memset(d_data, 0, sizeof(d_data_t) * element_num);
    memcpy(h_data, input_data, sizeof(float) * element_num);
  }
  ~Data() {
    free(h_data);
    dpct::dpct_free(d_data);
  }
  void H2D() {
    d_data_t* h_temp = (d_data_t*)malloc(sizeof(d_data_t) * element_num);
    memset(h_temp, 0, sizeof(d_data_t) * element_num);
    from_float_convert(h_data, h_temp);
    dpct::dpct_memcpy(d_data, h_temp, sizeof(d_data_t) * element_num,
                      dpct::host_to_device);
    free(h_temp);
  }
  void D2H() {
    d_data_t* h_temp = (d_data_t*)malloc(sizeof(d_data_t) * element_num);
    memset(h_temp, 0, sizeof(d_data_t) * element_num);
    dpct::dpct_memcpy(h_temp, d_data, sizeof(d_data_t) * element_num,
                      dpct::device_to_host);
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

void test_cusolverDnTsygvd() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::vector<float> a = {1, 2, 3, 2, 1, 2, 3, 2, 1};
  std::vector<float> b = {2, -1, 0, -1, 2, -1, 0, -1, 2};
  Data<float> a_s(a.data(), 9);
  Data<double> a_d(a.data(), 9);
  Data<float> b_s(b.data(), 9);
  Data<double> b_d(b.data(), 9);
  Data<float> w_s(3);
  Data<double> w_d(3);

  sycl::queue *handle;
  handle = &q_ct1;

  a_s.H2D();
  a_d.H2D();
  b_s.H2D();
  b_d.H2D();

  int lwork_s;
  int lwork_d;
  lwork_s = oneapi::mkl::lapack::sygvd_scratchpad_size<float>(
      *handle, 1, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 3, 3, 3);
  lwork_d = oneapi::mkl::lapack::sygvd_scratchpad_size<double>(
      *handle, 1, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 3, 3, 3);

  float* work_s;
  double* work_d;
  int *devInfo;
  work_s = (float *)dpct::dpct_malloc(sizeof(float) * lwork_s);
  work_d = (double *)dpct::dpct_malloc(sizeof(double) * lwork_d);
  devInfo = (int *)dpct::dpct_malloc(sizeof(int));

  dpct::lapack::sygvd(*handle, 1, oneapi::mkl::job::vec,
                      oneapi::mkl::uplo::upper, 3, a_s.d_data, 3, b_s.d_data, 3,
                      w_s.d_data, work_s, lwork_s, devInfo);
  dpct::lapack::sygvd(*handle, 1, oneapi::mkl::job::vec,
                      oneapi::mkl::uplo::upper, 3, a_d.d_data, 3, b_d.d_data, 3,
                      w_d.d_data, work_d, lwork_d, devInfo);

  a_s.D2H();
  a_d.D2H();
  b_s.D2H();
  b_d.D2H();
  w_s.D2H();
  w_d.D2H();

  q_ct1.wait();

  handle = nullptr;
  dpct::dpct_free(work_s);
  dpct::dpct_free(work_d);
  dpct::dpct_free(devInfo);

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
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::vector<float> a = {1, 2, 3, 2, 1, 2, 3, 2, 1};
  std::vector<float> b = {2, -1, 0, -1, 2, -1, 0, -1, 2};
  Data<sycl::float2> a_s(a.data(), 9);
  Data<sycl::double2> a_d(a.data(), 9);
  Data<sycl::float2> b_s(b.data(), 9);
  Data<sycl::double2> b_d(b.data(), 9);
  Data<float> w_s(3);
  Data<double> w_d(3);

  sycl::queue *handle;
  handle = &q_ct1;

  a_s.H2D();
  a_d.H2D();
  b_s.H2D();
  b_d.H2D();

  int lwork_s;
  int lwork_d;
  lwork_s = oneapi::mkl::lapack::hegvd_scratchpad_size<std::complex<float>>(
      *handle, 1, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 3, 3, 3);
  lwork_d = oneapi::mkl::lapack::hegvd_scratchpad_size<std::complex<double>>(
      *handle, 1, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 3, 3, 3);

  sycl::float2 *work_s;
  sycl::double2 *work_d;
  int *devInfo;
  work_s = (sycl::float2 *)dpct::dpct_malloc(sizeof(sycl::float2) * lwork_s);
  work_d = (sycl::double2 *)dpct::dpct_malloc(sizeof(sycl::double2) * lwork_d);
  devInfo = (int *)dpct::dpct_malloc(sizeof(int));

  dpct::lapack::hegvd(*handle, 1, oneapi::mkl::job::vec,
                      oneapi::mkl::uplo::upper, 3, a_s.d_data, 3, b_s.d_data, 3,
                      w_s.d_data, work_s, lwork_s, devInfo);
  dpct::lapack::hegvd(*handle, 1, oneapi::mkl::job::vec,
                      oneapi::mkl::uplo::upper, 3, a_d.d_data, 3, b_d.d_data, 3,
                      w_d.d_data, work_d, lwork_d, devInfo);

  a_s.D2H();
  a_d.D2H();
  b_s.D2H();
  b_d.D2H();
  w_s.D2H();
  w_d.D2H();

  q_ct1.wait();

  handle = nullptr;
  dpct::dpct_free(work_s);
  dpct::dpct_free(work_d);
  dpct::dpct_free(devInfo);

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

struct Ptr_Data {
  int group_num;
  void** h_data;
  void** d_data;
  Ptr_Data(int group_num) : group_num(group_num) {
    h_data = (void**)malloc(group_num * sizeof(void*));
    memset(h_data, 0, group_num * sizeof(void*));
    d_data = (void **)dpct::dpct_malloc(group_num * sizeof(void *));
    dpct::dpct_memset(d_data, 0, group_num * sizeof(void *));
  }
  ~Ptr_Data() {
    free(h_data);
    dpct::dpct_free(d_data);
  }
  void H2D() {
    dpct::dpct_memcpy(d_data, h_data, group_num * sizeof(void *),
                      dpct::host_to_device);
  }
};

#ifndef DPCT_USM_LEVEL_NONE
void test_cusolverDnTpotrfBatched() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::vector<float> a = {2, -1, 0, -1, 2, -1, 0, -1, 2,
                          2, -1, 0, -1, 2, -1, 0, -1, 2};
  Data<float> a_s(a.data(), 18);
  Data<double> a_d(a.data(), 18);
  Data<sycl::float2> a_c(a.data(), 18);
  Data<sycl::double2> a_z(a.data(), 18);

  Ptr_Data a_s_ptrs(2); a_s_ptrs.h_data[0] = a_s.d_data; a_s_ptrs.h_data[1] = a_s.d_data + 9;
  Ptr_Data a_d_ptrs(2); a_d_ptrs.h_data[0] = a_d.d_data; a_d_ptrs.h_data[1] = a_d.d_data + 9;
  Ptr_Data a_c_ptrs(2); a_c_ptrs.h_data[0] = a_c.d_data; a_c_ptrs.h_data[1] = a_c.d_data + 9;
  Ptr_Data a_z_ptrs(2); a_z_ptrs.h_data[0] = a_z.d_data; a_z_ptrs.h_data[1] = a_z.d_data + 9;

  sycl::queue *handle;
  handle = &q_ct1;

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();

  a_s_ptrs.H2D();
  a_d_ptrs.H2D();
  a_c_ptrs.H2D();
  a_z_ptrs.H2D();

  int *infoArray;
  infoArray = (int *)dpct::dpct_malloc(2 * sizeof(int));

  dpct::lapack::potrf_batch(*handle, oneapi::mkl::uplo::upper, 3,
                            (float **)a_s_ptrs.d_data, 3, infoArray, 2);
  dpct::lapack::potrf_batch(*handle, oneapi::mkl::uplo::upper, 3,
                            (double **)a_d_ptrs.d_data, 3, infoArray, 2);
  dpct::lapack::potrf_batch(*handle, oneapi::mkl::uplo::upper, 3,
                            (sycl::float2 **)a_c_ptrs.d_data, 3, infoArray, 2);
  dpct::lapack::potrf_batch(*handle, oneapi::mkl::uplo::upper, 3,
                            (sycl::double2 **)a_z_ptrs.d_data, 3, infoArray, 2);

  a_s.D2H();
  a_d.D2H();
  a_c.D2H();
  a_z.D2H();

  q_ct1.wait();

  handle = nullptr;

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
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::vector<float> a = {1.414214,-0.707107,0.000000,-0.707107,1.224745,-0.816497,0.000000,-0.816497,1.154701,
                          1.414214,-0.707107,0.000000,-0.707107,1.224745,-0.816497,0.000000,-0.816497,1.154701 };
  Data<float> a_s(a.data(), 18);
  Data<double> a_d(a.data(), 18);
  Data<sycl::float2> a_c(a.data(), 18);
  Data<sycl::double2> a_z(a.data(), 18);

  Ptr_Data a_s_ptrs(2); a_s_ptrs.h_data[0] = a_s.d_data; a_s_ptrs.h_data[1] = a_s.d_data + 9;
  Ptr_Data a_d_ptrs(2); a_d_ptrs.h_data[0] = a_d.d_data; a_d_ptrs.h_data[1] = a_d.d_data + 9;
  Ptr_Data a_c_ptrs(2); a_c_ptrs.h_data[0] = a_c.d_data; a_c_ptrs.h_data[1] = a_c.d_data + 9;
  Ptr_Data a_z_ptrs(2); a_z_ptrs.h_data[0] = a_z.d_data; a_z_ptrs.h_data[1] = a_z.d_data + 9;

  std::vector<float> b = {0, 0, 4,
                          0, 0, 4};
  Data<float> b_s(b.data(), 6);
  Data<double> b_d(b.data(), 6);
  Data<sycl::float2> b_c(b.data(), 6);
  Data<sycl::double2> b_z(b.data(), 6);

  Ptr_Data b_s_ptrs(2); b_s_ptrs.h_data[0] = b_s.d_data; b_s_ptrs.h_data[1] = b_s.d_data + 3;
  Ptr_Data b_d_ptrs(2); b_d_ptrs.h_data[0] = b_d.d_data; b_d_ptrs.h_data[1] = b_d.d_data + 3;
  Ptr_Data b_c_ptrs(2); b_c_ptrs.h_data[0] = b_c.d_data; b_c_ptrs.h_data[1] = b_c.d_data + 3;
  Ptr_Data b_z_ptrs(2); b_z_ptrs.h_data[0] = b_z.d_data; b_z_ptrs.h_data[1] = b_z.d_data + 3;

  sycl::queue *handle;
  handle = &q_ct1;

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
  infoArray = (int *)dpct::dpct_malloc(2 * sizeof(int));

  dpct::lapack::potrs_batch(*handle, oneapi::mkl::uplo::upper, 3, 1,
                            (float **)a_s_ptrs.d_data, 3,
                            (float **)b_s_ptrs.d_data, 3, infoArray, 2);
  dpct::lapack::potrs_batch(*handle, oneapi::mkl::uplo::upper, 3, 1,
                            (double **)a_d_ptrs.d_data, 3,
                            (double **)b_d_ptrs.d_data, 3, infoArray, 2);
  dpct::lapack::potrs_batch(*handle, oneapi::mkl::uplo::upper, 3, 1,
                            (sycl::float2 **)a_c_ptrs.d_data, 3,
                            (sycl::float2 **)b_c_ptrs.d_data, 3, infoArray, 2);
  dpct::lapack::potrs_batch(*handle, oneapi::mkl::uplo::upper, 3, 1,
                            (sycl::double2 **)a_z_ptrs.d_data, 3,
                            (sycl::double2 **)b_z_ptrs.d_data, 3, infoArray, 2);

  a_s.D2H();
  a_d.D2H();
  a_c.D2H();
  a_z.D2H();

  b_s.D2H();
  b_d.D2H();
  b_c.D2H();
  b_z.D2H();

  q_ct1.wait();

  handle = nullptr;

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

void test_helper() {
  sycl::queue *handle;
  handle = &dpct::get_default_queue();
  dpct::queue_ptr stream;
  stream = handle;
  handle = stream;
}

void test_cusolverDnTgesvdj() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::vector<float> a = {1, 2, 3, 4};
  Data<float> a_s(a.data(), 4);
  Data<double> a_d(a.data(), 4);
  Data<sycl::float2> a_c(a.data(), 4);
  Data<sycl::double2> a_z(a.data(), 4);

  Data<float> s_s(2);
  Data<double> s_d(2);
  Data<float> s_c(2);
  Data<double> s_z(2);

  Data<float> u_s(4);
  Data<double> u_d(4);
  Data<sycl::float2> u_c(4);
  Data<sycl::double2> u_z(4);

  Data<float> vt_s(4);
  Data<double> vt_d(4);
  Data<sycl::float2> vt_c(4);
  Data<sycl::double2> vt_z(4);

  Data<float> rwork_s(1);
  Data<double> rwork_d(1);
  Data<float> rwork_c(1);
  Data<double> rwork_z(1);

  sycl::queue *handle;
  handle = &q_ct1;

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();

  int device_ws_size_s;
  int device_ws_size_d;
  int device_ws_size_c;
  int device_ws_size_z;

  int gesvdjinfo;
  /*
  DPCT1026:0: The call to cusolverDnCreateGesvdjInfo was removed because this
  call is redundant in SYCL.
  */

  dpct::lapack::gesvd_scratchpad_size(
      *handle, oneapi::mkl::job::vec, 0, 2, 2, dpct::library_data_t::real_float,
      2, dpct::library_data_t::real_float, 2, dpct::library_data_t::real_float,
      2, &device_ws_size_s);
  dpct::lapack::gesvd_scratchpad_size(
      *handle, oneapi::mkl::job::vec, 0, 2, 2,
      dpct::library_data_t::real_double, 2, dpct::library_data_t::real_double,
      2, dpct::library_data_t::real_double, 2, &device_ws_size_d);
  dpct::lapack::gesvd_scratchpad_size(*handle, oneapi::mkl::job::vec, 0, 2, 2,
                                      dpct::library_data_t::complex_float, 2,
                                      dpct::library_data_t::complex_float, 2,
                                      dpct::library_data_t::complex_float, 2,
                                      &device_ws_size_c);
  dpct::lapack::gesvd_scratchpad_size(*handle, oneapi::mkl::job::vec, 0, 2, 2,
                                      dpct::library_data_t::complex_double, 2,
                                      dpct::library_data_t::complex_double, 2,
                                      dpct::library_data_t::complex_double, 2,
                                      &device_ws_size_z);

  void* device_ws_s;
  void* device_ws_d;
  void* device_ws_c;
  void* device_ws_z;
  device_ws_s = dpct::dpct_malloc(device_ws_size_s * sizeof(float));
  device_ws_d = dpct::dpct_malloc(device_ws_size_d * sizeof(double));
  device_ws_c = dpct::dpct_malloc(device_ws_size_c * sizeof(sycl::float2));
  device_ws_z = dpct::dpct_malloc(device_ws_size_z * sizeof(sycl::double2));

  int *info;
  info = (int *)dpct::dpct_malloc(sizeof(int));

  dpct::lapack::gesvd(*handle, oneapi::mkl::job::vec, 0, 2, 2,
                      dpct::library_data_t::real_float, (float *)a_s.d_data, 2,
                      dpct::library_data_t::real_float, (float *)s_s.d_data,
                      dpct::library_data_t::real_float, (float *)u_s.d_data, 2,
                      dpct::library_data_t::real_float, (float *)vt_s.d_data, 2,
                      (float *)device_ws_s, device_ws_size_s, info);
  dpct::lapack::gesvd(*handle, oneapi::mkl::job::vec, 0, 2, 2,
                      dpct::library_data_t::real_double, (double *)a_d.d_data,
                      2, dpct::library_data_t::real_double,
                      (double *)s_d.d_data, dpct::library_data_t::real_double,
                      (double *)u_d.d_data, 2,
                      dpct::library_data_t::real_double, (double *)vt_d.d_data,
                      2, (double *)device_ws_d, device_ws_size_d, info);
  dpct::lapack::gesvd(
      *handle, oneapi::mkl::job::vec, 0, 2, 2,
      dpct::library_data_t::complex_float, (sycl::float2 *)a_c.d_data, 2,
      dpct::library_data_t::real_float, (float *)s_c.d_data,
      dpct::library_data_t::complex_float, (sycl::float2 *)u_c.d_data, 2,
      dpct::library_data_t::complex_float, (sycl::float2 *)vt_c.d_data, 2,
      (sycl::float2 *)device_ws_c, device_ws_size_c, info);
  dpct::lapack::gesvd(
      *handle, oneapi::mkl::job::vec, 0, 2, 2,
      dpct::library_data_t::complex_double, (sycl::double2 *)a_z.d_data, 2,
      dpct::library_data_t::real_double, (double *)s_z.d_data,
      dpct::library_data_t::complex_double, (sycl::double2 *)u_z.d_data, 2,
      dpct::library_data_t::complex_double, (sycl::double2 *)vt_z.d_data, 2,
      (sycl::double2 *)device_ws_z, device_ws_size_z, info);

  s_s.D2H();
  s_d.D2H();
  s_c.D2H();
  s_z.D2H();

  u_s.D2H();
  u_d.D2H();
  u_c.D2H();
  u_z.D2H();

  vt_s.D2H();
  vt_d.D2H();
  vt_c.D2H();
  vt_z.D2H();

  q_ct1.wait();

  /*
  DPCT1026:1: The call to cusolverDnDestroyGesvdjInfo was removed because this
  call is redundant in SYCL.
  */
  handle = nullptr;
  dpct::dpct_free(device_ws_s);
  dpct::dpct_free(device_ws_d);
  dpct::dpct_free(device_ws_c);
  dpct::dpct_free(device_ws_z);
  dpct::dpct_free(info);

  float expect_s[2] = {5.464985,0.365966};
  float expect_u[4] = {0.576048,0.817416,-0.817416,0.576048};
  float expect_vt[4] = {0.404554,0.914514,0.914514,-0.404554};

  if (compare_result(expect_s, s_s.h_data, 2) &&
      compare_result(expect_s, s_d.h_data, 2) &&
      compare_result(expect_s, s_c.h_data, 2) &&
      compare_result(expect_s, s_z.h_data, 2) &&
      compare_result(expect_u, u_s.h_data, 4) &&
      compare_result(expect_u, u_d.h_data, 4) &&
      compare_result(expect_u, u_c.h_data, 4) &&
      compare_result(expect_u, u_z.h_data, 4) &&
      compare_result(expect_vt, vt_s.h_data, 4) &&
      compare_result(expect_vt, vt_d.h_data, 4) &&
      compare_result(expect_vt, vt_c.h_data, 4) &&
      compare_result(expect_vt, vt_z.h_data, 4))
    printf("DnTgesvdj pass\n");
  else {
    printf("DnTgesvdj fail\n");
    test_passed = false;
  }
}

void test_cusolverDnXgetrf() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::vector<float> a = {1, 2, 3, 4};
  Data<float> a_s(a.data(), 4);
  Data<double> a_d(a.data(), 4);
  Data<sycl::float2> a_c(a.data(), 4);
  Data<sycl::double2> a_z(a.data(), 4);
  Data<int64_t> ipiv_s(2);
  Data<int64_t> ipiv_d(2);
  Data<int64_t> ipiv_c(2);
  Data<int64_t> ipiv_z(2);

  sycl::queue *handle;
  handle = &q_ct1;

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();
  ipiv_s.H2D();
  ipiv_d.H2D();
  ipiv_c.H2D();
  ipiv_z.H2D();

  size_t device_ws_size_s;
  size_t device_ws_size_d;
  size_t device_ws_size_c;
  size_t device_ws_size_z;
  size_t host_ws_size_s;
  size_t host_ws_size_d;
  size_t host_ws_size_c;
  size_t host_ws_size_z;

  int params;
  /*
  DPCT1026:0: The call to cusolverDnCreateParams was removed because this call
  is redundant in SYCL.
  */

  dpct::lapack::getrf_scratchpad_size(*handle, 2, 2,
                                      dpct::library_data_t::real_float, 2,
                                      &device_ws_size_s, &host_ws_size_s);
  dpct::lapack::getrf_scratchpad_size(*handle, 2, 2,
                                      dpct::library_data_t::real_double, 2,
                                      &device_ws_size_d, &host_ws_size_d);
  dpct::lapack::getrf_scratchpad_size(*handle, 2, 2,
                                      dpct::library_data_t::complex_float, 2,
                                      &device_ws_size_c, &host_ws_size_c);
  dpct::lapack::getrf_scratchpad_size(*handle, 2, 2,
                                      dpct::library_data_t::complex_double, 2,
                                      &device_ws_size_z, &host_ws_size_z);

  void* device_ws_s;
  void* device_ws_d;
  void* device_ws_c;
  void* device_ws_z;
  void* host_ws_s;
  void* host_ws_d;
  void* host_ws_c;
  void* host_ws_z;
  device_ws_s = dpct::dpct_malloc(device_ws_size_s);
  device_ws_d = dpct::dpct_malloc(device_ws_size_d);
  device_ws_c = dpct::dpct_malloc(device_ws_size_c);
  device_ws_z = dpct::dpct_malloc(device_ws_size_z);
  host_ws_s = dpct::dpct_malloc(host_ws_size_s);
  host_ws_d = dpct::dpct_malloc(host_ws_size_d);
  host_ws_c = dpct::dpct_malloc(host_ws_size_c);
  host_ws_z = dpct::dpct_malloc(host_ws_size_z);

  int *info;
  info = (int *)dpct::dpct_malloc(sizeof(int));

  /*
  DPCT1047:1: The meaning of ipiv_s.d_data in the dpct::lapack::getrf is
  different from the cusolverDnXgetrf. You may need to check the migrated code.
  */
  dpct::lapack::getrf(*handle, 2, 2, dpct::library_data_t::real_float,
                      a_s.d_data, 2, ipiv_s.d_data, device_ws_s,
                      device_ws_size_s, info);
  /*
  DPCT1047:2: The meaning of ipiv_d.d_data in the dpct::lapack::getrf is
  different from the cusolverDnXgetrf. You may need to check the migrated code.
  */
  dpct::lapack::getrf(*handle, 2, 2, dpct::library_data_t::real_double,
                      a_d.d_data, 2, ipiv_d.d_data, device_ws_d,
                      device_ws_size_d, info);
  /*
  DPCT1047:3: The meaning of ipiv_c.d_data in the dpct::lapack::getrf is
  different from the cusolverDnXgetrf. You may need to check the migrated code.
  */
  dpct::lapack::getrf(*handle, 2, 2, dpct::library_data_t::complex_float,
                      a_c.d_data, 2, ipiv_c.d_data, device_ws_c,
                      device_ws_size_c, info);
  /*
  DPCT1047:4: The meaning of ipiv_z.d_data in the dpct::lapack::getrf is
  different from the cusolverDnXgetrf. You may need to check the migrated code.
  */
  dpct::lapack::getrf(*handle, 2, 2, dpct::library_data_t::complex_double,
                      a_z.d_data, 2, ipiv_z.d_data, device_ws_z,
                      device_ws_size_z, info);

  a_s.D2H();
  a_d.D2H();
  a_c.D2H();
  a_z.D2H();
  ipiv_s.D2H();
  ipiv_d.D2H();
  ipiv_c.D2H();
  ipiv_z.D2H();

  q_ct1.wait();

  /*
  DPCT1026:5: The call to cusolverDnDestroyParams was removed because this call
  is redundant in SYCL.
  */
  handle = nullptr;
  dpct::dpct_free(device_ws_s);
  dpct::dpct_free(device_ws_d);
  dpct::dpct_free(device_ws_c);
  dpct::dpct_free(device_ws_z);
  dpct::dpct_free(host_ws_s);
  dpct::dpct_free(host_ws_d);
  dpct::dpct_free(host_ws_c);
  dpct::dpct_free(host_ws_z);
  dpct::dpct_free(info);

  float expect_a[4] = {2, 0.5, 4, 1};
  float expect_ipiv[2] = {2, 2};
  if (compare_result(expect_a, a_s.h_data, 4) &&
      compare_result(expect_a, a_d.h_data, 4) &&
      compare_result(expect_a, a_c.h_data, 4) &&
      compare_result(expect_a, a_z.h_data, 4) &&
      compare_result(expect_ipiv, ipiv_s.h_data, 2) &&
      compare_result(expect_ipiv, ipiv_d.h_data, 2) &&
      compare_result(expect_ipiv, ipiv_c.h_data, 2) &&
      compare_result(expect_ipiv, ipiv_z.h_data, 2))
    printf("DnXgetrf pass\n");
  else {
    printf("DnXgetrf fail\n");
    test_passed = false;
  }
}

void test_cusolverDnXgetrfnp() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::vector<float> a = {1, 2, 3, 4};
  Data<float> a_s(a.data(), 4);
  Data<double> a_d(a.data(), 4);
  Data<sycl::float2> a_c(a.data(), 4);
  Data<sycl::double2> a_z(a.data(), 4);

  sycl::queue *handle;
  handle = &q_ct1;

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();

  size_t device_ws_size_s;
  size_t device_ws_size_d;
  size_t device_ws_size_c;
  size_t device_ws_size_z;
  size_t host_ws_size_s;
  size_t host_ws_size_d;
  size_t host_ws_size_c;
  size_t host_ws_size_z;

  int params;
  /*
  DPCT1026:6: The call to cusolverDnCreateParams was removed because this call
  is redundant in SYCL.
  */

  dpct::lapack::getrf_scratchpad_size(*handle, 2, 2,
                                      dpct::library_data_t::real_float, 2,
                                      &device_ws_size_s, &host_ws_size_s);
  dpct::lapack::getrf_scratchpad_size(*handle, 2, 2,
                                      dpct::library_data_t::real_double, 2,
                                      &device_ws_size_d, &host_ws_size_d);
  dpct::lapack::getrf_scratchpad_size(*handle, 2, 2,
                                      dpct::library_data_t::complex_float, 2,
                                      &device_ws_size_c, &host_ws_size_c);
  dpct::lapack::getrf_scratchpad_size(*handle, 2, 2,
                                      dpct::library_data_t::complex_double, 2,
                                      &device_ws_size_z, &host_ws_size_z);

  void* device_ws_s;
  void* device_ws_d;
  void* device_ws_c;
  void* device_ws_z;
  void* host_ws_s;
  void* host_ws_d;
  void* host_ws_c;
  void* host_ws_z;
  device_ws_s = dpct::dpct_malloc(device_ws_size_s);
  device_ws_d = dpct::dpct_malloc(device_ws_size_d);
  device_ws_c = dpct::dpct_malloc(device_ws_size_c);
  device_ws_z = dpct::dpct_malloc(device_ws_size_z);
  host_ws_s = dpct::dpct_malloc(host_ws_size_s);
  host_ws_d = dpct::dpct_malloc(host_ws_size_d);
  host_ws_c = dpct::dpct_malloc(host_ws_size_c);
  host_ws_z = dpct::dpct_malloc(host_ws_size_z);

  int *info;
  info = (int *)dpct::dpct_malloc(sizeof(int));

  /*
  DPCT1047:7: The meaning of nullptr in the dpct::lapack::getrf is different
  from the cusolverDnXgetrf. You may need to check the migrated code.
  */
  dpct::lapack::getrf(*handle, 2, 2, dpct::library_data_t::real_float,
                      a_s.d_data, 2, nullptr, device_ws_s, device_ws_size_s,
                      info);
  /*
  DPCT1047:8: The meaning of nullptr in the dpct::lapack::getrf is different
  from the cusolverDnXgetrf. You may need to check the migrated code.
  */
  dpct::lapack::getrf(*handle, 2, 2, dpct::library_data_t::real_double,
                      a_d.d_data, 2, nullptr, device_ws_d, device_ws_size_d,
                      info);
  /*
  DPCT1047:9: The meaning of nullptr in the dpct::lapack::getrf is different
  from the cusolverDnXgetrf. You may need to check the migrated code.
  */
  dpct::lapack::getrf(*handle, 2, 2, dpct::library_data_t::complex_float,
                      a_c.d_data, 2, nullptr, device_ws_c, device_ws_size_c,
                      info);
  /*
  DPCT1047:10: The meaning of nullptr in the dpct::lapack::getrf is different
  from the cusolverDnXgetrf. You may need to check the migrated code.
  */
  dpct::lapack::getrf(*handle, 2, 2, dpct::library_data_t::complex_double,
                      a_z.d_data, 2, nullptr, device_ws_z, device_ws_size_z,
                      info);

  a_s.D2H();
  a_d.D2H();
  a_c.D2H();
  a_z.D2H();

  q_ct1.wait();

  /*
  DPCT1026:11: The call to cusolverDnDestroyParams was removed because this call
  is redundant in SYCL.
  */
  handle = nullptr;
  dpct::dpct_free(device_ws_s);
  dpct::dpct_free(device_ws_d);
  dpct::dpct_free(device_ws_c);
  dpct::dpct_free(device_ws_z);
  dpct::dpct_free(host_ws_s);
  dpct::dpct_free(host_ws_d);
  dpct::dpct_free(host_ws_c);
  dpct::dpct_free(host_ws_z);
  dpct::dpct_free(info);

  float expect_a[4] = {1, 2, 3, -2};
  if (compare_result(expect_a, a_s.h_data, 4) &&
      compare_result(expect_a, a_d.h_data, 4) &&
      compare_result(expect_a, a_c.h_data, 4) &&
      compare_result(expect_a, a_z.h_data, 4))
    printf("DnXgetrfnp pass\n");
  else {
    printf("DnXgetrfnp fail\n");
    test_passed = false;
  }
}

void test_cusolverDnGetrf() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::vector<float> a = {1, 2, 3, 4};
  Data<float> a_s(a.data(), 4);
  Data<double> a_d(a.data(), 4);
  Data<sycl::float2> a_c(a.data(), 4);
  Data<sycl::double2> a_z(a.data(), 4);
  Data<int64_t> ipiv_s(2);
  Data<int64_t> ipiv_d(2);
  Data<int64_t> ipiv_c(2);
  Data<int64_t> ipiv_z(2);

  sycl::queue *handle;
  handle = &q_ct1;

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();
  ipiv_s.H2D();
  ipiv_d.H2D();
  ipiv_c.H2D();
  ipiv_z.H2D();

  size_t device_ws_size_s;
  size_t device_ws_size_d;
  size_t device_ws_size_c;
  size_t device_ws_size_z;

  int params;
  /*
  DPCT1026:12: The call to cusolverDnCreateParams was removed because this call
  is redundant in SYCL.
  */

  dpct::lapack::getrf_scratchpad_size(
      *handle, 2, 2, dpct::library_data_t::real_float, 2, &device_ws_size_s);
  dpct::lapack::getrf_scratchpad_size(
      *handle, 2, 2, dpct::library_data_t::real_double, 2, &device_ws_size_d);
  dpct::lapack::getrf_scratchpad_size(
      *handle, 2, 2, dpct::library_data_t::complex_float, 2, &device_ws_size_c);
  dpct::lapack::getrf_scratchpad_size(*handle, 2, 2,
                                      dpct::library_data_t::complex_double, 2,
                                      &device_ws_size_z);

  void* device_ws_s;
  void* device_ws_d;
  void* device_ws_c;
  void* device_ws_z;

  device_ws_s = dpct::dpct_malloc(device_ws_size_s);
  device_ws_d = dpct::dpct_malloc(device_ws_size_d);
  device_ws_c = dpct::dpct_malloc(device_ws_size_c);
  device_ws_z = dpct::dpct_malloc(device_ws_size_z);

  int *info;
  info = (int *)dpct::dpct_malloc(sizeof(int));

  /*
  DPCT1047:13: The meaning of ipiv_s.d_data in the dpct::lapack::getrf is
  different from the cusolverDnGetrf. You may need to check the migrated code.
  */
  dpct::lapack::getrf(*handle, 2, 2, dpct::library_data_t::real_float,
                      a_s.d_data, 2, ipiv_s.d_data, device_ws_s,
                      device_ws_size_s, info);
  /*
  DPCT1047:14: The meaning of ipiv_d.d_data in the dpct::lapack::getrf is
  different from the cusolverDnGetrf. You may need to check the migrated code.
  */
  dpct::lapack::getrf(*handle, 2, 2, dpct::library_data_t::real_double,
                      a_d.d_data, 2, ipiv_d.d_data, device_ws_d,
                      device_ws_size_d, info);
  /*
  DPCT1047:15: The meaning of ipiv_c.d_data in the dpct::lapack::getrf is
  different from the cusolverDnGetrf. You may need to check the migrated code.
  */
  dpct::lapack::getrf(*handle, 2, 2, dpct::library_data_t::complex_float,
                      a_c.d_data, 2, ipiv_c.d_data, device_ws_c,
                      device_ws_size_c, info);
  /*
  DPCT1047:16: The meaning of ipiv_z.d_data in the dpct::lapack::getrf is
  different from the cusolverDnGetrf. You may need to check the migrated code.
  */
  dpct::lapack::getrf(*handle, 2, 2, dpct::library_data_t::complex_double,
                      a_z.d_data, 2, ipiv_z.d_data, device_ws_z,
                      device_ws_size_z, info);

  a_s.D2H();
  a_d.D2H();
  a_c.D2H();
  a_z.D2H();
  ipiv_s.D2H();
  ipiv_d.D2H();
  ipiv_c.D2H();
  ipiv_z.D2H();

  q_ct1.wait();

  /*
  DPCT1026:17: The call to cusolverDnDestroyParams was removed because this call
  is redundant in SYCL.
  */
  handle = nullptr;
  dpct::dpct_free(device_ws_s);
  dpct::dpct_free(device_ws_d);
  dpct::dpct_free(device_ws_c);
  dpct::dpct_free(device_ws_z);
  dpct::dpct_free(info);

  float expect_a[4] = {2, 0.5, 4, 1};
  float expect_ipiv[2] = {2, 2};
  if (compare_result(expect_a, a_s.h_data, 4) &&
      compare_result(expect_a, a_d.h_data, 4) &&
      compare_result(expect_a, a_c.h_data, 4) &&
      compare_result(expect_a, a_z.h_data, 4) &&
      compare_result(expect_ipiv, ipiv_s.h_data, 2) &&
      compare_result(expect_ipiv, ipiv_d.h_data, 2) &&
      compare_result(expect_ipiv, ipiv_c.h_data, 2) &&
      compare_result(expect_ipiv, ipiv_z.h_data, 2))
    printf("DnGetrf pass\n");
  else {
    printf("DnGetrf fail\n");
    test_passed = false;
  }
}

void test_cusolverDnXgetrs() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::vector<float> a = {2, 0.5, 4, 1};
  std::vector<float> ipiv = {2, 2};
  std::vector<float> b = {23, 34, 31, 46, 39, 58};
  Data<float> a_s(a.data(), 4);
  Data<double> a_d(a.data(), 4);
  Data<sycl::float2> a_c(a.data(), 4);
  Data<sycl::double2> a_z(a.data(), 4);
  Data<int64_t> ipiv_s(ipiv.data(), 2);
  Data<int64_t> ipiv_d(ipiv.data(), 2);
  Data<int64_t> ipiv_c(ipiv.data(), 2);
  Data<int64_t> ipiv_z(ipiv.data(), 2);
  Data<float> b_s(b.data(), 6);
  Data<double> b_d(b.data(), 6);
  Data<sycl::float2> b_c(b.data(), 6);
  Data<sycl::double2> b_z(b.data(), 6);

  sycl::queue *handle;
  handle = &q_ct1;

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();
  ipiv_s.H2D();
  ipiv_d.H2D();
  ipiv_c.H2D();
  ipiv_z.H2D();
  b_s.H2D();
  b_d.H2D();
  b_c.H2D();
  b_z.H2D();

  int params;
  /*
  DPCT1026:18: The call to cusolverDnCreateParams was removed because this call
  is redundant in SYCL.
  */

  int *info;
  info = (int *)dpct::dpct_malloc(sizeof(int));

  dpct::lapack::getrs(*handle, oneapi::mkl::transpose::nontrans, 2, 3,
                      dpct::library_data_t::real_float, a_s.d_data, 2,
                      ipiv_s.d_data, dpct::library_data_t::real_float,
                      b_s.d_data, 2, info);
  dpct::lapack::getrs(*handle, oneapi::mkl::transpose::nontrans, 2, 3,
                      dpct::library_data_t::real_double, a_d.d_data, 2,
                      ipiv_d.d_data, dpct::library_data_t::real_double,
                      b_d.d_data, 2, info);
  dpct::lapack::getrs(*handle, oneapi::mkl::transpose::nontrans, 2, 3,
                      dpct::library_data_t::complex_float, a_c.d_data, 2,
                      ipiv_c.d_data, dpct::library_data_t::complex_float,
                      b_c.d_data, 2, info);
  dpct::lapack::getrs(*handle, oneapi::mkl::transpose::nontrans, 2, 3,
                      dpct::library_data_t::complex_double, a_z.d_data, 2,
                      ipiv_z.d_data, dpct::library_data_t::complex_double,
                      b_z.d_data, 2, info);

  b_s.D2H();
  b_d.D2H();
  b_c.D2H();
  b_z.D2H();

  q_ct1.wait();

  /*
  DPCT1026:19: The call to cusolverDnDestroyParams was removed because this call
  is redundant in SYCL.
  */
  handle = nullptr;
  dpct::dpct_free(info);

  float expect_b[6] = {5, 6, 7, 8, 9, 10};
  if (compare_result(expect_b, b_s.h_data, 6) &&
      compare_result(expect_b, b_d.h_data, 6) &&
      compare_result(expect_b, b_c.h_data, 6) &&
      compare_result(expect_b, b_z.h_data, 6))
    printf("DnXgetrs pass\n");
  else {
    printf("DnXgetrs fail\n");
    test_passed = false;
  }
}

void test_cusolverDnGetrs() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::vector<float> a = {2, 0.5, 4, 1};
  std::vector<float> ipiv = {2, 2};
  std::vector<float> b = {23, 34, 31, 46, 39, 58};
  Data<float> a_s(a.data(), 4);
  Data<double> a_d(a.data(), 4);
  Data<sycl::float2> a_c(a.data(), 4);
  Data<sycl::double2> a_z(a.data(), 4);
  Data<int64_t> ipiv_s(ipiv.data(), 2);
  Data<int64_t> ipiv_d(ipiv.data(), 2);
  Data<int64_t> ipiv_c(ipiv.data(), 2);
  Data<int64_t> ipiv_z(ipiv.data(), 2);
  Data<float> b_s(b.data(), 6);
  Data<double> b_d(b.data(), 6);
  Data<sycl::float2> b_c(b.data(), 6);
  Data<sycl::double2> b_z(b.data(), 6);

  sycl::queue *handle;
  handle = &q_ct1;

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();
  ipiv_s.H2D();
  ipiv_d.H2D();
  ipiv_c.H2D();
  ipiv_z.H2D();
  b_s.H2D();
  b_d.H2D();
  b_c.H2D();
  b_z.H2D();

  int params;
  /*
  DPCT1026:20: The call to cusolverDnCreateParams was removed because this call
  is redundant in SYCL.
  */

  int *info;
  info = (int *)dpct::dpct_malloc(sizeof(int));

  dpct::lapack::getrs(*handle, oneapi::mkl::transpose::nontrans, 2, 3,
                      dpct::library_data_t::real_float, a_s.d_data, 2,
                      ipiv_s.d_data, dpct::library_data_t::real_float,
                      b_s.d_data, 2, info);
  dpct::lapack::getrs(*handle, oneapi::mkl::transpose::nontrans, 2, 3,
                      dpct::library_data_t::real_double, a_d.d_data, 2,
                      ipiv_d.d_data, dpct::library_data_t::real_double,
                      b_d.d_data, 2, info);
  dpct::lapack::getrs(*handle, oneapi::mkl::transpose::nontrans, 2, 3,
                      dpct::library_data_t::complex_float, a_c.d_data, 2,
                      ipiv_c.d_data, dpct::library_data_t::complex_float,
                      b_c.d_data, 2, info);
  dpct::lapack::getrs(*handle, oneapi::mkl::transpose::nontrans, 2, 3,
                      dpct::library_data_t::complex_double, a_z.d_data, 2,
                      ipiv_z.d_data, dpct::library_data_t::complex_double,
                      b_z.d_data, 2, info);

  b_s.D2H();
  b_d.D2H();
  b_c.D2H();
  b_z.D2H();

  q_ct1.wait();

  /*
  DPCT1026:21: The call to cusolverDnDestroyParams was removed because this call
  is redundant in SYCL.
  */
  handle = nullptr;
  dpct::dpct_free(info);

  float expect_b[6] = {5, 6, 7, 8, 9, 10};
  if (compare_result(expect_b, b_s.h_data, 6) &&
      compare_result(expect_b, b_d.h_data, 6) &&
      compare_result(expect_b, b_c.h_data, 6) &&
      compare_result(expect_b, b_z.h_data, 6))
    printf("DnGetrs pass\n");
  else {
    printf("DnGetrs fail\n");
    test_passed = false;
  }
}

void test_cusolverDnXgeqrf() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::vector<float> a = {1, 2, 3, 4};
  Data<float> a_s(a.data(), 4);
  Data<double> a_d(a.data(), 4);
  Data<sycl::float2> a_c(a.data(), 4);
  Data<sycl::double2> a_z(a.data(), 4);
  Data<float> tau_s(2);
  Data<double> tau_d(2);
  Data<sycl::float2> tau_c(2);
  Data<sycl::double2> tau_z(2);

  sycl::queue *handle;
  handle = &q_ct1;

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();
  tau_s.H2D();
  tau_d.H2D();
  tau_c.H2D();
  tau_z.H2D();

  size_t device_ws_size_s;
  size_t device_ws_size_d;
  size_t device_ws_size_c;
  size_t device_ws_size_z;
  size_t host_ws_size_s;
  size_t host_ws_size_d;
  size_t host_ws_size_c;
  size_t host_ws_size_z;

  int params;
  /*
  DPCT1026:22: The call to cusolverDnCreateParams was removed because this call
  is redundant in SYCL.
  */

  dpct::lapack::geqrf_scratchpad_size(*handle, 2, 2,
                                      dpct::library_data_t::real_float, 2,
                                      &device_ws_size_s, &host_ws_size_s);
  dpct::lapack::geqrf_scratchpad_size(*handle, 2, 2,
                                      dpct::library_data_t::real_double, 2,
                                      &device_ws_size_d, &host_ws_size_d);
  dpct::lapack::geqrf_scratchpad_size(*handle, 2, 2,
                                      dpct::library_data_t::complex_float, 2,
                                      &device_ws_size_c, &host_ws_size_c);
  dpct::lapack::geqrf_scratchpad_size(*handle, 2, 2,
                                      dpct::library_data_t::complex_double, 2,
                                      &device_ws_size_z, &host_ws_size_z);

  void* device_ws_s;
  void* device_ws_d;
  void* device_ws_c;
  void* device_ws_z;
  void* host_ws_s;
  void* host_ws_d;
  void* host_ws_c;
  void* host_ws_z;
  device_ws_s = dpct::dpct_malloc(device_ws_size_s);
  device_ws_d = dpct::dpct_malloc(device_ws_size_d);
  device_ws_c = dpct::dpct_malloc(device_ws_size_c);
  device_ws_z = dpct::dpct_malloc(device_ws_size_z);
  host_ws_s = dpct::dpct_malloc(host_ws_size_s);
  host_ws_d = dpct::dpct_malloc(host_ws_size_d);
  host_ws_c = dpct::dpct_malloc(host_ws_size_c);
  host_ws_z = dpct::dpct_malloc(host_ws_size_z);

  int *info;
  info = (int *)dpct::dpct_malloc(sizeof(int));

  dpct::lapack::geqrf(*handle, 2, 2, dpct::library_data_t::real_float,
                      a_s.d_data, 2, dpct::library_data_t::real_float,
                      tau_s.d_data, device_ws_s, device_ws_size_s, info);
  dpct::lapack::geqrf(*handle, 2, 2, dpct::library_data_t::real_double,
                      a_d.d_data, 2, dpct::library_data_t::real_double,
                      tau_d.d_data, device_ws_d, device_ws_size_d, info);
  dpct::lapack::geqrf(*handle, 2, 2, dpct::library_data_t::complex_float,
                      a_c.d_data, 2, dpct::library_data_t::complex_float,
                      tau_c.d_data, device_ws_c, device_ws_size_c, info);
  dpct::lapack::geqrf(*handle, 2, 2, dpct::library_data_t::complex_double,
                      a_z.d_data, 2, dpct::library_data_t::complex_double,
                      tau_z.d_data, device_ws_z, device_ws_size_z, info);

  a_s.D2H();
  a_d.D2H();
  a_c.D2H();
  a_z.D2H();
  tau_s.D2H();
  tau_d.D2H();
  tau_c.D2H();
  tau_z.D2H();

  q_ct1.wait();

  /*
  DPCT1026:23: The call to cusolverDnDestroyParams was removed because this call
  is redundant in SYCL.
  */
  handle = nullptr;
  dpct::dpct_free(device_ws_s);
  dpct::dpct_free(device_ws_d);
  dpct::dpct_free(device_ws_c);
  dpct::dpct_free(device_ws_z);
  dpct::dpct_free(host_ws_s);
  dpct::dpct_free(host_ws_d);
  dpct::dpct_free(host_ws_c);
  dpct::dpct_free(host_ws_z);
  dpct::dpct_free(info);

  float expect_a[4] = {-2.236068, 0.618034, -4.919349, -0.894427};
  float expect_tau[2] = {1.447214, 0};

  if (compare_result(expect_a, a_s.h_data, 4) &&
      compare_result(expect_a, a_d.h_data, 4) &&
      compare_result(expect_a, a_c.h_data, 4) &&
      compare_result(expect_a, a_z.h_data, 4) &&
      compare_result(expect_tau, tau_s.h_data, 2) &&
      compare_result(expect_tau, tau_d.h_data, 2) &&
      compare_result(expect_tau, tau_c.h_data, 2) &&
      compare_result(expect_tau, tau_z.h_data, 2))
    printf("DnXgeqrf pass\n");
  else {
    printf("DnXgeqrf fail\n");
    test_passed = false;
  }
}

void test_cusolverDnGeqrf() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::vector<float> a = {1, 2, 3, 4};
  Data<float> a_s(a.data(), 4);
  Data<double> a_d(a.data(), 4);
  Data<sycl::float2> a_c(a.data(), 4);
  Data<sycl::double2> a_z(a.data(), 4);
  Data<float> tau_s(2);
  Data<double> tau_d(2);
  Data<sycl::float2> tau_c(2);
  Data<sycl::double2> tau_z(2);

  sycl::queue *handle;
  handle = &q_ct1;

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();
  tau_s.H2D();
  tau_d.H2D();
  tau_c.H2D();
  tau_z.H2D();

  size_t device_ws_size_s;
  size_t device_ws_size_d;
  size_t device_ws_size_c;
  size_t device_ws_size_z;

  int params;
  /*
  DPCT1026:24: The call to cusolverDnCreateParams was removed because this call
  is redundant in SYCL.
  */

  dpct::lapack::geqrf_scratchpad_size(
      *handle, 2, 2, dpct::library_data_t::real_float, 2, &device_ws_size_s);
  dpct::lapack::geqrf_scratchpad_size(
      *handle, 2, 2, dpct::library_data_t::real_double, 2, &device_ws_size_d);
  dpct::lapack::geqrf_scratchpad_size(
      *handle, 2, 2, dpct::library_data_t::complex_float, 2, &device_ws_size_c);
  dpct::lapack::geqrf_scratchpad_size(*handle, 2, 2,
                                      dpct::library_data_t::complex_double, 2,
                                      &device_ws_size_z);

  void* device_ws_s;
  void* device_ws_d;
  void* device_ws_c;
  void* device_ws_z;
  device_ws_s = dpct::dpct_malloc(device_ws_size_s);
  device_ws_d = dpct::dpct_malloc(device_ws_size_d);
  device_ws_c = dpct::dpct_malloc(device_ws_size_c);
  device_ws_z = dpct::dpct_malloc(device_ws_size_z);

  int *info;
  info = (int *)dpct::dpct_malloc(sizeof(int));

  dpct::lapack::geqrf(*handle, 2, 2, dpct::library_data_t::real_float,
                      a_s.d_data, 2, dpct::library_data_t::real_float,
                      tau_s.d_data, device_ws_s, device_ws_size_s, info);
  dpct::lapack::geqrf(*handle, 2, 2, dpct::library_data_t::real_double,
                      a_d.d_data, 2, dpct::library_data_t::real_double,
                      tau_d.d_data, device_ws_d, device_ws_size_d, info);
  dpct::lapack::geqrf(*handle, 2, 2, dpct::library_data_t::complex_float,
                      a_c.d_data, 2, dpct::library_data_t::complex_float,
                      tau_c.d_data, device_ws_c, device_ws_size_c, info);
  dpct::lapack::geqrf(*handle, 2, 2, dpct::library_data_t::complex_double,
                      a_z.d_data, 2, dpct::library_data_t::complex_double,
                      tau_z.d_data, device_ws_z, device_ws_size_z, info);

  a_s.D2H();
  a_d.D2H();
  a_c.D2H();
  a_z.D2H();
  tau_s.D2H();
  tau_d.D2H();
  tau_c.D2H();
  tau_z.D2H();

  q_ct1.wait();

  /*
  DPCT1026:25: The call to cusolverDnDestroyParams was removed because this call
  is redundant in SYCL.
  */
  handle = nullptr;
  dpct::dpct_free(device_ws_s);
  dpct::dpct_free(device_ws_d);
  dpct::dpct_free(device_ws_c);
  dpct::dpct_free(device_ws_z);
  dpct::dpct_free(info);

  float expect_a[4] = {-2.236068, 0.618034, -4.919349, -0.894427};
  float expect_tau[2] = {1.447214, 0};

  if (compare_result(expect_a, a_s.h_data, 4) &&
      compare_result(expect_a, a_d.h_data, 4) &&
      compare_result(expect_a, a_c.h_data, 4) &&
      compare_result(expect_a, a_z.h_data, 4) &&
      compare_result(expect_tau, tau_s.h_data, 2) &&
      compare_result(expect_tau, tau_d.h_data, 2) &&
      compare_result(expect_tau, tau_c.h_data, 2) &&
      compare_result(expect_tau, tau_z.h_data, 2))
    printf("DnGeqrf pass\n");
  else {
    printf("DnGeqrf fail\n");
    test_passed = false;
  }
}

void test_cusolverDnXgesvd() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::vector<float> a = {1, 2, 3, 4};
  Data<float> a_s(a.data(), 4);
  Data<double> a_d(a.data(), 4);
  Data<sycl::float2> a_c(a.data(), 4);
  Data<sycl::double2> a_z(a.data(), 4);

  Data<float> s_s(2);
  Data<double> s_d(2);
  Data<float> s_c(2);
  Data<double> s_z(2);

  Data<float> u_s(4);
  Data<double> u_d(4);
  Data<sycl::float2> u_c(4);
  Data<sycl::double2> u_z(4);

  Data<float> vt_s(4);
  Data<double> vt_d(4);
  Data<sycl::float2> vt_c(4);
  Data<sycl::double2> vt_z(4);

  sycl::queue *handle;
  handle = &q_ct1;

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();

  size_t device_ws_size_s;
  size_t device_ws_size_d;
  size_t device_ws_size_c;
  size_t device_ws_size_z;
  size_t host_ws_size_s;
  size_t host_ws_size_d;
  size_t host_ws_size_c;
  size_t host_ws_size_z;

  int params;
  /*
  DPCT1026:26: The call to cusolverDnCreateParams was removed because this call
  is redundant in SYCL.
  */

  dpct::lapack::gesvd_scratchpad_size(
      *handle, 'A', 'A', 2, 2, dpct::library_data_t::real_float, 2,
      dpct::library_data_t::real_float, 2, dpct::library_data_t::real_float, 2,
      &device_ws_size_s, &host_ws_size_s);
  dpct::lapack::gesvd_scratchpad_size(
      *handle, 'A', 'A', 2, 2, dpct::library_data_t::real_double, 2,
      dpct::library_data_t::real_double, 2, dpct::library_data_t::real_double,
      2, &device_ws_size_d, &host_ws_size_d);
  dpct::lapack::gesvd_scratchpad_size(*handle, 'A', 'A', 2, 2,
                                      dpct::library_data_t::complex_float, 2,
                                      dpct::library_data_t::complex_float, 2,
                                      dpct::library_data_t::complex_float, 2,
                                      &device_ws_size_c, &host_ws_size_c);
  dpct::lapack::gesvd_scratchpad_size(*handle, 'A', 'A', 2, 2,
                                      dpct::library_data_t::complex_double, 2,
                                      dpct::library_data_t::complex_double, 2,
                                      dpct::library_data_t::complex_double, 2,
                                      &device_ws_size_z, &host_ws_size_z);

  void* device_ws_s;
  void* device_ws_d;
  void* device_ws_c;
  void* device_ws_z;
  void* host_ws_s;
  void* host_ws_d;
  void* host_ws_c;
  void* host_ws_z;
  device_ws_s = dpct::dpct_malloc(device_ws_size_s);
  device_ws_d = dpct::dpct_malloc(device_ws_size_d);
  device_ws_c = dpct::dpct_malloc(device_ws_size_c);
  device_ws_z = dpct::dpct_malloc(device_ws_size_z);
  host_ws_s = dpct::dpct_malloc(host_ws_size_s);
  host_ws_d = dpct::dpct_malloc(host_ws_size_d);
  host_ws_c = dpct::dpct_malloc(host_ws_size_c);
  host_ws_z = dpct::dpct_malloc(host_ws_size_z);

  int *info;
  info = (int *)dpct::dpct_malloc(sizeof(int));

  dpct::lapack::gesvd(*handle, 'A', 'A', 2, 2, dpct::library_data_t::real_float,
                      a_s.d_data, 2, dpct::library_data_t::real_float,
                      s_s.d_data, dpct::library_data_t::real_float, u_s.d_data,
                      2, dpct::library_data_t::real_float, vt_s.d_data, 2,
                      device_ws_s, device_ws_size_s, info);
  dpct::lapack::gesvd(*handle, 'A', 'A', 2, 2,
                      dpct::library_data_t::real_double, a_d.d_data, 2,
                      dpct::library_data_t::real_double, s_d.d_data,
                      dpct::library_data_t::real_double, u_d.d_data, 2,
                      dpct::library_data_t::real_double, vt_d.d_data, 2,
                      device_ws_d, device_ws_size_d, info);
  dpct::lapack::gesvd(*handle, 'A', 'A', 2, 2,
                      dpct::library_data_t::complex_float, a_c.d_data, 2,
                      dpct::library_data_t::real_float, s_c.d_data,
                      dpct::library_data_t::complex_float, u_c.d_data, 2,
                      dpct::library_data_t::complex_float, vt_c.d_data, 2,
                      device_ws_c, device_ws_size_c, info);
  dpct::lapack::gesvd(*handle, 'A', 'A', 2, 2,
                      dpct::library_data_t::complex_double, a_z.d_data, 2,
                      dpct::library_data_t::real_double, s_z.d_data,
                      dpct::library_data_t::complex_double, u_z.d_data, 2,
                      dpct::library_data_t::complex_double, vt_z.d_data, 2,
                      device_ws_z, device_ws_size_z, info);

  s_s.D2H();
  s_d.D2H();
  s_c.D2H();
  s_z.D2H();

  u_s.D2H();
  u_d.D2H();
  u_c.D2H();
  u_z.D2H();

  vt_s.D2H();
  vt_d.D2H();
  vt_c.D2H();
  vt_z.D2H();

  q_ct1.wait();

  /*
  DPCT1026:27: The call to cusolverDnDestroyParams was removed because this call
  is redundant in SYCL.
  */
  handle = nullptr;
  dpct::dpct_free(device_ws_s);
  dpct::dpct_free(device_ws_d);
  dpct::dpct_free(device_ws_c);
  dpct::dpct_free(device_ws_z);
  dpct::dpct_free(host_ws_s);
  dpct::dpct_free(host_ws_d);
  dpct::dpct_free(host_ws_c);
  dpct::dpct_free(host_ws_z);
  dpct::dpct_free(info);

  float expect_s[2] = {5.464985,0.365966};
  float expect_u[4] = {0.576048,0.817416,-0.817416,0.576048};
  float expect_vt[4] = {0.404554,0.914514,0.914514,-0.404554};

  if (compare_result(expect_s, s_s.h_data, 2) &&
      compare_result(expect_s, s_d.h_data, 2) &&
      compare_result(expect_s, s_c.h_data, 2) &&
      compare_result(expect_s, s_z.h_data, 2) &&
      compare_result(expect_u, u_s.h_data, 4) &&
      compare_result(expect_u, u_d.h_data, 4) &&
      compare_result(expect_u, u_c.h_data, 4) &&
      compare_result(expect_u, u_z.h_data, 4) &&
      compare_result(expect_vt, vt_s.h_data, 4) &&
      compare_result(expect_vt, vt_d.h_data, 4) &&
      compare_result(expect_vt, vt_c.h_data, 4) &&
      compare_result(expect_vt, vt_z.h_data, 4))
    printf("DnXgesvd pass\n");
  else {
    printf("DnXgesvd fail\n");
    test_passed = false;
  }
}

void test_cusolverDnGesvd() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::vector<float> a = {1, 2, 3, 4};
  Data<float> a_s(a.data(), 4);
  Data<double> a_d(a.data(), 4);
  Data<sycl::float2> a_c(a.data(), 4);
  Data<sycl::double2> a_z(a.data(), 4);

  Data<float> s_s(2);
  Data<double> s_d(2);
  Data<float> s_c(2);
  Data<double> s_z(2);

  Data<float> u_s(4);
  Data<double> u_d(4);
  Data<sycl::float2> u_c(4);
  Data<sycl::double2> u_z(4);

  Data<float> vt_s(4);
  Data<double> vt_d(4);
  Data<sycl::float2> vt_c(4);
  Data<sycl::double2> vt_z(4);

  sycl::queue *handle;
  handle = &q_ct1;

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();

  size_t device_ws_size_s;
  size_t device_ws_size_d;
  size_t device_ws_size_c;
  size_t device_ws_size_z;

  int params;
  /*
  DPCT1026:28: The call to cusolverDnCreateParams was removed because this call
  is redundant in SYCL.
  */

  dpct::lapack::gesvd_scratchpad_size(
      *handle, 'A', 'A', 2, 2, dpct::library_data_t::real_float, 2,
      dpct::library_data_t::real_float, 2, dpct::library_data_t::real_float, 2,
      &device_ws_size_s);
  dpct::lapack::gesvd_scratchpad_size(
      *handle, 'A', 'A', 2, 2, dpct::library_data_t::real_double, 2,
      dpct::library_data_t::real_double, 2, dpct::library_data_t::real_double,
      2, &device_ws_size_d);
  dpct::lapack::gesvd_scratchpad_size(
      *handle, 'A', 'A', 2, 2, dpct::library_data_t::complex_float, 2,
      dpct::library_data_t::complex_float, 2,
      dpct::library_data_t::complex_float, 2, &device_ws_size_c);
  dpct::lapack::gesvd_scratchpad_size(
      *handle, 'A', 'A', 2, 2, dpct::library_data_t::complex_double, 2,
      dpct::library_data_t::complex_double, 2,
      dpct::library_data_t::complex_double, 2, &device_ws_size_z);

  void* device_ws_s;
  void* device_ws_d;
  void* device_ws_c;
  void* device_ws_z;
  device_ws_s = dpct::dpct_malloc(device_ws_size_s);
  device_ws_d = dpct::dpct_malloc(device_ws_size_d);
  device_ws_c = dpct::dpct_malloc(device_ws_size_c);
  device_ws_z = dpct::dpct_malloc(device_ws_size_z);

  int *info;
  info = (int *)dpct::dpct_malloc(sizeof(int));

  dpct::lapack::gesvd(*handle, 'A', 'A', 2, 2, dpct::library_data_t::real_float,
                      a_s.d_data, 2, dpct::library_data_t::real_float,
                      s_s.d_data, dpct::library_data_t::real_float, u_s.d_data,
                      2, dpct::library_data_t::real_float, vt_s.d_data, 2,
                      device_ws_s, device_ws_size_s, info);
  dpct::lapack::gesvd(*handle, 'A', 'A', 2, 2,
                      dpct::library_data_t::real_double, a_d.d_data, 2,
                      dpct::library_data_t::real_double, s_d.d_data,
                      dpct::library_data_t::real_double, u_d.d_data, 2,
                      dpct::library_data_t::real_double, vt_d.d_data, 2,
                      device_ws_d, device_ws_size_d, info);
  dpct::lapack::gesvd(*handle, 'A', 'A', 2, 2,
                      dpct::library_data_t::complex_float, a_c.d_data, 2,
                      dpct::library_data_t::real_float, s_c.d_data,
                      dpct::library_data_t::complex_float, u_c.d_data, 2,
                      dpct::library_data_t::complex_float, vt_c.d_data, 2,
                      device_ws_c, device_ws_size_c, info);
  dpct::lapack::gesvd(*handle, 'A', 'A', 2, 2,
                      dpct::library_data_t::complex_double, a_z.d_data, 2,
                      dpct::library_data_t::real_double, s_z.d_data,
                      dpct::library_data_t::complex_double, u_z.d_data, 2,
                      dpct::library_data_t::complex_double, vt_z.d_data, 2,
                      device_ws_z, device_ws_size_z, info);

  s_s.D2H();
  s_d.D2H();
  s_c.D2H();
  s_z.D2H();

  u_s.D2H();
  u_d.D2H();
  u_c.D2H();
  u_z.D2H();

  vt_s.D2H();
  vt_d.D2H();
  vt_c.D2H();
  vt_z.D2H();

  q_ct1.wait();

  /*
  DPCT1026:29: The call to cusolverDnDestroyParams was removed because this call
  is redundant in SYCL.
  */
  handle = nullptr;
  dpct::dpct_free(device_ws_s);
  dpct::dpct_free(device_ws_d);
  dpct::dpct_free(device_ws_c);
  dpct::dpct_free(device_ws_z);
  dpct::dpct_free(info);

  float expect_s[2] = {5.464985,0.365966};
  float expect_u[4] = {0.576048,0.817416,-0.817416,0.576048};
  float expect_vt[4] = {0.404554,0.914514,0.914514,-0.404554};

  if (compare_result(expect_s, s_s.h_data, 2) &&
      compare_result(expect_s, s_d.h_data, 2) &&
      compare_result(expect_s, s_c.h_data, 2) &&
      compare_result(expect_s, s_z.h_data, 2) &&
      compare_result(expect_u, u_s.h_data, 4) &&
      compare_result(expect_u, u_d.h_data, 4) &&
      compare_result(expect_u, u_c.h_data, 4) &&
      compare_result(expect_u, u_z.h_data, 4) &&
      compare_result(expect_vt, vt_s.h_data, 4) &&
      compare_result(expect_vt, vt_d.h_data, 4) &&
      compare_result(expect_vt, vt_c.h_data, 4) &&
      compare_result(expect_vt, vt_z.h_data, 4))
    printf("DnGesvd pass\n");
  else {
    printf("DnGesvd fail\n");
    test_passed = false;
  }
}

void test_cusolverDnXpotrf() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::vector<float> a = {2, -1, 0, -1, 2, -1, 0, -1, 2};
  Data<float> a_s(a.data(), 9);
  Data<double> a_d(a.data(), 9);
  Data<sycl::float2> a_c(a.data(), 9);
  Data<sycl::double2> a_z(a.data(), 9);

  sycl::queue *handle;
  handle = &q_ct1;

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();

  size_t device_ws_size_s;
  size_t device_ws_size_d;
  size_t device_ws_size_c;
  size_t device_ws_size_z;
  size_t host_ws_size_s;
  size_t host_ws_size_d;
  size_t host_ws_size_c;
  size_t host_ws_size_z;

  int params;
  /*
  DPCT1026:30: The call to cusolverDnCreateParams was removed because this call
  is redundant in SYCL.
  */

  dpct::lapack::potrf_scratchpad_size(*handle, oneapi::mkl::uplo::lower, 3,
                                      dpct::library_data_t::real_float, 3,
                                      &device_ws_size_s, &host_ws_size_s);
  dpct::lapack::potrf_scratchpad_size(*handle, oneapi::mkl::uplo::lower, 3,
                                      dpct::library_data_t::real_double, 3,
                                      &device_ws_size_d, &host_ws_size_d);
  dpct::lapack::potrf_scratchpad_size(*handle, oneapi::mkl::uplo::lower, 3,
                                      dpct::library_data_t::complex_float, 3,
                                      &device_ws_size_c, &host_ws_size_c);
  dpct::lapack::potrf_scratchpad_size(*handle, oneapi::mkl::uplo::lower, 3,
                                      dpct::library_data_t::complex_double, 3,
                                      &device_ws_size_z, &host_ws_size_z);

  void* device_ws_s;
  void* device_ws_d;
  void* device_ws_c;
  void* device_ws_z;
  void* host_ws_s;
  void* host_ws_d;
  void* host_ws_c;
  void* host_ws_z;
  device_ws_s = dpct::dpct_malloc(device_ws_size_s);
  device_ws_d = dpct::dpct_malloc(device_ws_size_d);
  device_ws_c = dpct::dpct_malloc(device_ws_size_c);
  device_ws_z = dpct::dpct_malloc(device_ws_size_z);
  host_ws_s = dpct::dpct_malloc(host_ws_size_s);
  host_ws_d = dpct::dpct_malloc(host_ws_size_d);
  host_ws_c = dpct::dpct_malloc(host_ws_size_c);
  host_ws_z = dpct::dpct_malloc(host_ws_size_z);

  int *info;
  info = (int *)dpct::dpct_malloc(sizeof(int));

  dpct::lapack::potrf(*handle, oneapi::mkl::uplo::lower, 3,
                      dpct::library_data_t::real_float, a_s.d_data, 3,
                      device_ws_s, device_ws_size_s, info);
  dpct::lapack::potrf(*handle, oneapi::mkl::uplo::lower, 3,
                      dpct::library_data_t::real_double, a_d.d_data, 3,
                      device_ws_d, device_ws_size_d, info);
  dpct::lapack::potrf(*handle, oneapi::mkl::uplo::lower, 3,
                      dpct::library_data_t::complex_float, a_c.d_data, 3,
                      device_ws_c, device_ws_size_c, info);
  dpct::lapack::potrf(*handle, oneapi::mkl::uplo::lower, 3,
                      dpct::library_data_t::complex_double, a_z.d_data, 3,
                      device_ws_z, device_ws_size_z, info);

  a_s.D2H();
  a_d.D2H();
  a_c.D2H();
  a_z.D2H();

  q_ct1.wait();

  /*
  DPCT1026:31: The call to cusolverDnDestroyParams was removed because this call
  is redundant in SYCL.
  */
  handle = nullptr;
  dpct::dpct_free(device_ws_s);
  dpct::dpct_free(device_ws_d);
  dpct::dpct_free(device_ws_c);
  dpct::dpct_free(device_ws_z);
  dpct::dpct_free(host_ws_s);
  dpct::dpct_free(host_ws_d);
  dpct::dpct_free(host_ws_c);
  dpct::dpct_free(host_ws_z);
  dpct::dpct_free(info);

  float expect_a[9] = {1.414214,-0.707107,0.000000,-1.000000,1.224745,-0.816497,0.000000,-1.000000,1.154701};
  if (compare_result(expect_a, a_s.h_data, 9) &&
      compare_result(expect_a, a_d.h_data, 9) &&
      compare_result(expect_a, a_c.h_data, 9) &&
      compare_result(expect_a, a_z.h_data, 9))
    printf("DnXpotrf pass\n");
  else {
    printf("DnXpotrf fail\n");
    test_passed = false;
  }
}

void test_cusolverDnPotrf() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::vector<float> a = {2, -1, 0, -1, 2, -1, 0, -1, 2};
  Data<float> a_s(a.data(), 9);
  Data<double> a_d(a.data(), 9);
  Data<sycl::float2> a_c(a.data(), 9);
  Data<sycl::double2> a_z(a.data(), 9);

  sycl::queue *handle;
  handle = &q_ct1;

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();

  size_t device_ws_size_s;
  size_t device_ws_size_d;
  size_t device_ws_size_c;
  size_t device_ws_size_z;

  int params;
  /*
  DPCT1026:32: The call to cusolverDnCreateParams was removed because this call
  is redundant in SYCL.
  */

  dpct::lapack::potrf_scratchpad_size(*handle, oneapi::mkl::uplo::lower, 3,
                                      dpct::library_data_t::real_float, 3,
                                      &device_ws_size_s);
  dpct::lapack::potrf_scratchpad_size(*handle, oneapi::mkl::uplo::lower, 3,
                                      dpct::library_data_t::real_double, 3,
                                      &device_ws_size_d);
  dpct::lapack::potrf_scratchpad_size(*handle, oneapi::mkl::uplo::lower, 3,
                                      dpct::library_data_t::complex_float, 3,
                                      &device_ws_size_c);
  dpct::lapack::potrf_scratchpad_size(*handle, oneapi::mkl::uplo::lower, 3,
                                      dpct::library_data_t::complex_double, 3,
                                      &device_ws_size_z);

  void* device_ws_s;
  void* device_ws_d;
  void* device_ws_c;
  void* device_ws_z;
  device_ws_s = dpct::dpct_malloc(device_ws_size_s);
  device_ws_d = dpct::dpct_malloc(device_ws_size_d);
  device_ws_c = dpct::dpct_malloc(device_ws_size_c);
  device_ws_z = dpct::dpct_malloc(device_ws_size_z);

  int *info;
  info = (int *)dpct::dpct_malloc(sizeof(int));

  dpct::lapack::potrf(*handle, oneapi::mkl::uplo::lower, 3,
                      dpct::library_data_t::real_float, a_s.d_data, 3,
                      device_ws_s, device_ws_size_s, info);
  dpct::lapack::potrf(*handle, oneapi::mkl::uplo::lower, 3,
                      dpct::library_data_t::real_double, a_d.d_data, 3,
                      device_ws_d, device_ws_size_d, info);
  dpct::lapack::potrf(*handle, oneapi::mkl::uplo::lower, 3,
                      dpct::library_data_t::complex_float, a_c.d_data, 3,
                      device_ws_c, device_ws_size_c, info);
  dpct::lapack::potrf(*handle, oneapi::mkl::uplo::lower, 3,
                      dpct::library_data_t::complex_double, a_z.d_data, 3,
                      device_ws_z, device_ws_size_z, info);

  a_s.D2H();
  a_d.D2H();
  a_c.D2H();
  a_z.D2H();

  q_ct1.wait();

  /*
  DPCT1026:33: The call to cusolverDnDestroyParams was removed because this call
  is redundant in SYCL.
  */
  handle = nullptr;
  dpct::dpct_free(device_ws_s);
  dpct::dpct_free(device_ws_d);
  dpct::dpct_free(device_ws_c);
  dpct::dpct_free(device_ws_z);
  dpct::dpct_free(info);

  float expect_a[9] = {1.414214,-0.707107,0.000000,-1.000000,1.224745,-0.816497,0.000000,-1.000000,1.154701};
  if (compare_result(expect_a, a_s.h_data, 9) &&
      compare_result(expect_a, a_d.h_data, 9) &&
      compare_result(expect_a, a_c.h_data, 9) &&
      compare_result(expect_a, a_z.h_data, 9))
    printf("DnPotrf pass\n");
  else {
    printf("DnPotrf fail\n");
    test_passed = false;
  }
}

void test_cusolverDnXpotrs() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::vector<float> a = {1.414214,-0.707107,0.000000,-0.707107,1.224745,-0.816497,0.000000,-0.816497,1.154701};
  Data<float> a_s(a.data(), 9);
  Data<double> a_d(a.data(), 9);
  Data<sycl::float2> a_c(a.data(), 9);
  Data<sycl::double2> a_z(a.data(), 9);
  std::vector<float> b = {0, 0, 4};
  Data<float> b_s(b.data(), 3);
  Data<double> b_d(b.data(), 3);
  Data<sycl::float2> b_c(b.data(), 3);
  Data<sycl::double2> b_z(b.data(), 3);

  sycl::queue *handle;
  handle = &q_ct1;

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();
  b_s.H2D();
  b_d.H2D();
  b_c.H2D();
  b_z.H2D();

  int params;
  /*
  DPCT1026:34: The call to cusolverDnCreateParams was removed because this call
  is redundant in SYCL.
  */

  int *info;
  info = (int *)dpct::dpct_malloc(sizeof(int));

  dpct::lapack::potrs(*handle, oneapi::mkl::uplo::lower, 3, 1,
                      dpct::library_data_t::real_float, a_s.d_data, 3,
                      dpct::library_data_t::real_float, b_s.d_data, 3, info);
  dpct::lapack::potrs(*handle, oneapi::mkl::uplo::lower, 3, 1,
                      dpct::library_data_t::real_double, a_d.d_data, 3,
                      dpct::library_data_t::real_double, b_d.d_data, 3, info);
  dpct::lapack::potrs(*handle, oneapi::mkl::uplo::lower, 3, 1,
                      dpct::library_data_t::complex_float, a_c.d_data, 3,
                      dpct::library_data_t::complex_float, b_c.d_data, 3, info);
  dpct::lapack::potrs(*handle, oneapi::mkl::uplo::lower, 3, 1,
                      dpct::library_data_t::complex_double, a_z.d_data, 3,
                      dpct::library_data_t::complex_double, b_z.d_data, 3,
                      info);

  b_s.D2H();
  b_d.D2H();
  b_c.D2H();
  b_z.D2H();

  q_ct1.wait();

  /*
  DPCT1026:35: The call to cusolverDnDestroyParams was removed because this call
  is redundant in SYCL.
  */
  handle = nullptr;
  dpct::dpct_free(info);

  float expect_b[3] = {1,2,3};
  if (compare_result(expect_b, b_s.h_data, 3) &&
      compare_result(expect_b, b_d.h_data, 3) &&
      compare_result(expect_b, b_c.h_data, 3) &&
      compare_result(expect_b, b_z.h_data, 3))
    printf("DnXpotrs pass\n");
  else {
    printf("DnXpotrs fail\n");
    test_passed = false;
  }
}

void test_cusolverDnPotrs() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::vector<float> a = {1.414214,-0.707107,0.000000,-0.707107,1.224745,-0.816497,0.000000,-0.816497,1.154701};
  Data<float> a_s(a.data(), 9);
  Data<double> a_d(a.data(), 9);
  Data<sycl::float2> a_c(a.data(), 9);
  Data<sycl::double2> a_z(a.data(), 9);
  std::vector<float> b = {0, 0, 4};
  Data<float> b_s(b.data(), 3);
  Data<double> b_d(b.data(), 3);
  Data<sycl::float2> b_c(b.data(), 3);
  Data<sycl::double2> b_z(b.data(), 3);

  sycl::queue *handle;
  handle = &q_ct1;

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();
  b_s.H2D();
  b_d.H2D();
  b_c.H2D();
  b_z.H2D();

  int params;
  /*
  DPCT1026:36: The call to cusolverDnCreateParams was removed because this call
  is redundant in SYCL.
  */

  int *info;
  info = (int *)dpct::dpct_malloc(sizeof(int));

  dpct::lapack::potrs(*handle, oneapi::mkl::uplo::lower, 3, 1,
                      dpct::library_data_t::real_float, a_s.d_data, 3,
                      dpct::library_data_t::real_float, b_s.d_data, 3, info);
  dpct::lapack::potrs(*handle, oneapi::mkl::uplo::lower, 3, 1,
                      dpct::library_data_t::real_double, a_d.d_data, 3,
                      dpct::library_data_t::real_double, b_d.d_data, 3, info);
  dpct::lapack::potrs(*handle, oneapi::mkl::uplo::lower, 3, 1,
                      dpct::library_data_t::complex_float, a_c.d_data, 3,
                      dpct::library_data_t::complex_float, b_c.d_data, 3, info);
  dpct::lapack::potrs(*handle, oneapi::mkl::uplo::lower, 3, 1,
                      dpct::library_data_t::complex_double, a_z.d_data, 3,
                      dpct::library_data_t::complex_double, b_z.d_data, 3,
                      info);

  b_s.D2H();
  b_d.D2H();
  b_c.D2H();
  b_z.D2H();

  q_ct1.wait();

  /*
  DPCT1026:37: The call to cusolverDnDestroyParams was removed because this call
  is redundant in SYCL.
  */
  handle = nullptr;
  dpct::dpct_free(info);

  float expect_b[3] = {1,2,3};
  if (compare_result(expect_b, b_s.h_data, 3) &&
      compare_result(expect_b, b_d.h_data, 3) &&
      compare_result(expect_b, b_c.h_data, 3) &&
      compare_result(expect_b, b_z.h_data, 3))
    printf("DnPotrs pass\n");
  else {
    printf("DnPotrs fail\n");
    test_passed = false;
  }
}

void test_cusolverDnSyevdx() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::vector<float> a = {1, 2, 2, 4};
  Data<float> a_s(a.data(), 4);
  Data<double> a_d(a.data(), 4);
  Data<sycl::float2> a_c(a.data(), 4);
  Data<sycl::double2> a_z(a.data(), 4);
  Data<float> w_s(2);
  Data<double> w_d(2);
  Data<float> w_c(2);
  Data<double> w_z(2);

  sycl::queue *handle;
  handle = &q_ct1;

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();

  int params;
  /*
  DPCT1026:38: The call to cusolverDnCreateParams was removed because this call
  is redundant in SYCL.
  */

  size_t lwork_s;
  size_t lwork_d;
  size_t lwork_c;
  size_t lwork_z;

  int64_t h_meig_s;
  int64_t h_meig_d;
  int64_t h_meig_c;
  int64_t h_meig_z;
  float vlvu_s = 0;
  double vlvu_d = 0;
  float vlvu_c = 0;
  double vlvu_z = 0;

  dpct::lapack::syheevx_scratchpad_size(
      *handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all,
      oneapi::mkl::uplo::upper, 2, dpct::library_data_t::real_float, 2, &vlvu_s,
      &vlvu_s, 0, 0, dpct::library_data_t::real_float, &lwork_s);
  dpct::lapack::syheevx_scratchpad_size(
      *handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all,
      oneapi::mkl::uplo::upper, 2, dpct::library_data_t::real_double, 2,
      &vlvu_d, &vlvu_d, 0, 0, dpct::library_data_t::real_double, &lwork_d);
  dpct::lapack::syheevx_scratchpad_size(
      *handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all,
      oneapi::mkl::uplo::upper, 2, dpct::library_data_t::complex_float, 2,
      &vlvu_c, &vlvu_c, 0, 0, dpct::library_data_t::real_float, &lwork_c);
  dpct::lapack::syheevx_scratchpad_size(
      *handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all,
      oneapi::mkl::uplo::upper, 2, dpct::library_data_t::complex_double, 2,
      &vlvu_z, &vlvu_z, 0, 0, dpct::library_data_t::real_double, &lwork_z);

  void* device_ws_s;
  void* device_ws_d;
  void* device_ws_c;
  void* device_ws_z;
  device_ws_s = dpct::dpct_malloc(lwork_s);
  device_ws_d = dpct::dpct_malloc(lwork_d);
  device_ws_c = dpct::dpct_malloc(lwork_c);
  device_ws_z = dpct::dpct_malloc(lwork_z);

  int *info;
  info = (int *)dpct::dpct_malloc(sizeof(int));

  dpct::lapack::syheevx(
      *handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all,
      oneapi::mkl::uplo::upper, 2, dpct::library_data_t::real_float, a_s.d_data,
      2, &vlvu_s, &vlvu_s, 0, 0, &h_meig_s, dpct::library_data_t::real_float,
      w_s.d_data, device_ws_s, lwork_s, info);
  dpct::lapack::syheevx(*handle, oneapi::mkl::job::vec,
                        oneapi::mkl::rangev::all, oneapi::mkl::uplo::upper, 2,
                        dpct::library_data_t::real_double, a_d.d_data, 2,
                        &vlvu_d, &vlvu_d, 0, 0, &h_meig_d,
                        dpct::library_data_t::real_double, w_d.d_data,
                        device_ws_d, lwork_d, info);
  dpct::lapack::syheevx(
      *handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all,
      oneapi::mkl::uplo::upper, 2, dpct::library_data_t::complex_float,
      a_c.d_data, 2, &vlvu_c, &vlvu_c, 0, 0, &h_meig_c,
      dpct::library_data_t::real_float, w_c.d_data, device_ws_c, lwork_c, info);
  dpct::lapack::syheevx(*handle, oneapi::mkl::job::vec,
                        oneapi::mkl::rangev::all, oneapi::mkl::uplo::upper, 2,
                        dpct::library_data_t::complex_double, a_z.d_data, 2,
                        &vlvu_z, &vlvu_z, 0, 0, &h_meig_z,
                        dpct::library_data_t::real_double, w_z.d_data,
                        device_ws_z, lwork_z, info);

  a_s.D2H();
  a_d.D2H();
  a_c.D2H();
  a_z.D2H();
  w_s.D2H();
  w_d.D2H();
  w_c.D2H();
  w_z.D2H();

  q_ct1.wait();

  /*
  DPCT1026:39: The call to cusolverDnDestroyParams was removed because this call
  is redundant in SYCL.
  */
  handle = nullptr;
  dpct::dpct_free(device_ws_s);
  dpct::dpct_free(device_ws_d);
  dpct::dpct_free(device_ws_c);
  dpct::dpct_free(device_ws_z);
  dpct::dpct_free(info);

  printf("a_s:%f,%f,%f,%f\n", a_s.h_data[0], a_s.h_data[1], a_s.h_data[2], a_s.h_data[3]);
  printf("h_meig_s:%ld\n", h_meig_s);
  printf("w_s:%f,%f\n", w_s.h_data[0], w_s.h_data[1]);

  float expect_a[4] = {0.894427,-0.447214,0.447214,0.894427};
  int64_t expect_h_meig = 2;
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
    printf("DnSyevdx pass\n");
  else {
    printf("DnSyevdx fail\n");
    test_passed = false;
  }
}

void test_cusolverDnXsyevdx() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::vector<float> a = {1, 2, 2, 4};
  Data<float> a_s(a.data(), 4);
  Data<double> a_d(a.data(), 4);
  Data<sycl::float2> a_c(a.data(), 4);
  Data<sycl::double2> a_z(a.data(), 4);
  Data<float> w_s(2);
  Data<double> w_d(2);
  Data<float> w_c(2);
  Data<double> w_z(2);

  sycl::queue *handle;
  handle = &q_ct1;

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();

  int params;
  /*
  DPCT1026:40: The call to cusolverDnCreateParams was removed because this call
  is redundant in SYCL.
  */

  size_t lwork_s;
  size_t lwork_d;
  size_t lwork_c;
  size_t lwork_z;
  size_t lwork_host_s;
  size_t lwork_host_d;
  size_t lwork_host_c;
  size_t lwork_host_z;

  int64_t h_meig_s;
  int64_t h_meig_d;
  int64_t h_meig_c;
  int64_t h_meig_z;
  float vlvu_s = 0;
  double vlvu_d = 0;
  float vlvu_c = 0;
  double vlvu_z = 0;

  dpct::lapack::syheevx_scratchpad_size(
      *handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all,
      oneapi::mkl::uplo::upper, 2, dpct::library_data_t::real_float, 2, &vlvu_s,
      &vlvu_s, 0, 0, dpct::library_data_t::real_float, &lwork_s, &lwork_host_s);
  dpct::lapack::syheevx_scratchpad_size(
      *handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all,
      oneapi::mkl::uplo::upper, 2, dpct::library_data_t::real_double, 2,
      &vlvu_d, &vlvu_d, 0, 0, dpct::library_data_t::real_double, &lwork_d,
      &lwork_host_d);
  dpct::lapack::syheevx_scratchpad_size(
      *handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all,
      oneapi::mkl::uplo::upper, 2, dpct::library_data_t::complex_float, 2,
      &vlvu_c, &vlvu_c, 0, 0, dpct::library_data_t::real_float, &lwork_c,
      &lwork_host_c);
  dpct::lapack::syheevx_scratchpad_size(
      *handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all,
      oneapi::mkl::uplo::upper, 2, dpct::library_data_t::complex_double, 2,
      &vlvu_z, &vlvu_z, 0, 0, dpct::library_data_t::real_double, &lwork_z,
      &lwork_host_z);

  void* device_ws_s;
  void* device_ws_d;
  void* device_ws_c;
  void* device_ws_z;
  device_ws_s = dpct::dpct_malloc(lwork_s);
  device_ws_d = dpct::dpct_malloc(lwork_d);
  device_ws_c = dpct::dpct_malloc(lwork_c);
  device_ws_z = dpct::dpct_malloc(lwork_z);
  void* host_ws_s;
  void* host_ws_d;
  void* host_ws_c;
  void* host_ws_z;
  host_ws_s = malloc(lwork_host_s);
  host_ws_d = malloc(lwork_host_d);
  host_ws_c = malloc(lwork_host_c);
  host_ws_z = malloc(lwork_host_z);

  int *info;
  info = (int *)dpct::dpct_malloc(sizeof(int));

  dpct::lapack::syheevx(
      *handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all,
      oneapi::mkl::uplo::upper, 2, dpct::library_data_t::real_float, a_s.d_data,
      2, &vlvu_s, &vlvu_s, 0, 0, &h_meig_s, dpct::library_data_t::real_float,
      w_s.d_data, device_ws_s, lwork_s, info);
  dpct::lapack::syheevx(*handle, oneapi::mkl::job::vec,
                        oneapi::mkl::rangev::all, oneapi::mkl::uplo::upper, 2,
                        dpct::library_data_t::real_double, a_d.d_data, 2,
                        &vlvu_d, &vlvu_d, 0, 0, &h_meig_d,
                        dpct::library_data_t::real_double, w_d.d_data,
                        device_ws_d, lwork_d, info);
  dpct::lapack::syheevx(
      *handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all,
      oneapi::mkl::uplo::upper, 2, dpct::library_data_t::complex_float,
      a_c.d_data, 2, &vlvu_c, &vlvu_c, 0, 0, &h_meig_c,
      dpct::library_data_t::real_float, w_c.d_data, device_ws_c, lwork_c, info);
  dpct::lapack::syheevx(*handle, oneapi::mkl::job::vec,
                        oneapi::mkl::rangev::all, oneapi::mkl::uplo::upper, 2,
                        dpct::library_data_t::complex_double, a_z.d_data, 2,
                        &vlvu_z, &vlvu_z, 0, 0, &h_meig_z,
                        dpct::library_data_t::real_double, w_z.d_data,
                        device_ws_z, lwork_z, info);

  a_s.D2H();
  a_d.D2H();
  a_c.D2H();
  a_z.D2H();
  w_s.D2H();
  w_d.D2H();
  w_c.D2H();
  w_z.D2H();

  q_ct1.wait();

  /*
  DPCT1026:41: The call to cusolverDnDestroyParams was removed because this call
  is redundant in SYCL.
  */
  handle = nullptr;
  dpct::dpct_free(device_ws_s);
  dpct::dpct_free(device_ws_d);
  dpct::dpct_free(device_ws_c);
  dpct::dpct_free(device_ws_z);
  free(host_ws_s);
  free(host_ws_d);
  free(host_ws_c);
  free(host_ws_z);
  dpct::dpct_free(info);

  printf("a_s:%f,%f,%f,%f\n", a_s.h_data[0], a_s.h_data[1], a_s.h_data[2], a_s.h_data[3]);
  printf("h_meig_s:%ld\n", h_meig_s);
  printf("w_s:%f,%f\n", w_s.h_data[0], w_s.h_data[1]);

  float expect_a[4] = {0.894427,-0.447214,0.447214,0.894427};
  int64_t expect_h_meig = 2;
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
    printf("DnXsyevdx pass\n");
  else {
    printf("DnXsyevdx fail\n");
    test_passed = false;
  }
}

void test_cusolverDnTsyevdx_cusolverDnTheevdx() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::vector<float> a = {1, 2, 2, 4};
  Data<float> a_s(a.data(), 4);
  Data<double> a_d(a.data(), 4);
  Data<sycl::float2> a_c(a.data(), 4);
  Data<sycl::double2> a_z(a.data(), 4);
  Data<float> w_s(2);
  Data<double> w_d(2);
  Data<float> w_c(2);
  Data<double> w_z(2);

  sycl::queue *handle;
  handle = &q_ct1;

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();

  int params;
  /*
  DPCT1026:42: The call to cusolverDnCreateParams was removed because this call
  is redundant in SYCL.
  */

  int lwork_s;
  int lwork_d;
  int lwork_c;
  int lwork_z;

  int h_meig_s;
  int h_meig_d;
  int h_meig_c;
  int h_meig_z;

  dpct::lapack::syheevx_scratchpad_size<float, float>(
      *handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all,
      oneapi::mkl::uplo::upper, 2, 2, 0, 0, 0, 0, &lwork_s);
  dpct::lapack::syheevx_scratchpad_size<double, double>(
      *handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all,
      oneapi::mkl::uplo::upper, 2, 2, 0, 0, 0, 0, &lwork_d);
  dpct::lapack::syheevx_scratchpad_size<sycl::float2, float>(
      *handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all,
      oneapi::mkl::uplo::upper, 2, 2, 0, 0, 0, 0, &lwork_c);
  dpct::lapack::syheevx_scratchpad_size<sycl::double2, double>(
      *handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all,
      oneapi::mkl::uplo::upper, 2, 2, 0, 0, 0, 0, &lwork_z);

  float* device_ws_s;
  double* device_ws_d;
  sycl::float2 *device_ws_c;
  sycl::double2 *device_ws_z;
  device_ws_s = (float *)dpct::dpct_malloc(lwork_s * sizeof(float));
  device_ws_d = (double *)dpct::dpct_malloc(lwork_d * sizeof(double));
  device_ws_c =
      (sycl::float2 *)dpct::dpct_malloc(lwork_c * sizeof(sycl::float2));
  device_ws_z =
      (sycl::double2 *)dpct::dpct_malloc(lwork_z * sizeof(sycl::double2));

  int *info;
  info = (int *)dpct::dpct_malloc(sizeof(int));

  dpct::lapack::syheevx<float, float>(
      *handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all,
      oneapi::mkl::uplo::upper, 2, a_s.d_data, 2, 0, 0, 0, 0, &h_meig_s,
      w_s.d_data, device_ws_s, lwork_s, info);
  dpct::lapack::syheevx<double, double>(
      *handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all,
      oneapi::mkl::uplo::upper, 2, a_d.d_data, 2, 0, 0, 0, 0, &h_meig_d,
      w_d.d_data, device_ws_d, lwork_d, info);
  dpct::lapack::syheevx<sycl::float2, float>(
      *handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all,
      oneapi::mkl::uplo::upper, 2, a_c.d_data, 2, 0, 0, 0, 0, &h_meig_c,
      w_c.d_data, device_ws_c, lwork_c, info);
  dpct::lapack::syheevx<sycl::double2, double>(
      *handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all,
      oneapi::mkl::uplo::upper, 2, a_z.d_data, 2, 0, 0, 0, 0, &h_meig_z,
      w_z.d_data, device_ws_z, lwork_z, info);

  a_s.D2H();
  a_d.D2H();
  a_c.D2H();
  a_z.D2H();
  w_s.D2H();
  w_d.D2H();
  w_c.D2H();
  w_z.D2H();

  q_ct1.wait();

  /*
  DPCT1026:43: The call to cusolverDnDestroyParams was removed because this call
  is redundant in SYCL.
  */
  handle = nullptr;
  dpct::dpct_free(device_ws_s);
  dpct::dpct_free(device_ws_d);
  dpct::dpct_free(device_ws_c);
  dpct::dpct_free(device_ws_z);
  dpct::dpct_free(info);

  printf("a_s:%f,%f,%f,%f\n", a_s.h_data[0], a_s.h_data[1], a_s.h_data[2], a_s.h_data[3]);
  printf("h_meig_s:%d\n", h_meig_s);
  printf("w_s:%f,%f\n", w_s.h_data[0], w_s.h_data[1]);

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
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::vector<float> a = {1, 2, 2, 4};
  Data<float> a_s(a.data(), 4);
  Data<double> a_d(a.data(), 4);
  Data<sycl::float2> a_c(a.data(), 4);
  Data<sycl::double2> a_z(a.data(), 4);
  std::vector<float> b = {1, 0, 0, 1};
  Data<float> b_s(b.data(), 4);
  Data<double> b_d(b.data(), 4);
  Data<sycl::float2> b_c(b.data(), 4);
  Data<sycl::double2> b_z(b.data(), 4);
  Data<float> w_s(2);
  Data<double> w_d(2);
  Data<float> w_c(2);
  Data<double> w_z(2);

  sycl::queue *handle;
  handle = &q_ct1;

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();
  b_s.H2D();
  b_d.H2D();
  b_c.H2D();
  b_z.H2D();

  int params;
  /*
  DPCT1026:44: The call to cusolverDnCreateParams was removed because this call
  is redundant in SYCL.
  */

  int lwork_s;
  int lwork_d;
  int lwork_c;
  int lwork_z;

  int h_meig_s;
  int h_meig_d;
  int h_meig_c;
  int h_meig_z;

  dpct::lapack::syhegvx_scratchpad_size<float, float>(
      *handle, 1, oneapi::mkl::job::vec, oneapi::mkl::rangev::all,
      oneapi::mkl::uplo::upper, 2, 2, 2, 0, 0, 0, 0, &lwork_s);
  dpct::lapack::syhegvx_scratchpad_size<double, double>(
      *handle, 1, oneapi::mkl::job::vec, oneapi::mkl::rangev::all,
      oneapi::mkl::uplo::upper, 2, 2, 2, 0, 0, 0, 0, &lwork_d);
  dpct::lapack::syhegvx_scratchpad_size<sycl::float2, float>(
      *handle, 1, oneapi::mkl::job::vec, oneapi::mkl::rangev::all,
      oneapi::mkl::uplo::upper, 2, 2, 2, 0, 0, 0, 0, &lwork_c);
  dpct::lapack::syhegvx_scratchpad_size<sycl::double2, double>(
      *handle, 1, oneapi::mkl::job::vec, oneapi::mkl::rangev::all,
      oneapi::mkl::uplo::upper, 2, 2, 2, 0, 0, 0, 0, &lwork_z);

  float* device_ws_s;
  double* device_ws_d;
  sycl::float2 *device_ws_c;
  sycl::double2 *device_ws_z;
  device_ws_s = (float *)dpct::dpct_malloc(lwork_s * sizeof(float));
  device_ws_d = (double *)dpct::dpct_malloc(lwork_d * sizeof(double));
  device_ws_c =
      (sycl::float2 *)dpct::dpct_malloc(lwork_c * sizeof(sycl::float2));
  device_ws_z =
      (sycl::double2 *)dpct::dpct_malloc(lwork_z * sizeof(sycl::double2));

  int *info;
  info = (int *)dpct::dpct_malloc(sizeof(int));

  dpct::lapack::syhegvx<float, float>(
      *handle, 1, oneapi::mkl::job::vec, oneapi::mkl::rangev::all,
      oneapi::mkl::uplo::upper, 2, a_s.d_data, 2, b_s.d_data, 2, 0, 0, 0, 0,
      &h_meig_s, w_s.d_data, device_ws_s, lwork_s, info);
  dpct::lapack::syhegvx<double, double>(
      *handle, 1, oneapi::mkl::job::vec, oneapi::mkl::rangev::all,
      oneapi::mkl::uplo::upper, 2, a_d.d_data, 2, b_d.d_data, 2, 0, 0, 0, 0,
      &h_meig_d, w_d.d_data, device_ws_d, lwork_d, info);
  dpct::lapack::syhegvx<sycl::float2, float>(
      *handle, 1, oneapi::mkl::job::vec, oneapi::mkl::rangev::all,
      oneapi::mkl::uplo::upper, 2, a_c.d_data, 2, b_c.d_data, 2, 0, 0, 0, 0,
      &h_meig_c, w_c.d_data, device_ws_c, lwork_c, info);
  dpct::lapack::syhegvx<sycl::double2, double>(
      *handle, 1, oneapi::mkl::job::vec, oneapi::mkl::rangev::all,
      oneapi::mkl::uplo::upper, 2, a_z.d_data, 2, b_z.d_data, 2, 0, 0, 0, 0,
      &h_meig_z, w_z.d_data, device_ws_z, lwork_z, info);

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

  q_ct1.wait();

  /*
  DPCT1026:45: The call to cusolverDnDestroyParams was removed because this call
  is redundant in SYCL.
  */
  handle = nullptr;
  dpct::dpct_free(device_ws_s);
  dpct::dpct_free(device_ws_d);
  dpct::dpct_free(device_ws_c);
  dpct::dpct_free(device_ws_z);
  dpct::dpct_free(info);

  printf("a_s:%f,%f,%f,%f\n", a_s.h_data[0], a_s.h_data[1], a_s.h_data[2], a_s.h_data[3]);
  printf("b_s:%f,%f,%f,%f\n", b_s.h_data[0], b_s.h_data[1], b_s.h_data[2], b_s.h_data[3]);
  printf("h_meig_s:%d\n", h_meig_s);
  printf("w_s:%f,%f\n", w_s.h_data[0], w_s.h_data[1]);

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
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::vector<float> a = {1, 2, 2, 4};
  Data<float> a_s(a.data(), 4);
  Data<double> a_d(a.data(), 4);
  Data<sycl::float2> a_c(a.data(), 4);
  Data<sycl::double2> a_z(a.data(), 4);
  std::vector<float> b = {1, 0, 0, 1};
  Data<float> b_s(b.data(), 4);
  Data<double> b_d(b.data(), 4);
  Data<sycl::float2> b_c(b.data(), 4);
  Data<sycl::double2> b_z(b.data(), 4);
  Data<float> w_s(2);
  Data<double> w_d(2);
  Data<float> w_c(2);
  Data<double> w_z(2);

  sycl::queue *handle;
  handle = &q_ct1;

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();
  b_s.H2D();
  b_d.H2D();
  b_c.H2D();
  b_z.H2D();

  int params;
  /*
  DPCT1026:46: The call to cusolverDnCreateSyevjInfo was removed because this
  call is redundant in SYCL.
  */

  int lwork_s;
  int lwork_d;
  int lwork_c;
  int lwork_z;

  dpct::lapack::syhegvd_scratchpad_size<float>(
      *handle, 1, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 2, 2, 2,
      &lwork_s);
  dpct::lapack::syhegvd_scratchpad_size<float>(
      *handle, 1, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 2, 2, 2,
      &lwork_d);
  dpct::lapack::syhegvd_scratchpad_size<sycl::float2>(
      *handle, 1, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 2, 2, 2,
      &lwork_c);
  dpct::lapack::syhegvd_scratchpad_size<sycl::double2>(
      *handle, 1, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 2, 2, 2,
      &lwork_z);

  float* device_ws_s;
  double* device_ws_d;
  sycl::float2 *device_ws_c;
  sycl::double2 *device_ws_z;
  device_ws_s = (float *)dpct::dpct_malloc(lwork_s * sizeof(float));
  device_ws_d = (double *)dpct::dpct_malloc(lwork_d * sizeof(double));
  device_ws_c =
      (sycl::float2 *)dpct::dpct_malloc(lwork_c * sizeof(sycl::float2));
  device_ws_z =
      (sycl::double2 *)dpct::dpct_malloc(lwork_z * sizeof(sycl::double2));

  int *info;
  info = (int *)dpct::dpct_malloc(sizeof(int));

  dpct::lapack::syhegvd<float, float>(
      *handle, 1, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 2,
      a_s.d_data, 2, b_s.d_data, 2, w_s.d_data, device_ws_s, lwork_s, info);
  dpct::lapack::syhegvd<double, double>(
      *handle, 1, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 2,
      a_d.d_data, 2, b_d.d_data, 2, w_d.d_data, device_ws_d, lwork_d, info);
  dpct::lapack::syhegvd<sycl::float2, float>(
      *handle, 1, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 2,
      a_c.d_data, 2, b_c.d_data, 2, w_c.d_data, device_ws_c, lwork_c, info);
  dpct::lapack::syhegvd<sycl::double2, double>(
      *handle, 1, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 2,
      a_z.d_data, 2, b_z.d_data, 2, w_z.d_data, device_ws_z, lwork_z, info);

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

  q_ct1.wait();

  /*
  DPCT1026:47: The call to cusolverDnDestroySyevjInfo was removed because this
  call is redundant in SYCL.
  */
  handle = nullptr;
  dpct::dpct_free(device_ws_s);
  dpct::dpct_free(device_ws_d);
  dpct::dpct_free(device_ws_c);
  dpct::dpct_free(device_ws_z);
  dpct::dpct_free(info);

  printf("a_s:%f,%f,%f,%f\n", a_s.h_data[0], a_s.h_data[1], a_s.h_data[2], a_s.h_data[3]);
  printf("b_s:%f,%f,%f,%f\n", b_s.h_data[0], b_s.h_data[1], b_s.h_data[2], b_s.h_data[3]);
  printf("w_s:%f,%f\n", w_s.h_data[0], w_s.h_data[1]);

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
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::vector<float> a = {1, 2, 2, 4};
  Data<float> a_s(a.data(), 4);
  Data<double> a_d(a.data(), 4);
  Data<sycl::float2> a_c(a.data(), 4);
  Data<sycl::double2> a_z(a.data(), 4);
  Data<float> w_s(2);
  Data<double> w_d(2);
  Data<float> w_c(2);
  Data<double> w_z(2);

  sycl::queue *handle;
  handle = &q_ct1;

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();

  int params;
  /*
  DPCT1026:2: The call to cusolverDnCreateSyevjInfo was removed because this
  call is redundant in SYCL.
  */

  int lwork_s;
  int lwork_d;
  int lwork_c;
  int lwork_z;

  dpct::lapack::syheev_scratchpad_size<float>(
      *handle, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 2, 2, &lwork_s);
  dpct::lapack::syheev_scratchpad_size<double>(
      *handle, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 2, 2, &lwork_d);
  dpct::lapack::syheev_scratchpad_size<sycl::float2>(
      *handle, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 2, 2, &lwork_c);
  dpct::lapack::syheev_scratchpad_size<sycl::double2>(
      *handle, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 2, 2, &lwork_z);

  float* device_ws_s;
  double* device_ws_d;
  sycl::float2 *device_ws_c;
  sycl::double2 *device_ws_z;
  device_ws_s = (float *)dpct::dpct_malloc(lwork_s * sizeof(float));
  device_ws_d = (double *)dpct::dpct_malloc(lwork_d * sizeof(double));
  device_ws_c =
      (sycl::float2 *)dpct::dpct_malloc(lwork_c * sizeof(sycl::float2));
  device_ws_z =
      (sycl::double2 *)dpct::dpct_malloc(lwork_z * sizeof(sycl::double2));

  int *info;
  info = (int *)dpct::dpct_malloc(sizeof(int));

  dpct::lapack::syheev<float, float>(*handle, oneapi::mkl::job::vec,
                                     oneapi::mkl::uplo::upper, 2, a_s.d_data, 2,
                                     w_s.d_data, device_ws_s, lwork_s, info);
  dpct::lapack::syheev<double, double>(
      *handle, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 2, a_d.d_data,
      2, w_d.d_data, device_ws_d, lwork_d, info);
  dpct::lapack::syheev<sycl::float2, float>(
      *handle, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 2, a_c.d_data,
      2, w_c.d_data, device_ws_c, lwork_c, info);
  dpct::lapack::syheev<sycl::double2, double>(
      *handle, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 2, a_z.d_data,
      2, w_z.d_data, device_ws_z, lwork_z, info);

  a_s.D2H();
  a_d.D2H();
  a_c.D2H();
  a_z.D2H();
  w_s.D2H();
  w_d.D2H();
  w_c.D2H();
  w_z.D2H();

  q_ct1.wait();

  /*
  DPCT1026:3: The call to cusolverDnDestroySyevjInfo was removed because this
  call is redundant in SYCL.
  */
  handle = nullptr;
  dpct::dpct_free(device_ws_s);
  dpct::dpct_free(device_ws_d);
  dpct::dpct_free(device_ws_c);
  dpct::dpct_free(device_ws_z);
  dpct::dpct_free(info);

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
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::vector<float> a = {1, 2, 2, 4};
  Data<float> a_s(a.data(), 4);
  Data<double> a_d(a.data(), 4);
  Data<sycl::float2> a_c(a.data(), 4);
  Data<sycl::double2> a_z(a.data(), 4);
  Data<float> w_s(2);
  Data<double> w_d(2);
  Data<float> w_c(2);
  Data<double> w_z(2);

  sycl::queue *handle;
  handle = &q_ct1;

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();

  int params;
  /*
  DPCT1026:46: The call to cusolverDnCreateParams was removed because this call
  is redundant in SYCL.
  */

  size_t lwork_s;
  size_t lwork_d;
  size_t lwork_c;
  size_t lwork_z;
  size_t lwork_host_s;
  size_t lwork_host_d;
  size_t lwork_host_c;
  size_t lwork_host_z;

  dpct::lapack::syheevd_scratchpad_size(
      *handle, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 2,
      dpct::library_data_t::real_float, 2, dpct::library_data_t::real_float,
      &lwork_s, &lwork_host_s);
  dpct::lapack::syheevd_scratchpad_size(
      *handle, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 2,
      dpct::library_data_t::real_double, 2, dpct::library_data_t::real_double,
      &lwork_d, &lwork_host_d);
  dpct::lapack::syheevd_scratchpad_size(
      *handle, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 2,
      dpct::library_data_t::complex_float, 2, dpct::library_data_t::real_float,
      &lwork_c, &lwork_host_c);
  dpct::lapack::syheevd_scratchpad_size(
      *handle, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 2,
      dpct::library_data_t::complex_double, 2,
      dpct::library_data_t::real_double, &lwork_z, &lwork_host_z);

  void* device_ws_s;
  void* device_ws_d;
  void* device_ws_c;
  void* device_ws_z;
  device_ws_s = dpct::dpct_malloc(lwork_s);
  device_ws_d = dpct::dpct_malloc(lwork_d);
  device_ws_c = dpct::dpct_malloc(lwork_c);
  device_ws_z = dpct::dpct_malloc(lwork_z);
  void* host_ws_s;
  void* host_ws_d;
  void* host_ws_c;
  void* host_ws_z;
  host_ws_s = malloc(lwork_host_s);
  host_ws_d = malloc(lwork_host_d);
  host_ws_c = malloc(lwork_host_c);
  host_ws_z = malloc(lwork_host_z);

  int *info;
  info = (int *)dpct::dpct_malloc(sizeof(int));

  dpct::lapack::syheevd(
      *handle, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 2,
      dpct::library_data_t::real_float, a_s.d_data, 2,
      dpct::library_data_t::real_float, w_s.d_data, device_ws_s, lwork_s, info);
  dpct::lapack::syheevd(*handle, oneapi::mkl::job::vec,
                        oneapi::mkl::uplo::upper, 2,
                        dpct::library_data_t::real_double, a_d.d_data, 2,
                        dpct::library_data_t::real_double, w_d.d_data,
                        device_ws_d, lwork_d, info);
  dpct::lapack::syheevd(
      *handle, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 2,
      dpct::library_data_t::complex_float, a_c.d_data, 2,
      dpct::library_data_t::real_float, w_c.d_data, device_ws_c, lwork_c, info);
  dpct::lapack::syheevd(*handle, oneapi::mkl::job::vec,
                        oneapi::mkl::uplo::upper, 2,
                        dpct::library_data_t::complex_double, a_z.d_data, 2,
                        dpct::library_data_t::real_double, w_z.d_data,
                        device_ws_z, lwork_z, info);

  a_s.D2H();
  a_d.D2H();
  a_c.D2H();
  a_z.D2H();
  w_s.D2H();
  w_d.D2H();
  w_c.D2H();
  w_z.D2H();

  q_ct1.wait();

  /*
  DPCT1026:47: The call to cusolverDnDestroyParams was removed because this call
  is redundant in SYCL.
  */
  handle = nullptr;
  dpct::dpct_free(device_ws_s);
  dpct::dpct_free(device_ws_d);
  dpct::dpct_free(device_ws_c);
  dpct::dpct_free(device_ws_z);
  free(host_ws_s);
  free(host_ws_d);
  free(host_ws_c);
  free(host_ws_z);
  dpct::dpct_free(info);

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
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::vector<float> a = {1, 2, 2, 4};
  Data<float> a_s(a.data(), 4);
  Data<double> a_d(a.data(), 4);
  Data<sycl::float2> a_c(a.data(), 4);
  Data<sycl::double2> a_z(a.data(), 4);
  Data<float> w_s(2);
  Data<double> w_d(2);
  Data<float> w_c(2);
  Data<double> w_z(2);

  sycl::queue *handle;
  handle = &q_ct1;

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();

  int params;
  /*
  DPCT1026:48: The call to cusolverDnCreateParams was removed because this call
  is redundant in SYCL.
  */

  size_t lwork_s;
  size_t lwork_d;
  size_t lwork_c;
  size_t lwork_z;

  dpct::lapack::syheevd_scratchpad_size(
      *handle, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 2,
      dpct::library_data_t::real_float, 2, dpct::library_data_t::real_float,
      &lwork_s);
  dpct::lapack::syheevd_scratchpad_size(
      *handle, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 2,
      dpct::library_data_t::real_double, 2, dpct::library_data_t::real_double,
      &lwork_d);
  dpct::lapack::syheevd_scratchpad_size(
      *handle, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 2,
      dpct::library_data_t::complex_float, 2, dpct::library_data_t::real_float,
      &lwork_c);
  dpct::lapack::syheevd_scratchpad_size(
      *handle, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 2,
      dpct::library_data_t::complex_double, 2,
      dpct::library_data_t::real_double, &lwork_z);

  void* device_ws_s;
  void* device_ws_d;
  void* device_ws_c;
  void* device_ws_z;
  device_ws_s = dpct::dpct_malloc(lwork_s);
  device_ws_d = dpct::dpct_malloc(lwork_d);
  device_ws_c = dpct::dpct_malloc(lwork_c);
  device_ws_z = dpct::dpct_malloc(lwork_z);

  int *info;
  info = (int *)dpct::dpct_malloc(sizeof(int));

  dpct::lapack::syheevd(
      *handle, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 2,
      dpct::library_data_t::real_float, a_s.d_data, 2,
      dpct::library_data_t::real_float, w_s.d_data, device_ws_s, lwork_s, info);
  dpct::lapack::syheevd(*handle, oneapi::mkl::job::vec,
                        oneapi::mkl::uplo::upper, 2,
                        dpct::library_data_t::real_double, a_d.d_data, 2,
                        dpct::library_data_t::real_double, w_d.d_data,
                        device_ws_d, lwork_d, info);
  dpct::lapack::syheevd(
      *handle, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, 2,
      dpct::library_data_t::complex_float, a_c.d_data, 2,
      dpct::library_data_t::real_float, w_c.d_data, device_ws_c, lwork_c, info);
  dpct::lapack::syheevd(*handle, oneapi::mkl::job::vec,
                        oneapi::mkl::uplo::upper, 2,
                        dpct::library_data_t::complex_double, a_z.d_data, 2,
                        dpct::library_data_t::real_double, w_z.d_data,
                        device_ws_z, lwork_z, info);

  a_s.D2H();
  a_d.D2H();
  a_c.D2H();
  a_z.D2H();
  w_s.D2H();
  w_d.D2H();
  w_c.D2H();
  w_z.D2H();

  q_ct1.wait();

  /*
  DPCT1026:49: The call to cusolverDnDestroyParams was removed because this call
  is redundant in SYCL.
  */
  handle = nullptr;
  dpct::dpct_free(device_ws_s);
  dpct::dpct_free(device_ws_d);
  dpct::dpct_free(device_ws_c);
  dpct::dpct_free(device_ws_z);
  dpct::dpct_free(info);

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
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::vector<float> a = {1, 2, 2, 4};
  Data<float> a_s(a.data(), 4);
  Data<double> a_d(a.data(), 4);
  Data<sycl::float2> a_c(a.data(), 4);
  Data<sycl::double2> a_z(a.data(), 4);

  sycl::queue *handle;
  handle = &q_ct1;

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

  dpct::lapack::trtri_scratchpad_size(
      *handle, oneapi::mkl::uplo::upper, oneapi::mkl::diag::nonunit, 2,
      dpct::library_data_t::real_float, 2, &lwork_s, &lwork_host_s);
  dpct::lapack::trtri_scratchpad_size(
      *handle, oneapi::mkl::uplo::upper, oneapi::mkl::diag::nonunit, 2,
      dpct::library_data_t::real_double, 2, &lwork_d, &lwork_host_d);
  dpct::lapack::trtri_scratchpad_size(
      *handle, oneapi::mkl::uplo::upper, oneapi::mkl::diag::nonunit, 2,
      dpct::library_data_t::complex_float, 2, &lwork_c, &lwork_host_c);
  dpct::lapack::trtri_scratchpad_size(
      *handle, oneapi::mkl::uplo::upper, oneapi::mkl::diag::nonunit, 2,
      dpct::library_data_t::complex_double, 2, &lwork_z, &lwork_host_z);

  void* device_ws_s;
  void* device_ws_d;
  void* device_ws_c;
  void* device_ws_z;
  device_ws_s = dpct::dpct_malloc(lwork_s);
  device_ws_d = dpct::dpct_malloc(lwork_d);
  device_ws_c = dpct::dpct_malloc(lwork_c);
  device_ws_z = dpct::dpct_malloc(lwork_z);
  void* host_ws_s;
  void* host_ws_d;
  void* host_ws_c;
  void* host_ws_z;
  host_ws_s = malloc(lwork_host_s);
  host_ws_d = malloc(lwork_host_d);
  host_ws_c = malloc(lwork_host_c);
  host_ws_z = malloc(lwork_host_z);

  int *info;
  info = (int *)dpct::dpct_malloc(sizeof(int));

  dpct::lapack::trtri(*handle, oneapi::mkl::uplo::upper,
                      oneapi::mkl::diag::nonunit, 2,
                      dpct::library_data_t::real_float, a_s.d_data, 2,
                      device_ws_s, lwork_s, info);
  dpct::lapack::trtri(*handle, oneapi::mkl::uplo::upper,
                      oneapi::mkl::diag::nonunit, 2,
                      dpct::library_data_t::real_double, a_d.d_data, 2,
                      device_ws_d, lwork_d, info);
  dpct::lapack::trtri(*handle, oneapi::mkl::uplo::upper,
                      oneapi::mkl::diag::nonunit, 2,
                      dpct::library_data_t::complex_float, a_c.d_data, 2,
                      device_ws_c, lwork_c, info);
  dpct::lapack::trtri(*handle, oneapi::mkl::uplo::upper,
                      oneapi::mkl::diag::nonunit, 2,
                      dpct::library_data_t::complex_double, a_z.d_data, 2,
                      device_ws_z, lwork_z, info);

  a_s.D2H();
  a_d.D2H();
  a_c.D2H();
  a_z.D2H();

  q_ct1.wait();

  handle = nullptr;
  dpct::dpct_free(device_ws_s);
  dpct::dpct_free(device_ws_d);
  dpct::dpct_free(device_ws_c);
  dpct::dpct_free(device_ws_z);
  free(host_ws_s);
  free(host_ws_d);
  free(host_ws_c);
  free(host_ws_z);
  dpct::dpct_free(info);

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
  test_helper();
  test_cusolverDnTsygvd();
  test_cusolverDnThegvd();
#ifndef DPCT_USM_LEVEL_NONE
  test_cusolverDnTpotrfBatched();
  test_cusolverDnTpotrsBatched();
#endif
  test_cusolverDnTgesvdj();
  test_cusolverDnXgetrf();
  test_cusolverDnXgetrfnp();
  test_cusolverDnGetrf();
  test_cusolverDnXgetrs();
  test_cusolverDnGetrs();
  test_cusolverDnXgeqrf();
  test_cusolverDnGeqrf();
  test_cusolverDnXgesvd();
  test_cusolverDnGesvd();
  test_cusolverDnXpotrf();
  test_cusolverDnPotrf();
  test_cusolverDnXpotrs();
  test_cusolverDnPotrs();
  test_cusolverDnSyevdx();
  test_cusolverDnXsyevdx();
  test_cusolverDnTsyevdx_cusolverDnTheevdx();
  test_cusolverDnTsygvdx_cusolverDnThegvdx();
  test_cusolverDnTsygvj_cusolverDnThegvj();
  test_cusolverDnTsyevj_cusolverDnTheevj();
  test_cusolverDnXsyevd();
  test_cusolverDnSyevd();
#ifndef DPCT_USM_LEVEL_NONE
  test_cusolverDnXtrtri();
#endif

  if (test_passed)
    return 0;
  return -1;
}
