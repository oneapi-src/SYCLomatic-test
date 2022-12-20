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

int main() {
  test_helper();
  test_cusolverDnTsygvd();
  test_cusolverDnThegvd();
#ifndef DPCT_USM_LEVEL_NONE
  test_cusolverDnTpotrfBatched();
  test_cusolverDnTpotrsBatched();
#endif

  if (test_passed)
    return 0;
  return -1;
}
