// ====------ blas_extension_api_buffer.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#define DPCT_USM_LEVEL_NONE
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <oneapi/mkl.hpp>
#include <dpct/blas_utils.hpp>
#include <cmath>
#include <vector>
#include <cstdio>
#include <complex>
#include <oneapi/mkl/bfloat16.hpp>

#include <dpct/lib_common_utils.hpp>


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
    dpct::dpct_memcpy(d_data, h_temp, sizeof(d_data_t) * element_num, dpct::host_to_device);
    free(h_temp);
  }
  void D2H() {
    d_data_t* h_temp = (d_data_t*)malloc(sizeof(d_data_t) * element_num);
    memset(h_temp, 0, sizeof(d_data_t) * element_num);
    dpct::dpct_memcpy(h_temp, d_data, sizeof(d_data_t) * element_num, dpct::device_to_host);
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
inline void Data<sycl::float2>::from_float_convert(float* in, sycl::float2* out) {
  for (int i = 0; i < element_num; i++)
    out[i].x() = in[i];
}
template <>
inline void Data<sycl::double2>::from_float_convert(float* in, sycl::double2* out) {
  for (int i = 0; i < element_num; i++)
    out[i].x() = in[i];
}
template <>
inline void Data<std::complex<int8_t>>::from_float_convert(float* in, std::complex<int8_t>* out) {
  for (int i = 0; i < element_num; i++)
    reinterpret_cast<int8_t(&)[2]>(out[i])[0] = int8_t(in[i]);
}

template <>
inline void Data<sycl::float2>::to_float_convert(sycl::float2* in, float* out) {
  for (int i = 0; i < element_num; i++)
    out[i] = in[i].x();
}
template <>
inline void Data<sycl::double2>::to_float_convert(sycl::double2* in, float* out) {
  for (int i = 0; i < element_num; i++)
    out[i] = in[i].x();
}
template <>
inline void Data<sycl::half>::to_float_convert(sycl::half* in, float* out) {
  for (int i = 0; i < element_num; i++)
    out[i] = sycl::vec<sycl::half, 1>{in[i]}.convert<float, sycl::rounding_mode::automatic>()[0];
}
template <>
inline void Data<oneapi::mkl::bfloat16>::to_float_convert(oneapi::mkl::bfloat16* in, float* out) {
  for (int i = 0; i < element_num; i++)
    out[i] = in[i];
}

bool compare_result(float* expect, float* result, int element_num) {
  for (int i = 0; i < element_num; i++) {
    if (std::abs(result[i]-expect[i]) >= 0.05) {
      return false;
    }
  }
  return true;
}

bool test_passed = true;

void test_cublasNrm2Ex() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::cout << "Device Name: " << dev_ct1.get_info<sycl::info::device::name>() << std::endl;
  std::vector<float> v = {2, 3, 5, 7};
  Data<float> x_f(v.data(), 4);
  Data<double> x_d(v.data(), 4);
  Data<sycl::float2> x_f2(v.data(), 4);
  Data<sycl::double2> x_d2(v.data(), 4);

  Data<float> res_f(1);
  Data<double> res_d(1);
  Data<float> res_f2(1);
  Data<double> res_d2(1);
  Data<sycl::half> res_h(1);

  sycl::queue* handle;
  handle = &q_ct1;
  /*
  DPCT1026:0: The call to cublasSetPointerMode was removed because the function call is redundant in DPC++.
  */

  x_f.H2D();
  x_d.H2D();
  x_f2.H2D();
  x_d2.H2D();

  dpct::nrm2(*handle, 4, x_f.d_data, dpct::library_data_t::real_float, 1, res_f.d_data, dpct::library_data_t::real_float);
  dpct::nrm2(*handle, 4, x_d.d_data, dpct::library_data_t::real_double, 1, res_d.d_data, dpct::library_data_t::real_double);
  dpct::nrm2(*handle, 4, x_f2.d_data, dpct::library_data_t::complex_float, 1, res_f2.d_data, dpct::library_data_t::real_float);
  dpct::nrm2(*handle, 4, x_d2.d_data, dpct::library_data_t::complex_double, 1, res_d2.d_data, dpct::library_data_t::real_double);

  res_f.D2H();
  res_d.D2H();
  res_f2.D2H();
  res_d2.D2H();

  q_ct1.wait();

  handle = nullptr;

  float expect = 9.32738;
  if (compare_result(&expect, res_f.h_data, 1)
      && compare_result(&expect, res_d.h_data, 1)
      && compare_result(&expect, res_f2.h_data, 1)
      && compare_result(&expect, res_d2.h_data, 1))
    printf("Nrm2Ex pass\n");
  else {
    printf("Nrm2Ex fail\n");
    test_passed = false;
  }
}

void test_cublasDotEx() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::cout << "Device Name: " << dev_ct1.get_info<sycl::info::device::name>() << std::endl;
  std::vector<float> v = {2, 3, 5, 7};
  Data<float> x_f(v.data(), 4);
  Data<double> x_d(v.data(), 4);
  Data<sycl::float2> x_f2(v.data(), 4);
  Data<sycl::double2> x_d2(v.data(), 4);

  Data<float> y_f(v.data(), 4);
  Data<double> y_d(v.data(), 4);
  Data<sycl::float2> y_f2(v.data(), 4);
  Data<sycl::double2> y_d2(v.data(), 4);

  Data<float> res_f(1);
  Data<double> res_d(1);
  Data<sycl::float2> res_f2(1);
  Data<sycl::double2> res_d2(1);

  sycl::queue* handle;
  handle = &q_ct1;
  /*
  DPCT1026:1: The call to cublasSetPointerMode was removed because the function call is redundant in DPC++.
  */

  x_f.H2D();
  x_d.H2D();
  x_f2.H2D();
  x_d2.H2D();

  y_f.H2D();
  y_d.H2D();
  y_f2.H2D();
  y_d2.H2D();

  dpct::dot(*handle, 4, x_f.d_data, dpct::library_data_t::real_float, 1, y_f.d_data, dpct::library_data_t::real_float, 1, res_f.d_data, dpct::library_data_t::real_float);
  dpct::dot(*handle, 4, x_d.d_data, dpct::library_data_t::real_double, 1, y_d.d_data, dpct::library_data_t::real_double, 1, res_d.d_data, dpct::library_data_t::real_double);
  dpct::dot(*handle, 4, x_f2.d_data, dpct::library_data_t::complex_float, 1, y_f2.d_data, dpct::library_data_t::complex_float, 1, res_f2.d_data, dpct::library_data_t::complex_float);
  dpct::dot(*handle, 4, x_d2.d_data, dpct::library_data_t::complex_double, 1, y_d2.d_data, dpct::library_data_t::complex_double, 1, res_d2.d_data, dpct::library_data_t::complex_double);

  res_f.D2H();
  res_d.D2H();
  res_f2.D2H();
  res_d2.D2H();

  q_ct1.wait();

  handle = nullptr;

  float expect = 87;
  if (compare_result(&expect, res_f.h_data, 1)
      && compare_result(&expect, res_d.h_data, 1)
      && compare_result(&expect, res_f2.h_data, 1)
      && compare_result(&expect, res_d2.h_data, 1))
    printf("DotEx pass\n");
  else {
    printf("DotEx fail\n");
    test_passed = false;
  }
}

void test_cublasDotcEx() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::cout << "Device Name: " << dev_ct1.get_info<sycl::info::device::name>() << std::endl;
  std::vector<float> v = {2, 3, 5, 7};
  Data<float> x_f(v.data(), 4);
  Data<double> x_d(v.data(), 4);
  Data<sycl::float2> x_f2(v.data(), 4);
  Data<sycl::double2> x_d2(v.data(), 4);

  Data<float> y_f(v.data(), 4);
  Data<double> y_d(v.data(), 4);
  Data<sycl::float2> y_f2(v.data(), 4);
  Data<sycl::double2> y_d2(v.data(), 4);

  Data<float> res_f(1);
  Data<double> res_d(1);
  Data<sycl::float2> res_f2(1);
  Data<sycl::double2> res_d2(1);

  sycl::queue* handle;
  handle = &q_ct1;
  /*
  DPCT1026:2: The call to cublasSetPointerMode was removed because the function call is redundant in DPC++.
  */

  x_f.H2D();
  x_d.H2D();
  x_f2.H2D();
  x_d2.H2D();

  y_f.H2D();
  y_d.H2D();
  y_f2.H2D();
  y_d2.H2D();

  dpct::dotc(*handle, 4, x_f.d_data, dpct::library_data_t::real_float, 1, y_f.d_data, dpct::library_data_t::real_float, 1, res_f.d_data, dpct::library_data_t::real_float);
  dpct::dotc(*handle, 4, x_d.d_data, dpct::library_data_t::real_double, 1, y_d.d_data, dpct::library_data_t::real_double, 1, res_d.d_data, dpct::library_data_t::real_double);
  dpct::dotc(*handle, 4, x_f2.d_data, dpct::library_data_t::complex_float, 1, y_f2.d_data, dpct::library_data_t::complex_float, 1, res_f2.d_data, dpct::library_data_t::complex_float);
  dpct::dotc(*handle, 4, x_d2.d_data, dpct::library_data_t::complex_double, 1, y_d2.d_data, dpct::library_data_t::complex_double, 1, res_d2.d_data, dpct::library_data_t::complex_double);

  res_f.D2H();
  res_d.D2H();
  res_f2.D2H();
  res_d2.D2H();

  q_ct1.wait();

  handle = nullptr;

  float expect = 87;
  if (compare_result(&expect, res_f.h_data, 1)
      && compare_result(&expect, res_d.h_data, 1)
      && compare_result(&expect, res_f2.h_data, 1)
      && compare_result(&expect, res_d2.h_data, 1))
    printf("DotcEx pass\n");
  else {
    printf("DotcEx fail\n");
    test_passed = false;
  }
}

void test_cublasScalEx() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::cout << "Device Name: " << dev_ct1.get_info<sycl::info::device::name>() << std::endl;
  std::vector<float> v = {2, 3, 5, 7};
  Data<float> x_f(v.data(), 4);
  Data<double> x_d(v.data(), 4);
  Data<sycl::float2> x_f2(v.data(), 4);
  Data<sycl::double2> x_d2(v.data(), 4);

  float alpha = 10;
  Data<float> alpha_f(&alpha, 1);
  Data<double> alpha_d(&alpha, 1);
  Data<sycl::float2> alpha_f2(&alpha, 1);
  Data<sycl::double2> alpha_d2(&alpha, 1);

  sycl::queue* handle;
  handle = &q_ct1;
  /*
  DPCT1026:3: The call to cublasSetPointerMode was removed because the function call is redundant in DPC++.
  */

  x_f.H2D();
  x_d.H2D();
  x_f2.H2D();
  x_d2.H2D();

  alpha_f.H2D();
  alpha_d.H2D();
  alpha_f2.H2D();
  alpha_d2.H2D();

  dpct::scal(*handle, 4, alpha_f.d_data, dpct::library_data_t::real_float, x_f.d_data, dpct::library_data_t::real_float, 1);
  dpct::scal(*handle, 4, alpha_d.d_data, dpct::library_data_t::real_double, x_d.d_data, dpct::library_data_t::real_double, 1);
  dpct::scal(*handle, 4, alpha_f2.d_data, dpct::library_data_t::complex_float, x_f2.d_data, dpct::library_data_t::complex_float, 1);
  dpct::scal(*handle, 4, alpha_d2.d_data, dpct::library_data_t::complex_double, x_d2.d_data, dpct::library_data_t::complex_double, 1);

  x_f.D2H();
  x_d.D2H();
  x_f2.D2H();
  x_d2.D2H();

  q_ct1.wait();

  handle = nullptr;

  float expect[4] = {20, 30, 50, 70};
  if (compare_result(expect, x_f.h_data, 4)
      && compare_result(expect, x_d.h_data, 4)
      && compare_result(expect, x_f2.h_data, 4)
      && compare_result(expect, x_d2.h_data, 4))
    printf("ScalEx pass\n");
  else {
    printf("ScalEx fail\n");
    test_passed = false;
  }
}

void test_cublasAxpyEx() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::cout << "Device Name: " << dev_ct1.get_info<sycl::info::device::name>() << std::endl;
  std::vector<float> v = {2, 3, 5, 7};
  Data<float> x_f(v.data(), 4);
  Data<double> x_d(v.data(), 4);
  Data<sycl::float2> x_f2(v.data(), 4);
  Data<sycl::double2> x_d2(v.data(), 4);

  Data<float> y_f(v.data(), 4);
  Data<double> y_d(v.data(), 4);
  Data<sycl::float2> y_f2(v.data(), 4);
  Data<sycl::double2> y_d2(v.data(), 4);

  float alpha = 10;
  Data<float> alpha_f(&alpha, 1);
  Data<double> alpha_d(&alpha, 1);
  Data<sycl::float2> alpha_f2(&alpha, 1);
  Data<sycl::double2> alpha_d2(&alpha, 1);

  sycl::queue* handle;
  handle = &q_ct1;
  /*
  DPCT1026:4: The call to cublasSetPointerMode was removed because the function call is redundant in DPC++.
  */

  x_f.H2D();
  x_d.H2D();
  x_f2.H2D();
  x_d2.H2D();

  y_f.H2D();
  y_d.H2D();
  y_f2.H2D();
  y_d2.H2D();

  alpha_f.H2D();
  alpha_d.H2D();
  alpha_f2.H2D();
  alpha_d2.H2D();

  dpct::axpy(*handle, 4, alpha_f.d_data, dpct::library_data_t::real_float, x_f.d_data, dpct::library_data_t::real_float, 1, y_f.d_data, dpct::library_data_t::real_float, 1);
  dpct::axpy(*handle, 4, alpha_d.d_data, dpct::library_data_t::real_double, x_d.d_data, dpct::library_data_t::real_double, 1, y_d.d_data, dpct::library_data_t::real_double, 1);
  dpct::axpy(*handle, 4, alpha_f2.d_data, dpct::library_data_t::complex_float, x_f2.d_data, dpct::library_data_t::complex_float, 1, y_f2.d_data, dpct::library_data_t::complex_float, 1);
  dpct::axpy(*handle, 4, alpha_d2.d_data, dpct::library_data_t::complex_double, x_d2.d_data, dpct::library_data_t::complex_double, 1, y_d2.d_data, dpct::library_data_t::complex_double, 1);

  y_f.D2H();
  y_d.D2H();
  y_f2.D2H();
  y_d2.D2H();

  q_ct1.wait();

  handle = nullptr;

  float expect[4] = {22, 33, 55, 77};
  if (compare_result(expect, y_f.h_data, 4)
      && compare_result(expect, y_d.h_data, 4)
      && compare_result(expect, y_f2.h_data, 4)
      && compare_result(expect, y_d2.h_data, 4))
    printf("AxpyEx pass\n");
  else {
    printf("AxpyEx fail\n");
    test_passed = false;
  }
}

void test_cublasRotEx() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::cout << "Device Name: " << dev_ct1.get_info<sycl::info::device::name>() << std::endl;
  std::vector<float> v = {2, 3, 5, 7};
  Data<float>         x2(v.data(), 4);
  Data<double>        x3(v.data(), 4);
  Data<sycl::float2>        x4(v.data(), 4);
  Data<sycl::double2>       x6(v.data(), 4);

  Data<float>         y2(v.data(), 4);
  Data<double>        y3(v.data(), 4);
  Data<sycl::float2>        y4(v.data(), 4);
  Data<sycl::double2>       y6(v.data(), 4);

  float cos = 0.866;
  float sin = 0.5;
  Data<float>         cos2(&cos, 1);
  Data<double>        cos3(&cos, 1);
  Data<float>         cos4(&cos, 1);
  Data<double>        cos6(&cos, 1);

  Data<float>         sin2(&sin, 1);
  Data<double>        sin3(&sin, 1);
  Data<float>         sin4(&sin, 1);
  Data<double>        sin6(&sin, 1);

  sycl::queue* handle;
  handle = &q_ct1;
  /*
  DPCT1026:5: The call to cublasSetPointerMode was removed because the function call is redundant in DPC++.
  */

  x2.H2D();
  x3.H2D();
  x4.H2D();
  x6.H2D();

  y2.H2D();
  y3.H2D();
  y4.H2D();
  y6.H2D();

  sin2.H2D();
  sin3.H2D();
  sin4.H2D();
  sin6.H2D();

  cos2.H2D();
  cos3.H2D();
  cos4.H2D();
  cos6.H2D();

  dpct::rot(*handle, 4, x2.d_data, dpct::library_data_t::real_float, 1, y2.d_data, dpct::library_data_t::real_float, 1, cos2.d_data, sin2.d_data, dpct::library_data_t::real_float);
  dpct::rot(*handle, 4, x3.d_data, dpct::library_data_t::real_double, 1, y3.d_data, dpct::library_data_t::real_double, 1, cos3.d_data, sin3.d_data, dpct::library_data_t::real_double);
  dpct::rot(*handle, 4, x4.d_data, dpct::library_data_t::complex_float, 1, y4.d_data, dpct::library_data_t::complex_float, 1, cos4.d_data, sin4.d_data, dpct::library_data_t::real_float);
  dpct::rot(*handle, 4, x6.d_data, dpct::library_data_t::complex_double, 1, y6.d_data, dpct::library_data_t::complex_double, 1, cos6.d_data, sin6.d_data, dpct::library_data_t::real_double);

  x2.D2H();
  x3.D2H();
  x4.D2H();
  x6.D2H();

  y2.D2H();
  y3.D2H();
  y4.D2H();
  y6.D2H();

  q_ct1.wait();

  handle = nullptr;

  float expect_x[4] = {2.732000,4.098000,6.830000,9.562000};
  float expect_y[4] = {0.732000,1.098000,1.830000,2.562000};
  if (compare_result(expect_x, x2.h_data, 4) &&
      compare_result(expect_x, x3.h_data, 4) &&
      compare_result(expect_x, x4.h_data, 4) &&
      compare_result(expect_x, x6.h_data, 4) &&
      compare_result(expect_y, y2.h_data, 4) &&
      compare_result(expect_y, y3.h_data, 4) &&
      compare_result(expect_y, y4.h_data, 4) &&
      compare_result(expect_y, y6.h_data, 4))
    printf("RotEx pass\n");
  else {
    printf("RotEx fail\n");
    test_passed = false;
  }
}

void test_cublasGemmEx() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::cout << "Device Name: " << dev_ct1.get_info<sycl::info::device::name>() << std::endl;
  std::vector<float> v = {2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7};
  Data<sycl::half> a0(v.data(), 16);
  Data<sycl::half> a3(v.data(), 16);
  Data<oneapi::mkl::bfloat16> a5(v.data(), 16);
  Data<sycl::half> a6(v.data(), 16);
  Data<float> a7(v.data(), 16);
  Data<sycl::float2> a9(v.data(), 16);
  Data<double> a10(v.data(), 16);
  Data<sycl::double2> a11(v.data(), 16);

  Data<sycl::half> b0(v.data(), 16);
  Data<sycl::half> b3(v.data(), 16);
  Data<oneapi::mkl::bfloat16> b5(v.data(), 16);
  Data<sycl::half> b6(v.data(), 16);
  Data<float> b7(v.data(), 16);
  Data<sycl::float2> b9(v.data(), 16);
  Data<double> b10(v.data(), 16);
  Data<sycl::double2> b11(v.data(), 16);

  Data<sycl::half> c0(16);
  Data<sycl::half> c3(16);
  Data<float> c5(16);
  Data<float> c6(16);
  Data<float> c7(16);
  Data<sycl::float2> c9(16);
  Data<double> c10(16);
  Data<sycl::double2> c11(16);

  float alpha = 2;
  float beta = 0;
  Data<sycl::half> alpha0(&alpha, 1);
  Data<float> alpha3(&alpha, 1);
  Data<float> alpha5(&alpha, 1);
  Data<float> alpha6(&alpha, 1);
  Data<float> alpha7(&alpha, 1);
  Data<sycl::float2> alpha9(&alpha, 1);
  Data<double> alpha10(&alpha, 1);
  Data<sycl::double2> alpha11(&alpha, 1);

  Data<sycl::half> beta0(&beta, 1);
  Data<float> beta3(&beta, 1);
  Data<float> beta5(&beta, 1);
  Data<float> beta6(&beta, 1);
  Data<float> beta7(&beta, 1);
  Data<sycl::float2> beta9(&beta, 1);
  Data<double> beta10(&beta, 1);
  Data<sycl::double2> beta11(&beta, 1);

  sycl::queue* handle;
  handle = &q_ct1;
  /*
  DPCT1026:6: The call to cublasSetPointerMode was removed because the function call is redundant in DPC++.
  */

  a0.H2D();
  a3.H2D();
  a5.H2D();
  a6.H2D();
  a7.H2D();
  a9.H2D();
  a10.H2D();
  a11.H2D();

  b0.H2D();
  b3.H2D();
  b5.H2D();
  b6.H2D();
  b7.H2D();
  b9.H2D();
  b10.H2D();
  b11.H2D();

  alpha0.H2D();
  alpha3.H2D();
  alpha5.H2D();
  alpha6.H2D();
  alpha7.H2D();
  alpha9.H2D();
  alpha10.H2D();
  alpha11.H2D();

  beta0.H2D();
  beta3.H2D();
  beta5.H2D();
  beta6.H2D();
  beta7.H2D();
  beta9.H2D();
  beta10.H2D();
  beta11.H2D();

  dpct::gemm(*handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 4, 4, 4, alpha0.d_data, a0.d_data, dpct::library_data_t::real_half, 4, b0.d_data, dpct::library_data_t::real_half, 4, beta0.d_data, c0.d_data, dpct::library_data_t::real_half, 4, dpct::library_data_t::real_half);
  dpct::gemm(*handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 4, 4, 4, alpha3.d_data, a3.d_data, dpct::library_data_t::real_half, 4, b3.d_data, dpct::library_data_t::real_half, 4, beta3.d_data, c3.d_data, dpct::library_data_t::real_half, 4, dpct::library_data_t::real_float);
  dpct::gemm(*handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 4, 4, 4, alpha5.d_data, a5.d_data, dpct::library_data_t::real_bfloat16, 4, b5.d_data, dpct::library_data_t::real_bfloat16, 4, beta5.d_data, c5.d_data, dpct::library_data_t::real_float, 4, dpct::library_data_t::real_float);
  dpct::gemm(*handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 4, 4, 4, alpha6.d_data, a6.d_data, dpct::library_data_t::real_half, 4, b6.d_data, dpct::library_data_t::real_half, 4, beta6.d_data, c6.d_data, dpct::library_data_t::real_float, 4, dpct::library_data_t::real_float);
  dpct::gemm(*handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 4, 4, 4, alpha7.d_data, a7.d_data, dpct::library_data_t::real_float, 4, b7.d_data, dpct::library_data_t::real_float, 4, beta7.d_data, c7.d_data, dpct::library_data_t::real_float, 4, dpct::library_data_t::real_float);
  dpct::gemm(*handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 4, 4, 4, alpha9.d_data, a9.d_data, dpct::library_data_t::complex_float, 4, b9.d_data, dpct::library_data_t::complex_float, 4, beta9.d_data, c9.d_data, dpct::library_data_t::complex_float, 4, dpct::library_data_t::real_float);
  dpct::gemm(*handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 4, 4, 4, alpha10.d_data, a10.d_data, dpct::library_data_t::real_double, 4, b10.d_data, dpct::library_data_t::real_double, 4, beta10.d_data, c10.d_data, dpct::library_data_t::real_double, 4, dpct::library_data_t::real_double);
  dpct::gemm(*handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 4, 4, 4, alpha11.d_data, a11.d_data, dpct::library_data_t::complex_double, 4, b11.d_data, dpct::library_data_t::complex_double, 4, beta11.d_data, c11.d_data, dpct::library_data_t::complex_double, 4, dpct::library_data_t::real_double);

  c0.D2H();
  c3.D2H();
  c5.D2H();
  c6.D2H();
  c7.D2H();
  c9.D2H();
  c10.D2H();
  c11.D2H();

  q_ct1.wait();

  handle = nullptr;

  float expect[16] = { 68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0 };
  if (compare_result(expect, c0.h_data, 16) &&
      compare_result(expect, c3.h_data, 16) &&
      compare_result(expect, c5.h_data, 16) &&
      compare_result(expect, c6.h_data, 16) &&
      compare_result(expect, c7.h_data, 16) &&
      compare_result(expect, c9.h_data, 16) &&
      compare_result(expect, c10.h_data, 16) &&
      compare_result(expect, c11.h_data, 16))
    printf("GemmEx pass\n");
  else {
    printf("GemmEx fail\n");
    test_passed = false;
  }
}

void test_cublasTsyrkx() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::cout << "Device Name: " << dev_ct1.get_info<sycl::info::device::name>() << std::endl;
  std::vector<float> v = {2, 3, 5, 7, 11, 13};
  Data<float> a0(v.data(), 6);
  Data<double> a1(v.data(), 6);
  Data<sycl::float2> a2(v.data(), 6);
  Data<sycl::double2> a3(v.data(), 6);

  Data<float> b0(v.data(), 6);
  Data<double> b1(v.data(), 6);
  Data<sycl::float2> b2(v.data(), 6);
  Data<sycl::double2> b3(v.data(), 6);

  Data<float> c0(4);
  Data<double> c1(4);
  Data<sycl::float2> c2(4);
  Data<sycl::double2> c3(4);

  float alpha = 2;
  float beta = 0;
  Data<float> alpha0(&alpha, 1);
  Data<double> alpha1(&alpha, 1);
  Data<sycl::float2> alpha2(&alpha, 1);
  Data<sycl::double2> alpha3(&alpha, 1);

  Data<float> beta0(&beta, 1);
  Data<double> beta1(&beta, 1);
  Data<sycl::float2> beta2(&beta, 1);
  Data<sycl::double2> beta3(&beta, 1);

  sycl::queue* handle;
  handle = &q_ct1;
  /*
  DPCT1026:7: The call to cublasSetPointerMode was removed because the function call is redundant in DPC++.
  */

  a0.H2D();
  a1.H2D();
  a2.H2D();
  a3.H2D();

  b0.H2D();
  b1.H2D();
  b2.H2D();
  b3.H2D();

  alpha0.H2D();
  alpha1.H2D();
  alpha2.H2D();
  alpha3.H2D();

  beta0.H2D();
  beta1.H2D();
  beta2.H2D();
  beta3.H2D();

  dpct::syrk(*handle, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::conjtrans, 2, 3, alpha0.d_data, a0.d_data, 3, b0.d_data, 3, beta0.d_data, c0.d_data, 2);
  dpct::syrk(*handle, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::conjtrans, 2, 3, alpha1.d_data, a1.d_data, 3, b1.d_data, 3, beta1.d_data, c1.d_data, 2);
  dpct::syrk(*handle, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::conjtrans, 2, 3, alpha2.d_data, a2.d_data, 3, b2.d_data, 3, beta2.d_data, c2.d_data, 2);
  dpct::syrk(*handle, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::conjtrans, 2, 3, alpha3.d_data, a3.d_data, 3, b3.d_data, 3, beta3.d_data, c3.d_data, 2);

  c0.D2H();
  c1.D2H();
  c2.D2H();
  c3.D2H();

  q_ct1.wait();

  handle = nullptr;

  float expect[4] = { 76.000000,0.000000,224.000000,678.000000 };
  if (compare_result(expect, c0.h_data, 4) &&
      compare_result(expect, c1.h_data, 4) &&
      compare_result(expect, c2.h_data, 4) &&
      compare_result(expect, c3.h_data, 4))
    printf("Tsyrkx pass\n");
  else {
    printf("Tsyrkx fail\n");
    test_passed = false;
  }
}

void test_cublasTherkx() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::cout << "Device Name: " << dev_ct1.get_info<sycl::info::device::name>() << std::endl;
  std::vector<float> v = {2, 3, 5, 7, 11, 13};
  Data<sycl::float2> a0(v.data(), 6);
  Data<sycl::double2> a1(v.data(), 6);

  Data<sycl::float2> b0(v.data(), 6);
  Data<sycl::double2> b1(v.data(), 6);

  Data<sycl::float2> c0(4);
  Data<sycl::double2> c1(4);

  float alpha = 2;
  float beta = 0;
  Data<sycl::float2> alpha0(&alpha, 1);
  Data<sycl::double2> alpha1(&alpha, 1);

  Data<float> beta0(&beta, 1);
  Data<double> beta1(&beta, 1);

  sycl::queue* handle;
  handle = &q_ct1;
  /*
  DPCT1026:8: The call to cublasSetPointerMode was removed because the function call is redundant in DPC++.
  */

  a0.H2D();
  a1.H2D();

  b0.H2D();
  b1.H2D();

  alpha0.H2D();
  alpha1.H2D();

  beta0.H2D();
  beta1.H2D();

  dpct::herk(*handle, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::trans, 2, 3, alpha0.d_data, a0.d_data, 3, b0.d_data, 3, beta0.d_data, c0.d_data, 2);
  dpct::herk(*handle, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::trans, 2, 3, alpha1.d_data, a1.d_data, 3, b1.d_data, 3, beta1.d_data, c1.d_data, 2);

  c0.D2H();
  c1.D2H();

  q_ct1.wait();

  handle = nullptr;

  float expect[4] = { 76.000000,0.000000,224.000000,678.000000 };
  if (compare_result(expect, c0.h_data, 4) &&
      compare_result(expect, c1.h_data, 4))
    printf("Therkx pass\n");
  else {
    printf("Therkx fail\n");
    test_passed = false;
  }
}

void test_cublasTdgmm() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::cout << "Device Name: " << dev_ct1.get_info<sycl::info::device::name>() << std::endl;
  std::vector<float> v = {2, 3, 5, 7};
  Data<float> a0(v.data(), 4);
  Data<double> a1(v.data(), 4);
  Data<sycl::float2> a2(v.data(), 4);
  Data<sycl::double2> a3(v.data(), 4);

  std::vector<float> x = {2, 3};
  Data<float> x0(v.data(), 2);
  Data<double> x1(v.data(), 2);
  Data<sycl::float2> x2(v.data(), 2);
  Data<sycl::double2> x3(v.data(), 2);

  Data<float> c0(4);
  Data<double> c1(4);
  Data<sycl::float2> c2(4);
  Data<sycl::double2> c3(4);

  sycl::queue* handle;
  handle = &q_ct1;
  /*
  DPCT1026:9: The call to cublasSetPointerMode was removed because the function call is redundant in DPC++.
  */

  a0.H2D();
  a1.H2D();
  a2.H2D();
  a3.H2D();

  x0.H2D();
  x1.H2D();
  x2.H2D();
  x3.H2D();

  {
  auto a0_d_data_buf_ct1 = dpct::get_buffer<float>(a0.d_data);
  auto x0_d_data_buf_ct2 = dpct::get_buffer<float>(x0.d_data);
  auto c0_d_data_buf_ct3 = dpct::get_buffer<float>(c0.d_data);
  oneapi::mkl::blas::column_major::dgmm_batch(*handle, oneapi::mkl::side::left, 2, 2, a0_d_data_buf_ct1, 2, 0, x0_d_data_buf_ct2, 1, 0, c0_d_data_buf_ct3, 2, 2 * 2, 1);
  }
  {
  auto a1_d_data_buf_ct4 = dpct::get_buffer<double>(a1.d_data);
  auto x1_d_data_buf_ct5 = dpct::get_buffer<double>(x1.d_data);
  auto c1_d_data_buf_ct6 = dpct::get_buffer<double>(c1.d_data);
  oneapi::mkl::blas::column_major::dgmm_batch(*handle, oneapi::mkl::side::left, 2, 2, a1_d_data_buf_ct4, 2, 0, x1_d_data_buf_ct5, 1, 0, c1_d_data_buf_ct6, 2, 2 * 2, 1);
  }
  {
  auto a2_d_data_buf_ct7 = dpct::get_buffer<std::complex<float>>(a2.d_data);
  auto x2_d_data_buf_ct8 = dpct::get_buffer<std::complex<float>>(x2.d_data);
  auto c2_d_data_buf_ct9 = dpct::get_buffer<std::complex<float>>(c2.d_data);
  oneapi::mkl::blas::column_major::dgmm_batch(*handle, oneapi::mkl::side::left, 2, 2, a2_d_data_buf_ct7, 2, 0, x2_d_data_buf_ct8, 1, 0, c2_d_data_buf_ct9, 2, 2 * 2, 1);
  }
  {
  auto a3_d_data_buf_ct10 = dpct::get_buffer<std::complex<double>>(a3.d_data);
  auto x3_d_data_buf_ct11 = dpct::get_buffer<std::complex<double>>(x3.d_data);
  auto c3_d_data_buf_ct12 = dpct::get_buffer<std::complex<double>>(c3.d_data);
  oneapi::mkl::blas::column_major::dgmm_batch(*handle, oneapi::mkl::side::left, 2, 2, a3_d_data_buf_ct10, 2, 0, x3_d_data_buf_ct11, 1, 0, c3_d_data_buf_ct12, 2, 2 * 2, 1);
  }

  c0.D2H();
  c1.D2H();
  c2.D2H();
  c3.D2H();

  q_ct1.wait();

  handle = nullptr;

  float expect[4] = { 4.0, 9.0, 10.0, 21.0 };
  if (compare_result(expect, c0.h_data, 4) &&
      compare_result(expect, c1.h_data, 4) &&
      compare_result(expect, c2.h_data, 4) &&
      compare_result(expect, c3.h_data, 4))
    printf("Tdgmm pass\n");
  else {
    printf("Tdgmm fail\n");
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
    d_data = (void **)dpct::dpct_malloc(group_num * sizeof(void*));
    dpct::dpct_memset(d_data, 0, group_num * sizeof(void*));
  }
  ~Ptr_Data() {
    free(h_data);
    dpct::dpct_free(d_data);
  }
  void H2D() {
    dpct::dpct_memcpy(d_data, h_data, group_num * sizeof(void*), dpct::host_to_device);
  }
};

#ifndef DPCT_USM_LEVEL_NONE
void test_cublasGemmBatchedEx() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::cout << "Device Name: " << dev_ct1.get_info<sycl::info::device::name>() << std::endl;
  std::vector<float> v = {2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7,
                          2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7};
  Data<sycl::half> a0(v.data(), 32);
  Data<float> a7(v.data(), 32);
  Data<sycl::float2> a9(v.data(), 32);
  Data<double> a10(v.data(), 32);
  Data<sycl::double2> a11(v.data(), 32);

  Ptr_Data a0_ptrs(2);  a0_ptrs.h_data[0] = a0.d_data; a0_ptrs.h_data[1] = a0.d_data + 16;
  Ptr_Data a7_ptrs(2);  a7_ptrs.h_data[0] = a7.d_data; a7_ptrs.h_data[1] = a7.d_data + 16;
  Ptr_Data a9_ptrs(2);  a9_ptrs.h_data[0] = a9.d_data; a9_ptrs.h_data[1] = a9.d_data + 16;
  Ptr_Data a10_ptrs(2); a10_ptrs.h_data[0] = a10.d_data; a10_ptrs.h_data[1] = a10.d_data + 16;
  Ptr_Data a11_ptrs(2); a11_ptrs.h_data[0] = a11.d_data; a11_ptrs.h_data[1] = a11.d_data + 16;

  Data<sycl::half> b0(v.data(), 32);
  Data<float> b7(v.data(), 32);
  Data<sycl::float2> b9(v.data(), 32);
  Data<double> b10(v.data(), 32);
  Data<sycl::double2> b11(v.data(), 32);

  Ptr_Data b0_ptrs(2);  b0_ptrs.h_data[0] = b0.d_data; b0_ptrs.h_data[1] = b0.d_data + 16;
  Ptr_Data b7_ptrs(2);  b7_ptrs.h_data[0] = b7.d_data; b7_ptrs.h_data[1] = b7.d_data + 16;
  Ptr_Data b9_ptrs(2);  b9_ptrs.h_data[0] = b9.d_data; b9_ptrs.h_data[1] = b9.d_data + 16;
  Ptr_Data b10_ptrs(2); b10_ptrs.h_data[0] = b10.d_data; b10_ptrs.h_data[1] = b10.d_data + 16;
  Ptr_Data b11_ptrs(2); b11_ptrs.h_data[0] = b11.d_data; b11_ptrs.h_data[1] = b11.d_data + 16;

  Data<sycl::half> c0(32);
  Data<float> c7(32);
  Data<sycl::float2> c9(32);
  Data<double> c10(32);
  Data<sycl::double2> c11(32);

  Ptr_Data c0_ptrs(2);  c0_ptrs.h_data[0] = c0.d_data; c0_ptrs.h_data[1] = c0.d_data + 16;
  Ptr_Data c7_ptrs(2);  c7_ptrs.h_data[0] = c7.d_data; c7_ptrs.h_data[1] = c7.d_data + 16;
  Ptr_Data c9_ptrs(2);  c9_ptrs.h_data[0] = c9.d_data; c9_ptrs.h_data[1] = c9.d_data + 16;
  Ptr_Data c10_ptrs(2); c10_ptrs.h_data[0] = c10.d_data; c10_ptrs.h_data[1] = c10.d_data + 16;
  Ptr_Data c11_ptrs(2); c11_ptrs.h_data[0] = c11.d_data; c11_ptrs.h_data[1] = c11.d_data + 16; 

  float alpha = 2;
  float beta = 0;
  Data<sycl::half> alpha0(&alpha, 1);
  Data<float> alpha7(&alpha, 1);
  Data<sycl::float2> alpha9(&alpha, 1);
  Data<double> alpha10(&alpha, 1);
  Data<sycl::double2> alpha11(&alpha, 1);

  Data<sycl::half> beta0(&beta, 1);
  Data<float> beta7(&beta, 1);
  Data<sycl::float2> beta9(&beta, 1);
  Data<double> beta10(&beta, 1);
  Data<sycl::double2> beta11(&beta, 1);

  sycl::queue* handle;
  handle = &q_ct1;
  /*
  DPCT1026:10: The call to cublasSetPointerMode was removed because the function call is redundant in DPC++.
  */

  a0.H2D();
  a7.H2D();
  a9.H2D();
  a10.H2D();
  a11.H2D();

  b0.H2D();
  b7.H2D();
  b9.H2D();
  b10.H2D();
  b11.H2D();

  a0_ptrs.H2D();
  a7_ptrs.H2D();
  a9_ptrs.H2D();
  a10_ptrs.H2D();
  a11_ptrs.H2D();

  b0_ptrs.H2D();
  b7_ptrs.H2D();
  b9_ptrs.H2D();
  b10_ptrs.H2D();
  b11_ptrs.H2D();

  c0_ptrs.H2D();
  c7_ptrs.H2D();
  c9_ptrs.H2D();
  c10_ptrs.H2D();
  c11_ptrs.H2D();

  alpha0.H2D();
  alpha7.H2D();
  alpha9.H2D();
  alpha10.H2D();
  alpha11.H2D();

  beta0.H2D();
  beta7.H2D();
  beta9.H2D();
  beta10.H2D();
  beta11.H2D();

  /*
  DPCT1007:11: Migration of cublasGemmBatchedEx is not supported.
  */
  cublasGemmBatchedEx(handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 4, 4, 4, alpha0.d_data, (const void**)a0_ptrs.d_data, dpct::library_data_t::real_half, 4, (const void**)b0_ptrs.d_data, dpct::library_data_t::real_half, 4, beta0.d_data, c0_ptrs.d_data, dpct::library_data_t::real_half, 4, 2, CUBLAS_COMPUTE_16F, -1);
  /*
  DPCT1007:12: Migration of cublasGemmBatchedEx is not supported.
  */
  cublasGemmBatchedEx(handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 4, 4, 4, alpha7.d_data, (const void**)a7_ptrs.d_data, dpct::library_data_t::real_float, 4, (const void**)b7_ptrs.d_data, dpct::library_data_t::real_float, 4, beta7.d_data, c7_ptrs.d_data, dpct::library_data_t::real_float, 4, 2, CUBLAS_COMPUTE_32F, -1);
  /*
  DPCT1007:13: Migration of cublasGemmBatchedEx is not supported.
  */
  cublasGemmBatchedEx(handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 4, 4, 4, alpha9.d_data, (const void**)a9_ptrs.d_data, dpct::library_data_t::complex_float, 4, (const void**)b9_ptrs.d_data, dpct::library_data_t::complex_float, 4, beta9.d_data, c9_ptrs.d_data, dpct::library_data_t::complex_float, 4, 2, CUBLAS_COMPUTE_32F, -1);
  /*
  DPCT1007:14: Migration of cublasGemmBatchedEx is not supported.
  */
  cublasGemmBatchedEx(handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 4, 4, 4, alpha10.d_data, (const void**)a10_ptrs.d_data, dpct::library_data_t::real_double, 4, (const void**)b10_ptrs.d_data, dpct::library_data_t::real_double, 4, beta10.d_data, c10_ptrs.d_data, dpct::library_data_t::real_double, 4, 2, CUBLAS_COMPUTE_64F, -1);
  /*
  DPCT1007:15: Migration of cublasGemmBatchedEx is not supported.
  */
  cublasGemmBatchedEx(handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 4, 4, 4, alpha11.d_data, (const void**)a11_ptrs.d_data, dpct::library_data_t::complex_double, 4, (const void**)b11_ptrs.d_data, dpct::library_data_t::complex_double, 4, beta11.d_data, c11_ptrs.d_data, dpct::library_data_t::complex_double, 4, 2, CUBLAS_COMPUTE_64F, -1);

  c0.D2H();
  c7.D2H();
  c9.D2H();
  c10.D2H();
  c11.D2H();

  q_ct1.wait();

  handle = nullptr;

  float expect[32] = { 68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0,
                       68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0 };
  if (compare_result(expect, c0.h_data, 32) &&
      compare_result(expect, c7.h_data, 32) &&
      compare_result(expect, c9.h_data, 32) &&
      compare_result(expect, c10.h_data, 32) &&
      compare_result(expect, c11.h_data, 32))
    printf("GemmBatchedEx pass\n");
  else {
    printf("GemmBatchedEx fail\n");
    test_passed = false;
  }
}
#endif

void test_cublasGemmStridedBatchedEx() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::cout << "Device Name: " << dev_ct1.get_info<sycl::info::device::name>() << std::endl;
  std::vector<float> v = {2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7,
                          2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7};
  std::vector<float> v2 = {2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7,
                            2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7};
  Data<sycl::half> a0(v.data(), 32);
  Data<float> a7(v.data(), 32);
  Data<sycl::float2> a9(v.data(), 32);
  Data<double> a10(v.data(), 32);
  Data<sycl::double2> a11(v.data(), 32);

  Data<sycl::half> b0(v.data(), 32);
  Data<float> b7(v.data(), 32);
  Data<sycl::float2> b9(v.data(), 32);
  Data<double> b10(v.data(), 32);
  Data<sycl::double2> b11(v.data(), 32);

  Data<sycl::half> c0(32);
  Data<float> c7(32);
  Data<sycl::float2> c9(32);
  Data<double> c10(32);
  Data<sycl::double2> c11(32);

  float alpha = 2;
  float beta = 0;
  Data<sycl::half> alpha0(&alpha, 1);
  Data<float> alpha7(&alpha, 1);
  Data<sycl::float2> alpha9(&alpha, 1);
  Data<double> alpha10(&alpha, 1);
  Data<sycl::double2> alpha11(&alpha, 1);

  Data<sycl::half> beta0(&beta, 1);
  Data<float> beta7(&beta, 1);
  Data<sycl::float2> beta9(&beta, 1);
  Data<double> beta10(&beta, 1);
  Data<sycl::double2> beta11(&beta, 1);

  sycl::queue* handle;
  handle = &q_ct1;
  /*
  DPCT1026:16: The call to cublasSetPointerMode was removed because the function call is redundant in DPC++.
  */

  a0.H2D();
  a7.H2D();
  a9.H2D();
  a10.H2D();
  a11.H2D();

  b0.H2D();
  b7.H2D();
  b9.H2D();
  b10.H2D();
  b11.H2D();

  alpha0.H2D();
  alpha7.H2D();
  alpha9.H2D();
  alpha10.H2D();
  alpha11.H2D();

  beta0.H2D();
  beta7.H2D();
  beta9.H2D();
  beta10.H2D();
  beta11.H2D();

  dpct::gemm_batch(*handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 4, 4, 4, alpha0.d_data, a0.d_data, dpct::library_data_t::real_half, 4, 16, b0.d_data, dpct::library_data_t::real_half, 4, 16, beta0.d_data, c0.d_data, dpct::library_data_t::real_half, 4, 16, 2, dpct::library_data_t::real_half);
  dpct::gemm_batch(*handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 4, 4, 4, alpha7.d_data, a7.d_data, dpct::library_data_t::real_float, 4, 16, b7.d_data, dpct::library_data_t::real_float, 4, 16, beta7.d_data, c7.d_data, dpct::library_data_t::real_float, 4, 16, 2, dpct::library_data_t::real_float);
  dpct::gemm_batch(*handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 4, 4, 4, alpha9.d_data, a9.d_data, dpct::library_data_t::complex_float, 4, 16, b9.d_data, dpct::library_data_t::complex_float, 4, 16, beta9.d_data, c9.d_data, dpct::library_data_t::complex_float, 4, 16, 2, dpct::library_data_t::real_float);
  dpct::gemm_batch(*handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 4, 4, 4, alpha10.d_data, a10.d_data, dpct::library_data_t::real_double, 4, 16, b10.d_data, dpct::library_data_t::real_double, 4, 16, beta10.d_data, c10.d_data, dpct::library_data_t::real_double, 4, 16, 2, dpct::library_data_t::real_double);
  dpct::gemm_batch(*handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 4, 4, 4, alpha11.d_data, a11.d_data, dpct::library_data_t::complex_double, 4, 16, b11.d_data, dpct::library_data_t::complex_double, 4, 16, beta11.d_data, c11.d_data, dpct::library_data_t::complex_double, 4, 16, 2, dpct::library_data_t::real_double);

  c0.D2H();
  c7.D2H();
  c9.D2H();
  c10.D2H();
  c11.D2H();

  q_ct1.wait();

  handle = nullptr;

  float expect[32] = { 68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0,
                       68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0 };
  if (compare_result(expect, c0.h_data, 32) &&
      compare_result(expect, c7.h_data, 32) &&
      compare_result(expect, c9.h_data, 32) &&
      compare_result(expect, c10.h_data, 32) &&
      compare_result(expect, c11.h_data, 32))
    printf("GemmStridedBatchedEx pass\n");
  else {
    printf("GemmStridedBatchedEx fail\n");
    test_passed = false;
  }
}

#ifndef DPCT_USM_LEVEL_NONE
void test_cublasTgemmBatched() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::cout << "Device Name: " << dev_ct1.get_info<sycl::info::device::name>() << std::endl;
  std::vector<float> v = {2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7,
                          2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7};
  Data<sycl::half> a0(v.data(), 32);
  Data<float> a1(v.data(), 32);
  Data<sycl::float2> a2(v.data(), 32);
  Data<double> a3(v.data(), 32);
  Data<sycl::double2> a4(v.data(), 32);

  Ptr_Data a0_ptrs(2);  a0_ptrs.h_data[0] = a0.d_data; a0_ptrs.h_data[1] = a0.d_data + 16;
  Ptr_Data a1_ptrs(2);  a1_ptrs.h_data[0] = a1.d_data; a1_ptrs.h_data[1] = a1.d_data + 16;
  Ptr_Data a2_ptrs(2);  a2_ptrs.h_data[0] = a2.d_data; a2_ptrs.h_data[1] = a2.d_data + 16;
  Ptr_Data a3_ptrs(2); a3_ptrs.h_data[0] = a3.d_data; a3_ptrs.h_data[1] = a3.d_data + 16;
  Ptr_Data a4_ptrs(2); a4_ptrs.h_data[0] = a4.d_data; a4_ptrs.h_data[1] = a4.d_data + 16;

  Data<sycl::half> b0(v.data(), 32);
  Data<float> b1(v.data(), 32);
  Data<sycl::float2> b2(v.data(), 32);
  Data<double> b3(v.data(), 32);
  Data<sycl::double2> b4(v.data(), 32);

  Ptr_Data b0_ptrs(2);  b0_ptrs.h_data[0] = b0.d_data; b0_ptrs.h_data[1] = b0.d_data + 16;
  Ptr_Data b1_ptrs(2);  b1_ptrs.h_data[0] = b1.d_data; b1_ptrs.h_data[1] = b1.d_data + 16;
  Ptr_Data b2_ptrs(2);  b2_ptrs.h_data[0] = b2.d_data; b2_ptrs.h_data[1] = b2.d_data + 16;
  Ptr_Data b3_ptrs(2); b3_ptrs.h_data[0] = b3.d_data; b3_ptrs.h_data[1] = b3.d_data + 16;
  Ptr_Data b4_ptrs(2); b4_ptrs.h_data[0] = b4.d_data; b4_ptrs.h_data[1] = b4.d_data + 16;

  Data<sycl::half> c0(32);
  Data<float> c1(32);
  Data<sycl::float2> c2(32);
  Data<double> c3(32);
  Data<sycl::double2> c4(32);

  Ptr_Data c0_ptrs(2);  c0_ptrs.h_data[0] = c0.d_data; c0_ptrs.h_data[1] = c0.d_data + 16;
  Ptr_Data c1_ptrs(2);  c1_ptrs.h_data[0] = c1.d_data; c1_ptrs.h_data[1] = c1.d_data + 16;
  Ptr_Data c2_ptrs(2);  c2_ptrs.h_data[0] = c2.d_data; c2_ptrs.h_data[1] = c2.d_data + 16;
  Ptr_Data c3_ptrs(2); c3_ptrs.h_data[0] = c3.d_data; c3_ptrs.h_data[1] = c3.d_data + 16;
  Ptr_Data c4_ptrs(2); c4_ptrs.h_data[0] = c4.d_data; c4_ptrs.h_data[1] = c4.d_data + 16;

  float alpha = 2;
  float beta = 0;
  Data<sycl::half> alpha0(&alpha, 1);
  Data<float> alpha1(&alpha, 1);
  Data<sycl::float2> alpha2(&alpha, 1);
  Data<double> alpha3(&alpha, 1);
  Data<sycl::double2> alpha4(&alpha, 1);

  Data<sycl::half> beta0(&beta, 1);
  Data<float> beta1(&beta, 1);
  Data<sycl::float2> beta2(&beta, 1);
  Data<double> beta3(&beta, 1);
  Data<sycl::double2> beta4(&beta, 1);

  sycl::queue* handle;
  handle = &q_ct1;
  /*
  DPCT1026:17: The call to cublasSetPointerMode was removed because the function call is redundant in DPC++.
  */

  a0.H2D();
  a1.H2D();
  a2.H2D();
  a3.H2D();
  a4.H2D();

  b0.H2D();
  b1.H2D();
  b2.H2D();
  b3.H2D();
  b4.H2D();

  a0_ptrs.H2D();
  a1_ptrs.H2D();
  a2_ptrs.H2D();
  a3_ptrs.H2D();
  a4_ptrs.H2D();

  b0_ptrs.H2D();
  b1_ptrs.H2D();
  b2_ptrs.H2D();
  b3_ptrs.H2D();
  b4_ptrs.H2D();

  c0_ptrs.H2D();
  c1_ptrs.H2D();
  c2_ptrs.H2D();
  c3_ptrs.H2D();
  c4_ptrs.H2D();

  alpha0.H2D();
  alpha1.H2D();
  alpha2.H2D();
  alpha3.H2D();
  alpha4.H2D();

  beta0.H2D();
  beta1.H2D();
  beta2.H2D();
  beta3.H2D();
  beta4.H2D();

  /*
  DPCT1007:18: Migration of cublasHgemmBatched is not supported.
  */
  cublasHgemmBatched(handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 4, 4, 4, alpha0.d_data, (const sycl::half**)a0_ptrs.d_data, 4, (const sycl::half**)b0_ptrs.d_data, 4, beta0.d_data, (sycl::half**)c0_ptrs.d_data, 4, 2);
  /*
  DPCT1007:19: Migration of cublasSgemmBatched is not supported.
  */
  cublasSgemmBatched(handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 4, 4, 4, alpha1.d_data, (const float**)a1_ptrs.d_data, 4, (const float**)b1_ptrs.d_data, 4, beta1.d_data, (float**)c1_ptrs.d_data, 4, 2);
  /*
  DPCT1007:20: Migration of cublasCgemmBatched is not supported.
  */
  cublasCgemmBatched(handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 4, 4, 4, alpha2.d_data, (const sycl::float2**)a2_ptrs.d_data, 4, (const sycl::float2**)b2_ptrs.d_data, 4, beta2.d_data, (sycl::float2**)c2_ptrs.d_data, 4, 2);
  /*
  DPCT1007:21: Migration of cublasDgemmBatched is not supported.
  */
  cublasDgemmBatched(handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 4, 4, 4, alpha3.d_data, (const double**)a3_ptrs.d_data, 4, (const double**)b3_ptrs.d_data, 4, beta3.d_data, (double**)c3_ptrs.d_data, 4, 2);
  /*
  DPCT1007:22: Migration of cublasZgemmBatched is not supported.
  */
  cublasZgemmBatched(handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 4, 4, 4, alpha4.d_data, (const sycl::double2**)a4_ptrs.d_data, 4, (const sycl::double2**)b4_ptrs.d_data, 4, beta4.d_data, (sycl::double2**)c4_ptrs.d_data, 4, 2);

  c0.D2H();
  c1.D2H();
  c2.D2H();
  c3.D2H();
  c4.D2H();

  q_ct1.wait();

  handle = nullptr;

  float expect[32] = { 68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0,
                       68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0, 68.0, 102.0, 170.0, 238.0 };
  if (compare_result(expect, c0.h_data, 32) &&
      compare_result(expect, c1.h_data, 32) &&
      compare_result(expect, c2.h_data, 32) &&
      compare_result(expect, c3.h_data, 32) &&
      compare_result(expect, c4.h_data, 32))
    printf("TgemmBatched pass\n");
  else {
    printf("TgemmBatched fail\n");
    test_passed = false;
  }
}
#endif

#ifndef DPCT_USM_LEVEL_NONE
void test_cublasTtrsmBatched() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::cout << "Device Name: " << dev_ct1.get_info<sycl::info::device::name>() << std::endl;
  std::vector<float> v = {2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7,
                          2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7};
  Data<float> a1(v.data(), 32);
  Data<sycl::float2> a2(v.data(), 32);
  Data<double> a3(v.data(), 32);
  Data<sycl::double2> a4(v.data(), 32);

  Ptr_Data a1_ptrs(2);  a1_ptrs.h_data[0] = a1.d_data; a1_ptrs.h_data[1] = a1.d_data + 16;
  Ptr_Data a2_ptrs(2);  a2_ptrs.h_data[0] = a2.d_data; a2_ptrs.h_data[1] = a2.d_data + 16;
  Ptr_Data a3_ptrs(2); a3_ptrs.h_data[0] = a3.d_data; a3_ptrs.h_data[1] = a3.d_data + 16;
  Ptr_Data a4_ptrs(2); a4_ptrs.h_data[0] = a4.d_data; a4_ptrs.h_data[1] = a4.d_data + 16;

  Data<float> b1(v.data(), 32);
  Data<sycl::float2> b2(v.data(), 32);
  Data<double> b3(v.data(), 32);
  Data<sycl::double2> b4(v.data(), 32);

  Ptr_Data b1_ptrs(2);  b1_ptrs.h_data[0] = b1.d_data; b1_ptrs.h_data[1] = b1.d_data + 16;
  Ptr_Data b2_ptrs(2);  b2_ptrs.h_data[0] = b2.d_data; b2_ptrs.h_data[1] = b2.d_data + 16;
  Ptr_Data b3_ptrs(2); b3_ptrs.h_data[0] = b3.d_data; b3_ptrs.h_data[1] = b3.d_data + 16;
  Ptr_Data b4_ptrs(2); b4_ptrs.h_data[0] = b4.d_data; b4_ptrs.h_data[1] = b4.d_data + 16;

  float alpha = 2;
  Data<float> alpha1(&alpha, 1);
  Data<sycl::float2> alpha2(&alpha, 1);
  Data<double> alpha3(&alpha, 1);
  Data<sycl::double2> alpha4(&alpha, 1);

  sycl::queue* handle;
  handle = &q_ct1;
  /*
  DPCT1026:23: The call to cublasSetPointerMode was removed because the function call is redundant in DPC++.
  */

  a1.H2D();
  a2.H2D();
  a3.H2D();
  a4.H2D();

  b1.H2D();
  b2.H2D();
  b3.H2D();
  b4.H2D();

  a1_ptrs.H2D();
  a2_ptrs.H2D();
  a3_ptrs.H2D();
  a4_ptrs.H2D();

  b1_ptrs.H2D();
  b2_ptrs.H2D();
  b3_ptrs.H2D();
  b4_ptrs.H2D();

  alpha1.H2D();
  alpha2.H2D();
  alpha3.H2D();
  alpha4.H2D();

  /*
  DPCT1007:24: Migration of cublasStrsmBatched is not supported.
  */
  cublasStrsmBatched(handle, oneapi::mkl::side::left, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, 4, 4, alpha1.d_data, (const float**)a1_ptrs.d_data, 4, (float**)b1_ptrs.d_data, 4, 2);
  /*
  DPCT1007:25: Migration of cublasCtrsmBatched is not supported.
  */
  cublasCtrsmBatched(handle, oneapi::mkl::side::left, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, 4, 4, alpha2.d_data, (const sycl::float2**)a2_ptrs.d_data, 4, (sycl::float2**)b2_ptrs.d_data, 4, 2);
  /*
  DPCT1007:26: Migration of cublasDtrsmBatched is not supported.
  */
  cublasDtrsmBatched(handle, oneapi::mkl::side::left, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, 4, 4, alpha3.d_data, (const double**)a3_ptrs.d_data, 4, (double**)b3_ptrs.d_data, 4, 2);
  /*
  DPCT1007:27: Migration of cublasZtrsmBatched is not supported.
  */
  cublasZtrsmBatched(handle, oneapi::mkl::side::left, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, 4, 4, alpha4.d_data, (const sycl::double2**)a4_ptrs.d_data, 4, (sycl::double2**)b4_ptrs.d_data, 4, 2);

  b1.D2H();
  b2.D2H();
  b3.D2H();
  b4.D2H();

  q_ct1.wait();

  handle = nullptr;

  float expect[32] = { 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0,
                       0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0 };
  if (compare_result(expect, b1.h_data, 32) &&
      compare_result(expect, b2.h_data, 32) &&
      compare_result(expect, b3.h_data, 32) &&
      compare_result(expect, b4.h_data, 32))
    printf("TtrsmBatched pass\n");
  else {
    printf("TtrsmBatched fail\n");
    test_passed = false;
  }
}
#endif

void test_cublasTtrmm() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::cout << "Device Name: " << dev_ct1.get_info<sycl::info::device::name>() << std::endl;
  std::vector<float> v = {2, 3, 5, 7};
  Data<float> a0(v.data(), 4);
  Data<double> a1(v.data(), 4);
  Data<sycl::float2> a2(v.data(), 4);
  Data<sycl::double2> a3(v.data(), 4);

  Data<float> b0(v.data(), 4);
  Data<double> b1(v.data(), 4);

  Data<float> c0(4);
  Data<double> c1(4);
  Data<sycl::float2> c2(v.data(), 4);
  Data<sycl::double2> c3(v.data(), 4);

  sycl::queue* handle;
  handle = &q_ct1;
  /*
  DPCT1026:28: The call to cublasSetPointerMode was removed because the function call is redundant in DPC++.
  */

  a0.H2D();
  a1.H2D();
  a2.H2D();
  a3.H2D();

  b0.H2D();
  b1.H2D();

  c2.H2D();
  c3.H2D();

  float alpha = 2;
  Data<float> alpha0(&alpha, 1);
  Data<double> alpha1(&alpha, 1);
  Data<sycl::float2> alpha2(&alpha, 1);
  Data<sycl::double2> alpha3(&alpha, 1);

  alpha0.H2D();
  alpha1.H2D();
  alpha2.H2D();
  alpha3.H2D();

  dpct::trmm(*handle, oneapi::mkl::side::left, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, 2, 2, alpha0.d_data, a0.d_data, 2, b0.d_data, 2, c0.d_data, 2);
  dpct::trmm(*handle, oneapi::mkl::side::left, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, 2, 2, alpha1.d_data, a1.d_data, 2, b1.d_data, 2, c1.d_data, 2);
  dpct::trmm(*handle, oneapi::mkl::side::left, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, 2, 2, alpha2.d_data, a2.d_data, 2, c2.d_data, 2, c2.d_data, 2);
  dpct::trmm(*handle, oneapi::mkl::side::left, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, 2, 2, alpha3.d_data, a3.d_data, 2, c3.d_data, 2, c3.d_data, 2);

  c0.D2H();
  c1.D2H();
  c2.D2H();
  c3.D2H();

  q_ct1.wait();

  handle = nullptr;

  float expect[4] = { 38.0, 42.0, 90.0, 98.0 };
  if (compare_result(expect, c0.h_data, 4) &&
      compare_result(expect, c1.h_data, 4) &&
      compare_result(expect, c2.h_data, 4) &&
      compare_result(expect, c3.h_data, 4))
    printf("Ttrmm pass\n");
  else {
    printf("Ttrmm fail\n");
    test_passed = false;
  }
}

int main() {
  test_cublasNrm2Ex();
  test_cublasDotEx();
  test_cublasDotcEx();
  test_cublasScalEx();
  test_cublasAxpyEx();
  test_cublasRotEx();
  test_cublasGemmEx();
  test_cublasTsyrkx();
  test_cublasTherkx();
  test_cublasTdgmm();
#ifndef DPCT_USM_LEVEL_NONE
  test_cublasGemmBatchedEx();
#endif
  test_cublasGemmStridedBatchedEx();
#ifndef DPCT_USM_LEVEL_NONE
  test_cublasTgemmBatched();
  test_cublasTtrsmBatched();
#endif
  test_cublasTtrmm();

  if (test_passed)
    return 0;
  return -1;
}
