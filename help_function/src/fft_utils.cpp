// ===------- fft_utils.cpp ----------------------------- *- C++ -* ------=== //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <oneapi/mkl.hpp>
#include <dpct/fft_utils.hpp>

// This is a helper function interface test. So we are not care the data.
int main() {
  dpct::fft::fft_dir fft_dir_0 = dpct::fft::fft_dir::forward;
  dpct::fft::fft_dir fft_dir_1 = dpct::fft::fft_dir::backward;

  dpct::fft::fft_type fft_type_0 = dpct::fft::fft_type::real_float_to_complex_float;
  dpct::fft::fft_type fft_type_1 = dpct::fft::fft_type::complex_float_to_real_float;
  dpct::fft::fft_type fft_type_2 = dpct::fft::fft_type::real_double_to_complex_double;
  dpct::fft::fft_type fft_type_3 = dpct::fft::fft_type::complex_double_to_real_double;
  dpct::fft::fft_type fft_type_4 = dpct::fft::fft_type::complex_float_to_complex_float;
  dpct::fft::fft_type fft_type_5 = dpct::fft::fft_type::complex_double_to_complex_double;

  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

  float *data_d = sycl::malloc_device<float>(1024, q_ct1);

  int dim = 1;
  long long n_ll[1] = {5};
  int n_i[1] = {5};
  std::shared_ptr<dpct::fft::fft_solver> plan1;
  plan1 = std::make_shared<dpct::fft::fft_solver>(
      dim, n_ll, nullptr, 0, 0, dpct::library_data_t::complex_float, nullptr,
      0, 0, dpct::library_data_t::complex_float, 1);
  
  std::shared_ptr<dpct::fft::fft_solver> plan2;
  plan2 = std::make_shared<dpct::fft::fft_solver>(
      dim, n_i, nullptr, 0, 0, dpct::library_data_t::complex_float, nullptr,
      0, 0, dpct::library_data_t::complex_float, 1);

  std::shared_ptr<dpct::fft::fft_solver> plan3;
  plan3 = std::make_shared<dpct::fft::fft_solver>(
      dim, n_ll, nullptr, 0, 0, nullptr,
      0, 0, dpct::fft::fft_type::complex_float_to_complex_float, 1);

  std::shared_ptr<dpct::fft::fft_solver> plan4;
  plan4 = std::make_shared<dpct::fft::fft_solver>(
      dim, n_i, nullptr, 0, 0, nullptr,
      0, 0, dpct::fft::fft_type::complex_float_to_complex_float, 1);

  std::shared_ptr<dpct::fft::fft_solver> plan5;
  plan5 = std::make_shared<dpct::fft::fft_solver>(
      7, 7, 7, dpct::fft::fft_type::complex_float_to_complex_float);

  std::shared_ptr<dpct::fft::fft_solver> plan6;
  plan6 = std::make_shared<dpct::fft::fft_solver>(
      7, 7, dpct::fft::fft_type::complex_float_to_complex_float);

  std::shared_ptr<dpct::fft::fft_solver> plan7;
  plan7 = std::make_shared<dpct::fft::fft_solver>(
      7, dpct::fft::fft_type::complex_float_to_complex_float, 2);
  plan7->compute(data_d, data_d, dpct::fft::fft_dir::forward);

  dev_ct1.queues_wait_and_throw();
  sycl::free(data_d, q_ct1);
  return 0;
}
