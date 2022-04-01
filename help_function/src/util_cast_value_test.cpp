// ====------ util_cast_value_test.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

double cast_value(const double &val) {
  int lo = dpct::cast_double_to_int(val, false);
  int hi = dpct::cast_double_to_int(val);
  return dpct::cast_ints_to_double(hi, lo);
}

void test_kernel(double *g_odata) {
  double a = 1.12123515e-25f;
  g_odata[0] = cast_value(a);

  a = 0.000000000000000000000000112123515f;
  g_odata[1] = cast_value(a);

  a = 3.1415926f;
  g_odata[2] = cast_value(a);
}

void double_int2_cast_test() {

  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

  unsigned int num_data = 3;
  unsigned int mem_size = sizeof(double) * num_data;

  double *h_out_data = (double *)malloc(mem_size);

  for (unsigned int i = 0; i < num_data; i++)
    h_out_data[i] = 0;

  double *d_out_data;
  d_out_data = (double *)sycl::malloc_device(mem_size, q_ct1);
  q_ct1.memcpy(d_out_data, h_out_data, mem_size).wait();

  q_ct1.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
      [=](sycl::nd_item<3> item_ct1) { test_kernel(d_out_data); });
  dev_ct1.queues_wait_and_throw();

  q_ct1.memcpy(h_out_data, d_out_data, mem_size).wait();

  if (h_out_data[0] != 1.12123515e-25f)
    exit(-1);

  if (h_out_data[1] != 0.000000000000000000000000112123515f)
    exit(-1);

  if (h_out_data[2] != 3.1415926f)
    exit(-1);

  free(h_out_data);
  sycl::free(d_out_data, q_ct1);

  printf("double_int2_cast_test passed!\n");
}

int main() {
  double_int2_cast_test();
  return 0;
}
