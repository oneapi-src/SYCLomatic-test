// ====------ util_vectorized_max_test.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

void test_kernel(unsigned int vect_count, unsigned int *input_1,
                 unsigned int *input_2, unsigned int *output,
                 sycl::nd_item<3> item_ct1, const sycl::stream &stream_ct1) {

  int index = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2);

  if (index < vect_count) {
    output[index] =
        dpct::vectorized_max<sycl::char4>(input_1[index], input_2[index]);
    stream_ct1 << "[ " << input_1[index] << " " << input_2[index]
               << " ]-> Max : " << output[index] << "\n";
  }
}

void vec_max_test() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

  const unsigned int num_data = 7;
  unsigned int mem_size = sizeof(unsigned int) * num_data;

  unsigned int *h_out_data = (unsigned int *)malloc(mem_size);
  unsigned int *h_data = (unsigned int *)malloc(mem_size);

  for (unsigned int i = 0; i < num_data; i++)
    h_out_data[i] = 0;

  unsigned int *d_out_data;
  d_out_data = (unsigned int *)sycl::malloc_device(mem_size, q_ct1);

  unsigned int *d_in_data_1;
  d_in_data_1 = (unsigned int *)sycl::malloc_device(mem_size, q_ct1);

  unsigned int *d_in_data_2;
  d_in_data_2 = (unsigned int *)sycl::malloc_device(mem_size, q_ct1);

  for (unsigned int i = 0; i < num_data; i++)
    h_data[i] = i;
  q_ct1.memcpy(d_in_data_1, h_data, mem_size).wait();

  for (unsigned int i = 0; i < num_data; i++)
    h_data[i] = num_data - 1 - i;
  q_ct1.memcpy(d_in_data_2, h_data, mem_size).wait();

  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::stream stream_ct1(64 * 1024, 80, cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 3) * sycl::range<3>(1, 1, 3),
                          sycl::range<3>(1, 1, 3)),
        [=](sycl::nd_item<3> item_ct1) {
          test_kernel(num_data, d_in_data_1, d_in_data_2, d_out_data, item_ct1,
                      stream_ct1);
        });
  });
  dev_ct1.queues_wait_and_throw();

  q_ct1.memcpy(h_out_data, d_out_data, mem_size).wait();

  unsigned int ref_data[num_data] = {6, 5, 4, 3, 4, 5, 6};
  for (unsigned int i = 0; i < num_data; i++) {
    if (h_out_data[i] != ref_data[i])
      exit(-1);
  }

  free(h_out_data);
  free(h_data);
  sycl::free(d_out_data, q_ct1);
  sycl::free(d_in_data_1, q_ct1);
  sycl::free(d_in_data_2, q_ct1);
  printf("vectorized_max test passed!\n");
}

int main() {

  vec_max_test();

  return 0;
}