// ====------ util_logical_group.cpp --------------------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdio>

// work-item: 0 ... 7, 8 ... 15, 16 ... 23, 24 ... 31, 32 ... 39, 40 ... 47, 48 ... 51
//            -------  --------  ---------  ---------  ---------  ---------  ---------
//            0        1         2          3          4          5          6

void kernel(unsigned int *result, sycl::nd_item<3> item_ct1) {
  auto ttb = item_ct1.get_group();
  dpct::experimental::logical_group tbt =
      dpct::experimental::logical_group(item_ct1, item_ct1.get_group(), 8);

  if (item_ct1.get_local_id(2) == 50) {
    result[0] = tbt.get_local_linear_range();
    result[1] = tbt.get_local_linear_id();
    result[2] = tbt.get_group_linear_range();
    result[3] = tbt.get_group_linear_id();
  }
}

int main() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  unsigned int result_host[4];
  unsigned int *result_device;
  result_device = sycl::malloc_device<unsigned int>(4, q_ct1);
  q_ct1.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, 52), sycl::range<3>(1, 1, 52)),
      [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
        kernel(result_device, item_ct1);
      });
  q_ct1.memcpy(result_host, result_device, sizeof(unsigned int) * 4).wait();
  sycl::free(result_device, q_ct1);

  if (result_host[0] == 4 &&
      result_host[1] == 2 &&
      result_host[2] == 7 &&
      result_host[3] == 6) {
    return 0;
  }
  printf("test failed\n");
  printf("%d, %d, %d, %d\n", result_host[0], result_host[1], result_host[2], result_host[3]);
  return -1;
}

