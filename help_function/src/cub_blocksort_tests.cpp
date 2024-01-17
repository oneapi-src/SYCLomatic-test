// ====------ cub_blocksort_tests.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_extras/dpcpp_extensions.h>
#include <sycl/ext/oneapi/group_local_memory.hpp>

#include <stdio.h>
#include <string.h>

#define VALUES_PER_THREAD  4
#define SIZE  16

void init_data(int* data, int num) {
  for(int i = 0; i < num; i++)
    data[i] = i;
}

template <typename T = int>
void verify_data(T *data, T *expect, int num, int step = 1) {
  for (int i = 0; i < num; i = i + step) {
    assert(data[i] == expect[i]);
  }

void print_data(int* data, int num) {
  for (int i = 0; i < num; i++) {
    std::cout << data[i] << " ";
  }
  std::cout << std::endl;
}

void sort_blocked_kernel(int* data,
                         sycl::nd_item<3> item_ct1){
                            
                            
  int threadid = item_ct1.get_local_id(2);
  sycl::multi_ptr<uint8_t*, access::address_space::local_space> _local_memory = sycl::ext::oneapi::group_local_memory_for_overwrite<uint8_t*>(item_ct1.get_group());
  dpct::group::radix_sort<int, VALUES_PER_THREAD> radix_sort(_local_memory).sort_blocked(item_ct1.get_group(), data);
  item_ct1.barrier(sycl::access::fence_space::local_space);
                            
}

void sort_blocked_to_striped_kernel(int* data,
                                    sycl::nd_item<3> item_ct1){
                        
  int threadid = item_ct1.get_local_id(2);
  sycl::multi_ptr<uint8_t*, access::address_space::local_space> _local_memory = sycl::ext::oneapi::group_local_memory_for_overwrite<uint8_t*>(item_ct1.get_group());
  dpct::group::radix_sort<int, VALUES_PER_THREAD> radix_sort(_local_memory).sort_blocked_to_striped(item_ct1.get_group(), data);
  item_ct1.barrier(sycl::access::fence_space::local_space);
                        
}

int main () {
  
  int* data = static_cast<int*>(dpct::dpct_malloc(SIZE * sizeof(int)));
  init_data(data, SIZE);
  sycl::buffer<int> buffer_data(data, sycl::range<1>(SIZE));
  unsigned int expect[SIZE] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  
  //sort blocked
  dpct::get_default_queue().submit(
    [&](sycl::handler &cgh) {
      auto dataAccessor = buffer_data.get_access<sycl::access::mode::read_write>(cgh);
  
      cgh.parallel_for(
        sycl::range<3>(SIZE, 1, 1), 
        [=](sycl::nd_item<3> item_ct1) {
          sort_blocked_kernel(dataAccessor.get_pointer(), item_ct1);
        });
    });
  
  
  dpct::get_current_device().queues_wait_and_throw();
  verify_data<unsigned int>(data, expect, SIZE);
   
  //sort blocked_to_striped
  dpct::get_default_queue().submit(
    [&](sycl::handler &cgh) {
      auto dataAccessor = buffer_data.get_access<sycl::access::mode::read_write>(cgh);
  
      cgh.parallel_for(
        sycl::range<3>(SIZE, 1, 1), 
        [=](sycl::nd_item<3> item_ct1) {
          sort_blocked_to_striped_kernel(dataAccessor.get_pointer(), item_ct1);
        });
    });
  
  
  dpct::get_current_device().queues_wait_and_throw();
  verify_data<unsigned int>(data, expect, SIZE);
  
  return 0;
}