// ====------ cub_blockexchange_tests.cpp---------- -*- C++ -* ----===////
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
void verify_data(int* data, int num) {
  return;
}
void print_data(int* data, int num) {
  for (int i = 0; i < num; i++) {
    std::cout << data[i] << " ";
  }
  std::cout << std::endl;
}

void striped_to_blocked_kernel(int* data,
                               sycl::nd_item<3> item_ct1){
                            
                            
  int threadid = item_ct1.get_local_id(2);
  sycl::multi_ptr<uint8_t*, access::address_space::local_space> _local_memory = sycl::ext::oneapi::group_local_memory_for_overwrite<uint8_t*>(item_ct1.get_group());
  dpct::group::exchange<int, VALUES_PER_THREAD> exchange(_local_memory).striped_to_blocked(item_ct1.get_group(), data);
  item_ct1.barrier(sycl::access::fence_space::local_space);
                            
}

void blocked_to_striped_kernel(int* data,
                               sycl::nd_item<3> item_ct1){
                        
  int threadid = item_ct1.get_local_id(2);
  sycl::multi_ptr<uint8_t*, access::address_space::local_space> _local_memory = sycl::ext::oneapi::group_local_memory_for_overwrite<uint8_t*>(item_ct1.get_group());
  dpct::group::exchange<int, VALUES_PER_THREAD> exchange(_local_memory).blocked_to_striped(item_ct1.get_group(), data);
  item_ct1.barrier(sycl::access::fence_space::local_space);
                        
}

void scatter_to_blocked_kernel(int* data, int* rank_data,
                               sycl::nd_item<3> item_ct1){
                            
                            
  int threadid = item_ct1.get_local_id(2);
  sycl::multi_ptr<uint8_t*, access::address_space::local_space> _local_memory = sycl::ext::oneapi::group_local_memory_for_overwrite<uint8_t*>(item_ct1.get_group());
  dpct::group::exchange<int, VALUES_PER_THREAD> exchange(_local_memory).scatter_to_blocked(item_ct1.get_group(), data, rank_data);
  item_ct1.barrier(sycl::access::fence_space::local_space);
                            
}

void scatter_to_striped_kernel(int* data, int* rank_data,
                               sycl::nd_item<3> item_ct1){
                            
                            
  int threadid = item_ct1.get_local_id(2);
  sycl::multi_ptr<uint8_t*, access::address_space::local_space> _local_memory = sycl::ext::oneapi::group_local_memory_for_overwrite<uint8_t*>(item_ct1.get_group());
  dpct::group::exchange<int, VALUES_PER_THREAD> exchange(_local_memory).scatter_to_striped(item_ct1.get_group(), data, rank_data);
  item_ct1.barrier(sycl::access::fence_space::local_space);
                            
}

int main () {
  
  int* data = static_cast<int*>(dpct::dpct_malloc(SIZE * sizeof(int)));
  int* rank_data = static_cast<int*>(dpct::dpct_malloc(SIZE * sizeof(int)));
  init_data(data, SIZE);
  init_data(rank_data, SIZE);
  sycl::buffer<int> buffer_data(data, sycl::range<1>(SIZE));
  sycl::buffer<int> buffer_rank_data(rank_data, sycl::range<1>(SIZE));
  
  //striped_to_blocked
  dpct::get_default_queue().submit(
    [&](sycl::handler &cgh) {
      auto dataAccessor = buffer_data.get_access<sycl::access::mode::read_write>(cgh);
  
      cgh.parallel_for(
        sycl::range<3>(SIZE, 1, 1), 
        [=](sycl::nd_item<3> item_ct1) {
          striped_to_blocked_kernel(dataAccessor.get_pointer(), item_ct1);
        });
    });
  
  
  dpct::get_current_device().queues_wait_and_throw();
   
  //blocked_to_striped
  dpct::get_default_queue().submit(
    [&](sycl::handler &cgh) {
      auto dataAccessor = buffer_data.get_access<sycl::access::mode::read_write>(cgh);
  
      cgh.parallel_for(
        sycl::range<3>(SIZE, 1, 1), 
        [=](sycl::nd_item<3> item_ct1) {
          blocked_to_striped_kernel(dataAccessor.get_pointer(), item_ct1);
        });
    });
  
  
  dpct::get_current_device().queues_wait_and_throw();
  
  //scatter_to_blocked
  dpct::get_default_queue().submit(
    [&](sycl::handler &cgh) {
      auto dataAccessor = buffer_data.get_access<sycl::access::mode::read_write>(cgh);
      auto dataRankAccessor = buffer_rank_data.get_access<sycl::access::mode::read_write>(cgh);
      cgh.parallel_for(
        sycl::range<3>(SIZE, 1, 1), sycl::range<3>(SIZE, 1, 1)
        [=](sycl::nd_item<3> item_ct1) {
          scatter_to_blocked_kernel(dataAccessor.get_pointer(), dataRankAccessor.get_pointer(), item_ct1);
        });
    });
  
  
  dpct::get_current_device().queues_wait_and_throw();

  //scatter_to_striped
  dpct::get_default_queue().submit(
    [&](sycl::handler &cgh) {
      auto dataAccessor = buffer_data.get_access<sycl::access::mode::read_write>(cgh);
      auto dataRankAccessor = buffer_rank_data.get_access<sycl::access::mode::read_write>(cgh);
      cgh.parallel_for(
        sycl::range<3>(SIZE, 1, 1), sycl::range<3>(SIZE, 1, 1)
        [=](sycl::nd_item<3> item_ct1) {
          scatter_to_striped_kernel(dataAccessor.get_pointer(), dataRankAccessor.get_pointer(), item_ct1);
        });
    });
  
  
  dpct::get_current_device().queues_wait_and_throw();
  
  
  return 0;
}
