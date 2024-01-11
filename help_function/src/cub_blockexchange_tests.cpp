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

#define VALUES_PER_THREAD = 4
#define SIZE = 16

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

dpct::shared_memory<float, 1> array(N);
dpct::shared_memory<float, 1> result(M*N);

void striped_to_blocked_kernel(int* data,
                               sycl::nd_item<3> item_ct1){
                            
                            
  int threadid = item_ct1.get_local_id(2);
  int input = data[threadid];
  uint8_t* _local_memory = sycl::ext::oneapi::group_local_memory_for_overwrite<uint8_t*>(item_ct1.get_group())
  dpct::group::exchange<input, VALUES_PER_THREAD> exchange(_local_memory).striped_to_blocked(item_ct1.get_group(), input);
  item_ct1.barrier(sycl::access::fence_space::local_space);
                            
}

void blocked_to_striped_kernel(int* data,
                               sycl::nd_item<3> item_ct1){
                        
  int threadid = item_ct1.get_local_id(2);
  int input = data[threadid];
  uint8_t* _local_memory = sycl::ext::oneapi::group_local_memory_for_overwrite<uint8_t*>(item_ct1.get_group())
  dpct::group::exchange<input, VALUES_PER_THREAD> exchange(_local_memory).blocked_to_striped(item_ct1.get_group(), input);
  item_ct1.barrier(sycl::access::fence_space::local_space);
                        


}

int main () {
  
  int* data = static_cast<int*>(dpct::dpct_malloc(SIZE * sizeof(int)));
  init_data(data, SIZE);
  sycl::buffer<int> buffer_data(data, cl::sycl::range<1>(SIZE));
  
  //striped_to_blocked
  dpct::get_default_queue().submit(
    [&](sycl::handler &cgh) {
      auto dataAccessor = dataBuffer.get_access<sycl::access::mode::read_write>(cgh);
  
      cgh.parallel_for(
        cl::sycl::range<3>(SIZE, 1, 1), 
        [=](sycl::nd_item<3> item_ct1) {
          striped_to_blocked_kernel(dataAccessor.get_pointer(), item_ct1);
        });
    });
  
  
  dpct::get_current_device().queues_wait_and_throw();
   
  //blocked_to_striped
  dpct::get_default_queue().submit(
    [&](sycl::handler &cgh) {
      auto dataAccessor = dataBuffer.get_access<sycl::access::mode::read_write>(cgh);
  
      cgh.parallel_for(
        cl::sycl::range<3>(SIZE, 1, 1), 
        [=](sycl::nd_item<3> item_ct1) {
          blocked_to_striped_kernel(dataAccessor.get_pointer(), item_ct1);
        });
    });
  
  
  dpct::get_current_device().queues_wait_and_throw();
  

  return 0;
}