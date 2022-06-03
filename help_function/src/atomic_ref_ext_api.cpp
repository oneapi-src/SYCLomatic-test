// ====------ atomic_ref_ext_api.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//


#include "dpct/atomic_ref_ext.hpp"
#include <CL/sycl.hpp>
#include <assert.h>
#include <dpct/dpct.hpp>
#include <stdio.h>

#define loop_num 50

void atomicRefExtKernel(int* atom_arr){
  ////default constructor
  dpct::atomic_ref_ext<int> atomic_ref{0};
  int temp1=3,temp2 = 4;
  for(int i = 0;i<loop_num;++i){
    // Atomic_ref_ext store
    atomic_ref.store(1);

    // Atomic_ref_ext load
    atom_arr[0] = atomic_ref.load();

    // Atomic_ref_ext exchange
    atom_arr[1] = atomic_ref.exchange(3);

    // Atomic_ref_ext compare_exchange_weak
    atomic_ref.compare_exchange_weak(temp1,4);
    atom_arr[2] = atomic_ref.load();
    // Atomic_ref_ext compare_exchange_strong
    atomic_ref.compare_exchange_strong(temp2,8);
    atom_arr[3] = atomic_ref.load();

    //Atomic_ref_ext fetch_add
    atom_arr[4] =  atomic_ref.fetch_add(1);

    //Atomic_ref_ext fetch_sub
    atom_arr[5] = atomic_ref.fetch_sub(-1);

    //Atomic_ref_ext fetch_and
    atom_arr[6] = atomic_ref.fetch_and(4);

    //Atomic_ref_ext fetch_or
    atom_arr[7] = atomic_ref.fetch_or(2);

    //Atomic_ref_ext fetch_xor
    atom_arr[8] = atomic_ref.fetch_and(4);
  }
  
}

int verify(int *testDataDevice,int *testDataHost, const int len){
  bool result = true;
  for(int i = 0;i<len;++i ){
    if(testDataDevice[i]!=testDataHost[i])
    {
      printf("device result: %d , cpu result: %d . failure with %d\n",testDataDevice[i],testDataHost[i],i);
      result = false;
    }
  }
  return result;
}



int main(int argc, char **argv) {
  printf("atomic_ref_ext test \n");
  if (false) {
    fprintf(stderr, "This sample requires a device in either default or "
                    "process exclusive mode\n");
    exit(-1);
  }

  //default constructor
  dpct::atomic_ref_ext<int> tmp;

  unsigned int numThreads = 256;
  unsigned int numBlocks = 64;
  unsigned int numData = 9;

  int *atom_arr_device;

  atom_arr_device = sycl::malloc_shared<int>(numData, dpct::get_default_queue());

  for (unsigned int i = 0; i < numData; i++)
    atom_arr_device[i] = 0;
  

  std::cout << "Selected device: "
            << dpct::get_default_queue()
                   .get_device()
                   .get_info<sycl::info::device::name>()
            << "\n";

  std::chrono::time_point<std::chrono::steady_clock> start_ct1;
  std::chrono::time_point<std::chrono::steady_clock> stop_ct1;

  start_ct1 = std::chrono::steady_clock::now();
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, numBlocks) *
                              sycl::range<3>(1, 1, numThreads),
                          sycl::range<3>(1, 1, numThreads)),
        [=](sycl::nd_item<3> item_ct1) { atomicRefExtKernel(atom_arr_device); });
  });

  dpct::get_current_device().queues_wait_and_throw();

  stop_ct1 = std::chrono::steady_clock::now();

  float elapsed_time = 0;
  // calculate elapsed time
  elapsed_time =
      std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();

  printf("Measured time for parallel execution with "
         "std::chrono::steady_clock = %.3f ms\n",
         elapsed_time);

  int *atom_arr_cpu;

  atom_arr_cpu = sycl::malloc_shared<int>(numData, dpct::get_default_queue());

  for (unsigned int i = 0; i < numData; i++)
    atom_arr_cpu[i] = 0;

  atomicRefExtKernel(atom_arr_cpu);
  int testResult = verify(atom_arr_device, atom_arr_cpu,numData);
  { sycl::free(atom_arr_device, dpct::get_default_queue()); }
  { sycl::free(atom_arr_cpu, dpct::get_default_queue()); }
  printf("Atomics test completed, returned %s \n",
         testResult ? "OK" : "ERROR!");
  exit(testResult ? 0 : -1);
}