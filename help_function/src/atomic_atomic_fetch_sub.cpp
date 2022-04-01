// ====------ atomic_atomic_fetch_sub.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>

void atomic_test_kernel(int *ddata, cl::sycl::nd_item<3> item_ct1) {
  unsigned int tid = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);
  // add test
  dpct::atomic_fetch_sub(ddata, 1);
}

int main(int argc, char **argv) try {
  unsigned int numThreads = 256;
  unsigned int numBlocks = 64;
  int err = 0;

  int Hdata;
  int Hdata2;

  printf("atomic test \n");

  Hdata = numThreads * numBlocks;

  // allocate device memory for result
  int *Ddata;
  *((void **)&Ddata) = cl::sycl::malloc_device(sizeof(int), dpct::dev_mgr::instance().current_device(), dpct::get_default_queue().get_context());

  dpct::get_default_queue().memcpy((void*)(Ddata), (void*)(&Hdata), sizeof(int)).wait();

  {
    dpct::get_default_queue().submit(
      [&](cl::sycl::handler &cgh) {
        auto dpct_global_range = cl::sycl::range<3>(numBlocks, 1, 1) * cl::sycl::range<3>(numThreads, 1, 1);
        auto dpct_local_range = cl::sycl::range<3>(numThreads, 1, 1);
        cgh.parallel_for(
          cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)),
                                cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
          [=](cl::sycl::nd_item<3> item_ct1) {
            atomic_test_kernel(Ddata, item_ct1);
          });
      });
  }

  dpct::get_default_queue().memcpy((void*)(&Hdata2), (void*)(Ddata), sizeof(int)).wait();

  if (Hdata2 != 0) {
    err = -1;
    printf("atomicSub test failed\n");
  }

  cl::sycl::free(Ddata, dpct::get_default_queue().get_context());
  printf("atomic test completed, returned %s\n", err == 0 ? "OK" : "ERROR");
  return err;
}
catch (cl::sycl::exception const &exc) {
  std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
  std::exit(1);
}