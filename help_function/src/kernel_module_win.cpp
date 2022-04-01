// ====------ kernel_module_win.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

void foo(int* k, sycl::nd_item<3> item_ct1, uint8_t *dpct_local){
    k[item_ct1.get_global_linear_id()] = item_ct1.get_global_linear_id();
}

extern "C" {
  __declspec(dllexport) void foo_wrapper(sycl::queue &queue, const sycl::nd_range<3> &nr, unsigned int localMemSize, void **kernelParams, void **extra){
    int * k;
    k = (int *)kernelParams[0];
    queue.submit(
      [&](sycl::handler &cgh) {
        sycl::accessor<uint8_t, 1, sycl::access_mode::read_write, sycl::access::target::local> dpct_local_acc_ct1(sycl::range<1>(localMemSize), cgh);
        cgh.parallel_for(
          nr,
          [=](sycl::nd_item<3> item_ct1) {
            foo(k, item_ct1, dpct_local_acc_ct1.get_pointer());
          });
      });
  }
}