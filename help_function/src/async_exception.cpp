// ====------ async_exception.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>

int main() try {
  dpct::get_default_queue().submit(
    [&](cl::sycl::handler &cgh) {
      cgh.parallel_for<class kernel>(
        cl::sycl::nd_range<3>(cl::sycl::range<3>(256, 1, 1), cl::sycl::range<3>(16384, 1, 1)),
          [=](cl::sycl::nd_item<3> item_ct1) {
      });
  });
} catch (sycl::exception const &exec) {
  std::cerr << exec.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);

}
