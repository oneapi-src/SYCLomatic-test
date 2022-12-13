// ====------ async_exception.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

int main() {
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    cgh.host_task([=] {
      throw sycl::exception(sycl::make_error_code(sycl::errc::invalid), "test_dpct_async_handler");
    });
  }).wait();
  return 0;
}

