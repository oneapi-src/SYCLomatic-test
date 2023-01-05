// ====------ test_default_queue_1.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>
#include <iostream>

// This file is compiled together with test_default_queue_2.cpp.
sycl::queue &get_queue_1(void)
{
    auto v = dpct::malloc_device(32);
    std::cout << v.get() << std::endl;
    return dpct::get_default_queue();
}
