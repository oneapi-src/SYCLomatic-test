// ====------ onedpl_test_count_if.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include "oneapi/dpl/execution"
#include "oneapi/dpl/iterator"
#include "oneapi/dpl/algorithm"

#include "dpct/dpct.hpp"
#include "dpct/dpl_utils.hpp"

#include <iostream>

#include <sycl/sycl.hpp>

template<typename String, typename _T1, typename _T2>
int ASSERT_EQUAL(String msg, _T1&& X, _T2&& Y) {
    if(X!=Y) {
        std::cout << "FAIL: " << msg << " - (" << X << "," << Y << ")" << std::endl;
        return 1;
    }
    return 0;
}

int main() {

    // used to detect failures
    int failed_tests = 0;

    // test 1/1

    // call algorithm
    auto result = std::count_if
    (
        dpct::make_counting_iterator(0),
        dpct::make_counting_iterator(0) + 10,
        ([=](int i) {
            return (i%2) == 0;
        })
    );

    std::string test_name = "Regular call to std::count_if";

    failed_tests += ASSERT_EQUAL(test_name, result, 5);

    if (failed_tests == 0) {
        std::cout << "PASS: " << test_name << std::endl;
    }

    std::cout << std::endl << failed_tests << " failing test(s) detected." << std::endl;
    if (failed_tests == 0) {
        return 0;
    }
    return 1;
}

