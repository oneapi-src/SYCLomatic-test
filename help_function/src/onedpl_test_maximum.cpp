// ====------ onedpl_test_maximum.cpp---------- -*- C++ -* ----===////
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

#include <sycl/sycl.hpp>

#include <iostream>

template<typename String, typename _T1, typename _T2>
int ASSERT_EQUAL(String msg, _T1&& X, _T2&& Y) {
    if(X!=Y) {
        std::cout << "FAIL: " << msg << " - (" << X << "," << Y << ")" << std::endl;
        return 1;
    }
    else {
        std::cout << "PASS: " << msg << std::endl;
        return 0;
    }
}

int main() {

    // used to detect failures
    int failed_tests = 0;

    // test 1/2

    // create array
    int hostArray[8] = { 3, 17, -5, 6, 2, 9, 0, -11 };

    // call algorithm
    oneapi::dpl::maximum<int> mx;

    failed_tests += ASSERT_EQUAL("maximum identifies max value with negative number", mx(hostArray[5], hostArray[7]), 9);

    // test 2/2

    // create vector
    auto vec = std::vector<int>(hostArray, hostArray + 8);

    // call algorithm
    auto result = std::accumulate(vec.begin(), vec.end(), 0, mx);

    failed_tests += ASSERT_EQUAL("maximum of array", result, 17);

    std::cout << std::endl << failed_tests << " failing test(s) detected." << std::endl;
    if (failed_tests == 0) {
        return 0;
    }
    return 1;
}
