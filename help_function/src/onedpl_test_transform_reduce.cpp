// ====------ onedpl_test_transform_reduce.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include "oneapi/dpl/execution"
#include "oneapi/dpl/algorithm"
#include "oneapi/dpl/iterator"

#include "dpct/dpct.hpp"
#include "dpct/dpl_utils.hpp"

#include <iostream>
#include <iomanip>

#include <sycl/sycl.hpp>

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

struct square {
    double operator()(double& x) const {
        return x*x;
    }
};

int main() {

    // used to detect failures
    int failed_tests = 0;
    std::string test_name = "";

    // test 1/2

    // create queue
    sycl::queue myQueue;
    auto dev = myQueue.get_device();
    auto ctxt = myQueue.get_context();

    // create host and device arrays
    double hostArray[8];
    double *deviceArray = (double*) malloc_device(8 * sizeof(double), dev, ctxt);

    // hostArray = { 0, 1, 2, 3, 4, 5, 6, 7 }
    for (int i = 0; i != 8; ++i) {
        hostArray[i] = i;
    }

    myQueue.submit([&](sycl::handler& h) {
        // copy hostArray to deviceArray
        h.memcpy(deviceArray, hostArray, 8 * sizeof(double));
    });
    myQueue.wait();

    {
        auto dptr_begin = dpct::device_pointer<double>(deviceArray);
        auto dptr_end = dpct::device_pointer<double>(deviceArray + 4);

        // call algorithm
        auto result = std::transform_reduce(oneapi::dpl::execution::dpcpp_default, dptr_begin, dptr_end, 0., std::plus<double>(), square());

        test_name = "transform_reduce with USM allocation";
        failed_tests += ASSERT_EQUAL(test_name, result, 14);
    }

    // test 2/2

    bool hostArray2[8];
    bool *deviceArray2 = (bool*) malloc_device(8 * sizeof(bool), dev, ctxt);

    // hostArray2 = { 1, 1, 1, 1, 0, 0, 0, 0 }
    for (int i = 0; i != 8; ++i) {
        if (i < 4)
            hostArray2[i] = 1;
        else
            hostArray2[i] = 0;
    }

    myQueue.submit([&](sycl::handler& h) {
        // copy hostArray2 to deviceArray2
        h.memcpy(deviceArray2, hostArray2, 8 * sizeof(bool));
    });
    myQueue.wait();

    {
        auto dptr_begin = dpct::device_pointer<bool>(deviceArray2 + 4);
        auto dptr_end = dpct::device_pointer<bool>(deviceArray2 + 8);

        // call algorithm
        auto result = std::transform_reduce(oneapi::dpl::execution::dpcpp_default, dptr_begin, dptr_end, 0, std::plus<bool>(), oneapi::dpl::identity());

        test_name = "transform_reduce with USM allocation 2";
        failed_tests += ASSERT_EQUAL(test_name, result, 0);

    }

    std::cout << std::endl << failed_tests << " failing test(s) detected." << std::endl;
    if (failed_tests == 0) {
        return 0;
    }
    return 1;
}
