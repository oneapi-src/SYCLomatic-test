// ====------ onedpl_test_raw_pointer_cast.cpp---------- -*- C++ -* ----===////
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

#include <CL/sycl.hpp>

#include <iostream>

template<typename String, typename _T1, typename _T2>
int ASSERT_EQUAL(String msg, _T1&& X, _T2&& Y) {
    if(X!=Y) {
        std::cout << "FAIL: " << msg << " - (" << X << "," << Y << ")" << std::endl;
        return 1;
    }
    return 0;
}

int test_passed(int failing_elems, std::string test_name) {
    if (failing_elems == 0) {
        std::cout << "PASS: " << test_name << std::endl;
        return 0;
    }
    return 1;
}

int main() {

    // used to detect failures
    int failed_tests = 0;
    int num_failing = 0;
    std::string test_name = "";

    // First Group: dpct::get_raw_pointer calls with device_vector passed in

    // test 1/2

    std::vector<float> src_vec(8);

    for (int i = 0; i != 8; ++i) {
        src_vec[i] = i;
    }

    dpct::device_vector<float> src_dv(src_vec);

    // call algorithm
    float* src_rpc = dpct::get_raw_pointer(src_dv.data());

    test_name = "dpct::get_raw_pointer with device_vector 1/2";

    // checking src_rpc against 0 fails since it is infinitely small but not quite 0
    for (int i = 0; i != 8; ++i) {
        num_failing += ASSERT_EQUAL(test_name, *(src_rpc + i), i);
    }

    failed_tests += test_passed(num_failing, test_name);
    num_failing = 0;

    // test 2/2
    dpct::device_vector<int> *src_dv2[4];
    std::vector<int> src_vec2(8);

    for (int i = 0; i != 8; ++i) {
        src_vec2[i] = i;
    }

    for (int i = 0; i!= 4; ++i) {
        src_dv2[i] = new dpct::device_vector<int>(src_vec2);
    }

    test_name = "dpct::get_raw_pointer with device_vector 2/2";

    for (int i = 0; i != 4; ++i) {
        for (int j = 0; j != 8; ++j) {
            // call algorithm
            num_failing += ASSERT_EQUAL(test_name, dpct::get_raw_pointer((*src_dv2[i])[j]), j);
        }
    }

    failed_tests += test_passed(num_failing, test_name);
    num_failing = 0;

    // Second Group: dpct::get_raw_pointer calls with host_vector (-> std::vector) passed in

    // test 1/1

    std::vector<double> src_vec3(8);

    for (int i = 0; i != 8; ++i) {
        src_vec3[i] = i;
    }

    std::vector<double> src_dv3(src_vec3);

    // call algorithm
    double* src_rpc3 = dpct::get_raw_pointer(&src_dv3[0]);

    test_name = "dpct::get_raw_pointer with std::vector";

    for (int i = 0; i != 8; ++i) {
        num_failing += ASSERT_EQUAL(test_name, *(src_rpc3 + i), i);
    }

    failed_tests += test_passed(num_failing, test_name);

    std::cout << std::endl << failed_tests << " failing test(s) detected." << std::endl;
    if (failed_tests == 0) {
        return 0;
    }
    return 1;
}
