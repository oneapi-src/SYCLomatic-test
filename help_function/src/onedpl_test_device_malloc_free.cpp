// ====------ onedpl_test_device_malloc_free.cpp---------- -*- C++ -* ----===////
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

#include <CL/sycl.hpp>

#include <sys/resource.h>

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

      {
        // Test One, test normal calls for dpct::malloc_device and dpct::free_device

        {
            dpct::device_pointer<double> double_ptr = dpct::malloc_device<double>(sizeof(double));
            double num_data = 1.0;

            dpct::get_default_queue().submit([&](sycl::handler& h) {
                h.memcpy(double_ptr, &num_data, sizeof(double));
            });
            dpct::get_default_queue().wait();

            dpct::get_default_queue().submit([&](sycl::handler& h) {
                h.memcpy(&num_data, double_ptr,  sizeof(double));
            });
            dpct::get_default_queue().wait();

            std::string test_name = "free_device test 1";
            failed_tests += ASSERT_EQUAL(test_name, num_data, 1.0);
            test_passed(failed_tests, test_name);

            dpct::free_device<double>(double_ptr);
        }
    }

    {
        // Test Two, test normal calls for dpct::malloc_device and dpct::free_device
        {

            dpct::device_pointer<int> array_ptr = dpct::malloc_device<int>(6 * sizeof(int));
            int data[6] = {8, 1, 8, 5, 6, 6};
            int expected_data[6] = {8, 1, 8, 5, 6, 6};

            // Load Data
            dpct::get_default_queue().submit([&](sycl::handler& h) {
                h.memcpy(array_ptr, data, 6 * sizeof(int));
            });
            dpct::get_default_queue().wait();

            // Copy Back
            dpct::get_default_queue().submit([&](sycl::handler& h) {
                h.memcpy(data, array_ptr, 6 * sizeof(int));
            });
            dpct::get_default_queue().wait();

            // check that it was copied properly
            std::string test_name = "free_device test 2";
            for (int i = 0; i != 6; ++i) {
                num_failing += ASSERT_EQUAL(test_name, expected_data[i], data[i]);
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
            dpct::free_device<int>(array_ptr);
        }
    }

    {
        // Test Three, test normal calls for dpct::malloc_device and dpct::free_device
        {
            dpct::device_vector<int, sycl::usm_allocator<int, sycl::usm::alloc::shared>> vec(8);
            std::vector<int> data = {0, 1, 4, 4, 2, 1, 0, 3};
            int expected_data[8] = {0, 1, 4, 4, 2, 1, 0, 3};

            // load in data
            dpct::get_default_queue().submit([&](sycl::handler& h) {
                h.memcpy(vec.data(), data.data(), 8 * sizeof(int));
            });
            dpct::get_default_queue().wait();

            // copy back
            dpct::get_default_queue().submit([&](sycl::handler& h) {
                h.memcpy(data.data(), vec.data(), 8 * sizeof(int));
            });
            dpct::get_default_queue().wait();

            // check that data is properly loaded into vec

            std::string test_name = "free_device test 3";
            for (int i = 0; i != 8; ++i) {
                num_failing += ASSERT_EQUAL(test_name, data[i], expected_data[i]);
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
         }
         // TODO add memory check here to confirm device_vector destructor freed memory.
    }

    std::cout << std::endl << failed_tests << " failing test(s) detected." << std::endl;
    if (failed_tests == 0) {
        return 0;
    }
    return 1;
}
