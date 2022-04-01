// ====------ onedpl_test_set_symmetric_difference_by_key.cpp---------- -*- C++ -* ----===////
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

#include <CL/sycl.hpp>

#include <iostream>
#include <iomanip>

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

class SetSymDiffByKey {};        // name for policy
class SetSymDiffByKeyGreater {}; // name for policy

int main() {

    // used to detect failures
    int failed_tests = 0;
    int num_failing = 0;
    std::string test_name = "";

    // #12 SET SYMMETRIC DIFFERENCE BY KEY TEST //

    {
        // create buffers
        cl::sycl::buffer<uint64_t, 1> key_buf1{ cl::sycl::range<1>(8) };
        cl::sycl::buffer<uint64_t, 1> key_buf2{ cl::sycl::range<1>(10) };
        cl::sycl::buffer<uint64_t, 1> val_buf1{ cl::sycl::range<1>(8) };
        cl::sycl::buffer<uint64_t, 1> val_buf2{ cl::sycl::range<1>(10) };
        cl::sycl::buffer<uint64_t, 1> key_res_buf{ cl::sycl::range<1>(18) };
        cl::sycl::buffer<uint64_t, 1> val_res_buf{ cl::sycl::range<1>(18) };

        // create sycl iterators
        auto key_beg1_it = oneapi::dpl::begin(key_buf1);
        auto key_end1_it = oneapi::dpl::end(key_buf1);
        auto key_beg2_it = oneapi::dpl::begin(key_buf2);
        auto key_end2_it = oneapi::dpl::end(key_buf2);
        auto val_beg1_it = oneapi::dpl::begin(val_buf1);
        auto val_beg2_it = oneapi::dpl::begin(val_buf2);
        auto key_res_beg_it = oneapi::dpl::begin(key_res_buf);
        auto val_res_beg_it = oneapi::dpl::begin(val_res_buf);

        //T A_keys[n1] = { 1, 1, 3, 3, 5, 5, 7, 9 };
        //T A_vals[n1] = { 0, 1, 0, 1, 0, 1, 0, 0 };
        //T B_keys[n2] = { 0, 1, 2, 3, 4, 5, 5, 6, 8, 10 };
        //T B_vals[n2] = { 9, 9, 9, 9, 9, 9, 9, 9, 9, 9 };

        // keys_result = { 0, 1, 2, 3, 4, 6, 7, 8, 9, 10 }
        // vals_result = { 9, 1, 9, 1, 9, 9, 0, 9, 0, 9 }

	// Initialize data
        {
            auto key_beg1 = key_beg1_it.get_buffer().template get_access<cl::sycl::access::mode::write>();
            auto key_beg2 = key_beg2_it.get_buffer().template get_access<cl::sycl::access::mode::write>();
            auto val_beg1 = val_beg1_it.get_buffer().template get_access<cl::sycl::access::mode::write>();
            auto val_beg2 = val_beg2_it.get_buffer().template get_access<cl::sycl::access::mode::write>();
            auto key_res_beg = key_res_beg_it.get_buffer().template get_access<cl::sycl::access::mode::write>();
            auto val_res_beg = val_res_beg_it.get_buffer().template get_access<cl::sycl::access::mode::write>();

            key_beg1[0] = 1; key_beg1[1] = 1; key_beg1[2] = 3; key_beg1[3] = 3; key_beg1[4] = 5;
            key_beg1[5] = 5; key_beg1[6] = 7; key_beg1[7] = 9;
            key_beg2[0] = 0; key_beg2[1] = 1; key_beg2[2] = 2; key_beg2[3] = 3; key_beg2[4] = 4;
            key_beg2[5] = 5; key_beg2[6] = 5; key_beg2[7] = 6; key_beg2[8] = 8; key_beg2[9] = 10;

            val_beg1[0] = 0; val_beg1[1] = 1; val_beg1[2] = 0; val_beg1[3] = 1; val_beg1[4] = 0;
            val_beg1[5] = 1; val_beg1[6] = 0; val_beg1[7] = 0;
            val_beg2[0] = 9; val_beg2[1] = 9; val_beg2[2] = 9; val_beg2[3] = 9; val_beg2[4] = 9;
            val_beg2[5] = 9; val_beg2[6] = 9; val_beg2[7] = 9; val_beg2[8] = 9; val_beg2[9] = 9;

            for (int i = 0; i != 18; ++i) {
              key_res_beg[i] = 11;
              val_res_beg[i] = 11;
            }
        }

        // create named policy from existing one
        auto new_policy = oneapi::dpl::execution::make_device_policy<SetSymDiffByKey>(oneapi::dpl::execution::dpcpp_default);

        // call algorithm
	dpct::set_symmetric_difference(new_policy, key_beg1_it, key_end1_it, key_beg2_it,
                                              key_end2_it, val_beg1_it, val_beg2_it, key_res_beg_it,
                                              val_res_beg_it);

	// check values
        {
            auto key_beg1 = key_beg1_it.get_buffer().template get_access<cl::sycl::access::mode::read_write>();
            auto key_beg2 = key_beg2_it.get_buffer().template get_access<cl::sycl::access::mode::read_write>();
            auto val_beg1 = val_beg1_it.get_buffer().template get_access<cl::sycl::access::mode::read_write>();
            auto val_beg2 = val_beg2_it.get_buffer().template get_access<cl::sycl::access::mode::read_write>();
            auto key_res_beg = key_res_beg_it.get_buffer().template get_access<cl::sycl::access::mode::read_write>();
            auto val_res_beg = val_res_beg_it.get_buffer().template get_access<cl::sycl::access::mode::read_write>();
            for (int i = 1; i != 18; ++i) {
                if (key_res_beg[i-1] > key_res_beg[i]) {
                    std::cout << "fail - keys not sorted" << std::endl;
                }
            }

            test_name = "set_symmetric_difference_by_key test 1";

            num_failing += ASSERT_EQUAL(test_name, key_res_beg[0], 0);
            num_failing += ASSERT_EQUAL(test_name, key_res_beg[1], 1);
            num_failing += ASSERT_EQUAL(test_name, key_res_beg[2], 2);
            num_failing += ASSERT_EQUAL(test_name, key_res_beg[3], 3);
            num_failing += ASSERT_EQUAL(test_name, key_res_beg[4], 4);
            num_failing += ASSERT_EQUAL(test_name, key_res_beg[5], 6);
            num_failing += ASSERT_EQUAL(test_name, key_res_beg[6], 7);
            num_failing += ASSERT_EQUAL(test_name, key_res_beg[7], 8);
            num_failing += ASSERT_EQUAL(test_name, key_res_beg[8], 9);
            num_failing += ASSERT_EQUAL(test_name, key_res_beg[9], 10);

            num_failing += ASSERT_EQUAL(test_name, val_res_beg[0], 9);
            num_failing += ASSERT_EQUAL(test_name, val_res_beg[1], 1);
            num_failing += ASSERT_EQUAL(test_name, val_res_beg[2], 9);
            num_failing += ASSERT_EQUAL(test_name, val_res_beg[3], 1);
            num_failing += ASSERT_EQUAL(test_name, val_res_beg[4], 9);
            num_failing += ASSERT_EQUAL(test_name, val_res_beg[5], 9);
            num_failing += ASSERT_EQUAL(test_name, val_res_beg[6], 0);
            num_failing += ASSERT_EQUAL(test_name, val_res_beg[7], 9);
            num_failing += ASSERT_EQUAL(test_name, val_res_beg[8], 0);
            num_failing += ASSERT_EQUAL(test_name, val_res_beg[9], 9);

            for (int i = 10; i != 18; ++i) {
                num_failing += ASSERT_EQUAL(test_name, key_res_beg[i], 11);
                num_failing += ASSERT_EQUAL(test_name, val_res_beg[i], 11);
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;

        // reinitialize data for comparator operator>
            key_beg1[0] = 9; key_beg1[1] = 7; key_beg1[2] = 5; key_beg1[3] = 5; key_beg1[4] = 3;
            key_beg1[5] = 3; key_beg1[6] = 1; key_beg1[7] = 1;
            key_beg2[0] = 10; key_beg2[1] = 8; key_beg2[2] = 6; key_beg2[3] = 5; key_beg2[4] = 5;
            key_beg2[5] = 4; key_beg2[6] = 3; key_beg2[7] = 2; key_beg2[8] = 1; key_beg2[9] = 0;

            val_beg1[0] = 0; val_beg1[1] = 0; val_beg1[2] = 0; val_beg1[3] = 1; val_beg1[4] = 0;
            val_beg1[5] = 1; val_beg1[6] = 0; val_beg1[7] = 1;
            val_beg2[0] = 9; val_beg2[1] = 9; val_beg2[2] = 9; val_beg2[3] = 9; val_beg2[4] = 9;
            val_beg2[5] = 9; val_beg2[6] = 9; val_beg2[7] = 9; val_beg2[8] = 9; val_beg2[9] = 9;

            for (int i = 0; i != 18; ++i) {
              key_res_beg[i] = 11;
              val_res_beg[i] = 11;
            }
        }

        // create named policy from existing one.
        auto new_policy2 = oneapi::dpl::execution::make_device_policy<SetSymDiffByKeyGreater>(oneapi::dpl::execution::dpcpp_default);

        // call algorithm
	dpct::set_symmetric_difference(new_policy2, key_beg1_it, key_end1_it, key_beg2_it,
                                              key_end2_it, val_beg1_it, val_beg2_it, key_res_beg_it,
                                              val_res_beg_it, std::greater<uint64_t>());

	// check values
        {
            auto key_res_beg = key_res_beg_it.get_buffer().template get_access<cl::sycl::access::mode::write>();
            auto val_res_beg = val_res_beg_it.get_buffer().template get_access<cl::sycl::access::mode::write>();
            for (int i = 1; i != 10; ++i) {
                if (key_res_beg[i-1] < key_res_beg[i]) {
                    std::cout << "fail - keys not sorted" << std::endl;
                }
            }

            test_name = "set_symmetric_difference_by_key test 2";

            for (int i = 10; i != 18; ++i) {
                num_failing += ASSERT_EQUAL(test_name, key_res_beg[i], 11);
            }

            num_failing += ASSERT_EQUAL(test_name, key_res_beg[0], 10);
            num_failing += ASSERT_EQUAL(test_name, key_res_beg[1], 9);
            num_failing += ASSERT_EQUAL(test_name, key_res_beg[2], 8);
            num_failing += ASSERT_EQUAL(test_name, key_res_beg[3], 7);
            num_failing += ASSERT_EQUAL(test_name, key_res_beg[4], 6);
            num_failing += ASSERT_EQUAL(test_name, key_res_beg[5], 4);
            num_failing += ASSERT_EQUAL(test_name, key_res_beg[6], 3);
            num_failing += ASSERT_EQUAL(test_name, key_res_beg[7], 2);
            num_failing += ASSERT_EQUAL(test_name, key_res_beg[8], 1);
            num_failing += ASSERT_EQUAL(test_name, key_res_beg[9], 0);

            num_failing += ASSERT_EQUAL(test_name, val_res_beg[0], 9);
            num_failing += ASSERT_EQUAL(test_name, val_res_beg[1], 0);
            num_failing += ASSERT_EQUAL(test_name, val_res_beg[2], 9);
            num_failing += ASSERT_EQUAL(test_name, val_res_beg[3], 0);
            num_failing += ASSERT_EQUAL(test_name, val_res_beg[4], 9);
            num_failing += ASSERT_EQUAL(test_name, val_res_beg[5], 9);
            num_failing += ASSERT_EQUAL(test_name, val_res_beg[6], 1);
            num_failing += ASSERT_EQUAL(test_name, val_res_beg[7], 9);
            num_failing += ASSERT_EQUAL(test_name, val_res_beg[8], 1);
            num_failing += ASSERT_EQUAL(test_name, val_res_beg[9], 9);

            failed_tests += test_passed(num_failing, test_name);
        }
    }

    std::cout << std::endl << failed_tests << " failing test(s) detected." << std::endl;
    if (failed_tests == 0) {
        return 0;
    }
    return 1;
}
