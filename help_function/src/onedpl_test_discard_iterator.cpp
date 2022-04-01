// ====------ onedpl_test_discard_iterator.cpp---------- -*- C++ -* ----===////
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

int main() {

    // used to detect failures
    int num_failing = 0;
    int failed_tests = 0;

    {
        // create buffer
        cl::sycl::buffer<uint64_t, 1> src_buf{ cl::sycl::range<1>(8) };

        oneapi::dpl::discard_iterator dst;

        auto src_it = oneapi::dpl::begin(src_buf);
        auto src_end_it = oneapi::dpl::end(src_buf);

        {
            auto src = src_it.get_buffer().template get_access<cl::sycl::access::mode::write>();
            for (int i=0; i != 8; ++i) {
                src[i] = i;
            }
        }

        // create named policy from existing one
        auto new_policy = oneapi::dpl::execution::make_device_policy<class CopyDiscard>(oneapi::dpl::execution::dpcpp_default);

        // call algorithm:
        std::copy(new_policy, src_it, src_end_it, dst);

        // no data to check since discard_iterator drops all writes.
    }

    {
        // create buffer
        cl::sycl::buffer<uint64_t, 1> src_buf{ cl::sycl::range<1>(8) };
        cl::sycl::buffer<uint64_t, 1> dst_buf{ cl::sycl::range<1>(8) };

        auto src_it = oneapi::dpl::begin(src_buf);
        auto src_end_it = oneapi::dpl::end(src_buf);
        auto dst_it = oneapi::dpl::begin(dst_buf);
        auto dst_end_it = oneapi::dpl::end(dst_buf);

        oneapi::dpl::discard_iterator discard;

        {
            auto src = src_it.get_buffer().template get_access<cl::sycl::access::mode::write>();
            auto dst = dst_it.get_buffer().template get_access<cl::sycl::access::mode::write>();
            for (int i=0; i != 8; ++i) {
                src[i] = i;
                dst[i] = 0;
            }
        }

        // create named policy from existing one
        auto new_policy = oneapi::dpl::execution::make_device_policy<class CopyDiscardTuple>(oneapi::dpl::execution::dpcpp_default);
        // call algorithm:
        std::copy(new_policy, oneapi::dpl::make_zip_iterator(src_it, src_it), oneapi::dpl::make_zip_iterator(src_end_it, src_end_it), oneapi::dpl::make_zip_iterator(dst_it, discard));

        // check that dst is written correctly when discard_iterator is used.
        {
            auto dst = dst_it.get_buffer().template get_access<cl::sycl::access::mode::read>();
            for (int i=0; i != 8; ++i) {
                num_failing += ASSERT_EQUAL("PASS: discard_it test", dst[i], i);
            }
            if (num_failing == 0) {
                std::cout << "PASS: discard_it test" << std::endl;
            }
            else {
                ++failed_tests;
            }
        }
    }

    std::cout << std::endl << failed_tests << " failing test(s) detected." << std::endl;
    if (failed_tests == 0) {
        return 0;
    }
    return 1;
}
