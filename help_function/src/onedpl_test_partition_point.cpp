// ====------ onedpl_test_partition_point.cpp---------- -*- C++ -* ----===////
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

#include <sycl/sycl.hpp>

#include <iostream>
#include <iomanip>

struct less_than_n
{
private:
    int64_t n;

public:
    less_than_n(int64_t _n) : n(_n) {}

    bool operator()(const int64_t &x) const {
        return x < n;
    }
};

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

class PartitionPoint {};     // name for policy

int main() {

    // #22 PARTITION POINT TEST //

    // used to detect failure
    int failed_tests = 0;

    {
        // create buffer
        sycl::buffer<int64_t, 1> input_buf{ sycl::range<1>(8) };

        auto inp_it = oneapi::dpl::begin(input_buf);
        auto inp_end_it = oneapi::dpl::end(input_buf);

        {
            auto inp = inp_it.get_buffer().template get_access<sycl::access::mode::write>();
            for (int i = 0; i != 8; ++i) {
                inp[i] = i;
            }
        }

        // create named policy from existing one
        auto new_policy = oneapi::dpl::execution::make_device_policy<PartitionPoint>(oneapi::dpl::execution::dpcpp_default);

        // call algorithm:
        auto res_it = dpct::partition_point(new_policy, inp_it, inp_end_it, less_than_n(4));
        auto result = res_it.get_buffer().template get_access<sycl::access::mode::read>();

        failed_tests += ASSERT_EQUAL("Regular call to partition_point", result[std::distance(inp_it, res_it)], 4);
    }

    std::cout << std::endl << failed_tests << " failing test(s) detected." << std::endl;
    if (failed_tests == 0) {
        return 0;
    }
    return 1;
}
