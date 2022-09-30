// ====------ onedpl_test_merge_by_key.cpp---------- -*- C++ -* ----===////
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

template<typename _T1, typename _T2>
void ASSERT_EQUAL(_T1&& X, _T2&& Y) {
    if(X!=Y)
        std::cout << "CHECK CORRECTNESS (PSTL WITH SYCL): fail (" << X << "," << Y << ")" << std::endl;
}

class MergeByKey {};        // name for policy
class MergeByKeyGreater {}; // name for policy

// comparator implementing operator>
class greater
{
public:
    explicit greater() {}

    template <typename _Xp, typename _Yp>
    bool operator()(_Xp&& __x, _Yp&& __y) const {
        return std::forward<_Xp>(__x) > std::forward<_Yp>(__y);
    }
};

int main() {

    // #5 MERGE BY KEY TEST //

    // used to detect failures
    int failed_tests = 0;

    // input on host
    std::vector<uint64_t> key1 = {1, 3, 5, 7, 9, 11};
    std::vector<uint64_t> key2 = {1, 1, 2, 3, 5, 8, 13};
    std::vector<uint64_t> val1 = {0, 0, 0, 0, 0, 0};
    std::vector<uint64_t> val2 = {1, 1, 1, 1, 1, 1, 1};
    std::vector<uint64_t> keys = {9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9};
    std::vector<uint64_t> vals = {9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9};

    {
        // create buffers
        sycl::buffer<uint64_t, 1> key_buf1{ key1.data(), sycl::range<1>(6) };
        sycl::buffer<uint64_t, 1> key_buf2{ key2.data(), sycl::range<1>(7) };
        sycl::buffer<uint64_t, 1> val_buf1{ val1.data(), sycl::range<1>(6) };
        sycl::buffer<uint64_t, 1> val_buf2{ val2.data(), sycl::range<1>(7) };
        sycl::buffer<uint64_t, 1> key_res_buf{ keys.data(), sycl::range<1>(13) };
        sycl::buffer<uint64_t, 1> val_res_buf{ vals.data(), sycl::range<1>(13) };

        // create sycl iterators
        auto key_beg1 = oneapi::dpl::begin(key_buf1);
        auto key_end1 = oneapi::dpl::end(key_buf1);
        auto key_beg2 = oneapi::dpl::begin(key_buf2);
        auto key_end2 = oneapi::dpl::end(key_buf2);
        auto val_beg1 = oneapi::dpl::begin(val_buf1);
        auto val_beg2 = oneapi::dpl::begin(val_buf2);
        auto key_res_beg = oneapi::dpl::begin(key_res_buf);
        auto val_res_beg = oneapi::dpl::begin(val_res_buf);

        // create named policy from existing one
        auto new_policy = oneapi::dpl::execution::make_device_policy<MergeByKey>(oneapi::dpl::execution::dpcpp_default);

        // call algorithm
	    dpct::merge(new_policy, key_beg1, key_end1, key_beg2, key_end2, val_beg1, val_beg2,
                             key_res_beg, val_res_beg);

	// check values
        // keys_result = {1, 1, 1, 2, 3, 3, 5, 5, 7, 8, 9, 11, 13}
        // vals_result = {0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0,  0,  1}

        bool pass = true;

	auto key_acc = key_res_buf.template get_access<sycl::access::mode::read>();
	auto res_acc = val_res_buf.template get_access<sycl::access::mode::read>();

        for (int i = 1; i != 13; ++i) {
            pass = pass && key_acc[i-1] <= key_acc[i];
            if (key_acc[i-1] == key_acc[i])
                pass = pass && res_acc[i-1] <= res_acc[i];
        }

        if (!pass) {
            ++failed_tests;
            std::cout << "FAIL: merge_by_key default comparator" << std::endl;
        }
        else {
            std::cout << "PASS: merge_by_key default comparator" << std::endl;
        }
    }

    // reinitialize input for merge using operator>
    key1 = {11, 9, 7, 5, 3, 1};
    key2 = {13, 8, 5, 3, 2, 1, 1};

    for (int i = 0; i != 13; ++i) {
        keys[i] = 9;
        vals[i] = 9;
    }

    {
        // create buffers
        sycl::buffer<uint64_t, 1> key_buf1{ key1.data(), sycl::range<1>(6) };
        sycl::buffer<uint64_t, 1> key_buf2{ key2.data(), sycl::range<1>(7) };
        sycl::buffer<uint64_t, 1> val_buf1{ val1.data(), sycl::range<1>(6) };
        sycl::buffer<uint64_t, 1> val_buf2{ val2.data(), sycl::range<1>(7) };
        sycl::buffer<uint64_t, 1> key_res_buf{ keys.data(), sycl::range<1>(13) };
        sycl::buffer<uint64_t, 1> val_res_buf{ vals.data(), sycl::range<1>(13) };

        // create sycl iterators
        auto key_beg1 = oneapi::dpl::begin(key_buf1);
        auto key_end1 = oneapi::dpl::end(key_buf1);
        auto key_beg2 = oneapi::dpl::begin(key_buf2);
        auto key_end2 = oneapi::dpl::end(key_buf2);
        auto val_beg1 = oneapi::dpl::begin(val_buf1);
        auto val_beg2 = oneapi::dpl::begin(val_buf2);
        auto key_res_beg = oneapi::dpl::begin(key_res_buf);
        auto val_res_beg = oneapi::dpl::begin(val_res_buf);

        // create named policy from existing one.
        auto new_policy2 = oneapi::dpl::execution::make_device_policy<MergeByKeyGreater>(oneapi::dpl::execution::dpcpp_default);

        // call algorithm
	    dpct::merge(new_policy2, key_beg1, key_end1, key_beg2, key_end2, val_beg1, val_beg2,
                             key_res_beg, val_res_beg, greater());

	    // check values
        // keys_result = {13, 11, 9, 8, 7, 5, 5, 3, 3, 2, 1, 1, 1}
        // vals_result = { 1,  0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1}

        bool pass = true;

	auto key_acc = key_res_buf.template get_access<sycl::access::mode::read>();
	auto res_acc = val_res_buf.template get_access<sycl::access::mode::read>();

        for (int i = 1; i != 13; ++i) {
          pass = pass && key_acc[i-1] >= key_acc[i];

          if (key_acc[i-1] == key_acc[i])
            pass = pass && res_acc[i-1] <= res_acc[i];
        }

        if (!pass) {
          ++failed_tests;
          std::cout << "FAIL: merge_by_key user comparator" << std::endl;
        }
        else {
          std::cout << "PASS: merge_by_key user comparator" << std::endl;
        }
    }

    std::cout << std::endl << failed_tests << " failing test(s) detected." << std::endl;
    if (failed_tests == 0) {
        return 0;
    }
    return 1;
}
