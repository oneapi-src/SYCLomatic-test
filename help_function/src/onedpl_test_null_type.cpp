// ====------ onedpl_test_reduce.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include "oneapi/dpl/execution"

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

template <typename KeyT>
int reorder_key(KeyT& a, KeyT& b)
{
	if (b < a)
	{
		::std::swap(a, b);
	}
	// returns 1 if reorder key is used
	return 1;
}

//shows example usage of dpct::null_type, this has actual ValueT arguments
template <typename KeyT, typename ValueT>
typename ::std::enable_if<!::std::is_same<ValueT,dpct::null_type>::value, int>::type 
reorder_pair(KeyT& a_key, KeyT& b_key, ValueT& a_val, ValueT& b_val)
{
	if (b_key < a_key)
	{
		::std::swap(a_key,b_key);
		::std::swap(a_val,b_val);
	}
	//returns 2 if reorder_pair is used
	return 2;
}

//shows example usage of dpct::null_typeas an indicator to convert to key only
template <typename KeyT, typename ValueT>
typename ::std::enable_if<::std::is_same<ValueT,dpct::null_type>::value, int>::type 
reorder_pair(KeyT& a_key, KeyT& b_key, ValueT, ValueT)
{
	return reorder_key(a_key, b_key);
}

int main() {

    // used to detect failures
    int failed_tests = 0;
    int num_failing = 0;
    std::string test_name = "";

    {
        // test normal usage
        int a = 5;
        int b = 3;
        int64_t a_val = -5;
        int64_t b_val = -3;
        int ret = reorder_pair(a, b, a_val, b_val);
        bool result = (ret == 2) && a == 3 && b == 5 && a_val == -3 && b_val == -5;
        test_name = "Testing normal usage of helpers 1/2";
        failed_tests += ASSERT_EQUAL(test_name, result, true);

        a = 5;
        b = 3;
        a_val = -5;
        b_val = -3;
        ret = reorder_key(a, b);
        result = (ret == 1) && a == 3 && b == 5 && a_val == -5 && b_val == -3;
        test_name = "Testing normal usage of helpers 2/2";
        failed_tests += ASSERT_EQUAL(test_name, result, true);

        // test null_type redirect
        a = 5;
        b = 3;
        a_val = -5;
        b_val = -3;
        ret = reorder_pair(a, b, dpct::null_type{}, dpct::null_type{});
        result = (ret == 1) && a == 3 && b == 5 && a_val == -5 && b_val == -3;
        test_name = "Testing null_type redirect";
        failed_tests += ASSERT_EQUAL(test_name, result, true);

    }


    std::cout << std::endl << failed_tests << " failing test(s) detected." << std::endl;
    if (failed_tests == 0) {
        return 0;
    }
    return 1;
}
