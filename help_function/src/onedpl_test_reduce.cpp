// ====------ onedpl_test_reduce.cpp---------- -*- C++ -* ----===////
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
    else {
        std::cout << "PASS: " << msg << std::endl;
        return 0;
    }
}

template<typename Buffer> void fill_buffer(Buffer& src_buf, int start_index, int end_index, uint64_t value) {
    auto src = src_buf.template get_access<sycl::access::mode::write>();
    for (int i = start_index; i != end_index; ++i) {
        src[i] = value;
    }
}

template<typename Buffer> void iota_buffer(Buffer& dst_buf, int start_index, int end_index) {
    auto dst = dst_buf.template get_access<sycl::access::mode::write>();
    for (int i = start_index; i != end_index; ++i) {
        dst[i] = i;
    }
}

struct oppositeFunctor {
    std::tuple<bool, bool, bool> operator()(const std::tuple<bool, bool, bool> &t) const {
        return std::tuple<bool, bool, bool>
        (
            !std::get<0>(t),
            !std::get<1>(t),
            !std::get<2>(t)
        );
    }
};

struct andBooleanFunctor {
    std::tuple<bool, bool, bool> operator()(const std::tuple<bool, bool, bool>& t1, const std::tuple<bool, bool, bool>& t2) const {
        return std::tuple<bool, bool, bool>
        (
            std::get<0>(t1) && std::get<0>(t2),
            std::get<1>(t1) && std::get<1>(t2),
            std::get<2>(t1) && std::get<2>(t2)
        );
    }
};

struct square {
    int operator()(int x) const {
        return x*x;
    }
};

struct multiply {
    template <typename T>
    typename std::remove_reference<typename std::tuple_element<0, typename std::decay<T>::type>::type>::type
    operator()(T&& t) const {
        return std::get<0>(t) * std::get<1>(t);
    }
};

int main() {

    // used to detect failures
    int failed_tests = 0;
    int num_failing = 0;
    std::string test_name = "";

    // First Group: Testing regular calls to std::reduce
    {
        // test 1/2

        // create buffer
        sycl::buffer<uint64_t, 1> src_buf { sycl::range<1>(8) };

        auto src_it = oneapi::dpl::begin(src_buf);

        iota_buffer(src_buf, 0, 8);

        // call algorithm
        auto result = std::reduce(oneapi::dpl::execution::dpcpp_default, src_it, src_it + 4, 0, std::plus<int>());

        test_name = "Regular call to std::reduce 1/2";
        failed_tests += ASSERT_EQUAL(test_name, result, 6);

        // test 2/2

        // call algorithm
        result = std::reduce(oneapi::dpl::execution::dpcpp_default, src_it + 2, src_it + 6);

        test_name = "Regular call to std::reduce 2/2";
        failed_tests += ASSERT_EQUAL(test_name, result, 14);
    }

    // Second Group: Testing calls to std::reduce using make_transform_iterator

    {
        // test 1/3

        // create buffer
        sycl::buffer<std::tuple<bool, bool, bool>, 1> src_buf { sycl::range<1>(3) };

        auto src_it = oneapi::dpl::begin(src_buf);
        auto src_end_it = oneapi::dpl::end(src_buf);

        {
            auto src = src_buf.template get_access<sycl::access::mode::write>();
            src[0] = std::tuple<bool, bool, bool>(false, false, false);
            src[1] = std::tuple<bool, bool, bool>(false, true, true);
            src[2] = std::tuple<bool, bool, bool>(false, false, true);
        }

        {
            // call algorithm
            auto result = std::reduce
            (
                oneapi::dpl::execution::dpcpp_default,
                oneapi::dpl::make_transform_iterator
                (
                    src_it,
                    oppositeFunctor()
                ),
                oneapi::dpl::make_transform_iterator
                (
                    src_end_it,
                    oppositeFunctor()
               ),
                std::tuple<bool, bool, bool>(true, true, true),
                andBooleanFunctor()
            );

            test_name = "std::reduce with make_transform_iterator and tuple output";
            failed_tests += ASSERT_EQUAL(test_name, std::get<0>(result), true);
            failed_tests += ASSERT_EQUAL(test_name, std::get<1>(result), false);
            failed_tests += ASSERT_EQUAL(test_name, std::get<2>(result), false);
        }


        // test 2/3

        // call algorithm
        auto result = std::reduce
        (
            oneapi::dpl::execution::dpcpp_default,
            oneapi::dpl::make_transform_iterator
            (
                dpct::make_counting_iterator(4),
                square()
            ),
            oneapi::dpl::make_transform_iterator
            (
                dpct::make_counting_iterator(8),
                square()
            )
        );

        test_name = "std::reduce with make_transform_iterator 2";
        failed_tests += ASSERT_EQUAL(test_name, result, 126);

        // test 3/3

        // create buffer
        sycl::buffer<uint64_t, 1> rand_buf { sycl::range<1>(8) };
        sycl::buffer<uint64_t, 1> field_buf { sycl::range<1>(8) };

        auto rand_it = oneapi::dpl::begin(rand_buf);
        auto field_it = oneapi::dpl::begin(field_buf);

        {
            auto rand = rand_buf.template get_access<sycl::access::mode::write>();
            auto field = field_buf.template get_access<sycl::access::mode::write>();

            rand[0] = 4; rand[1] = 1; rand[2] = 9; rand[3] = 8; rand[4] = 5; rand[5] = 3; rand[6] = 0; rand[7] = 7;

            field[0] = 7; field[1] = 3; field[2] = 2; field[3] = 4; field[4] = 0; field[5] = 9; field[6] = 8; field[7] = 5;

            // rand: { 4, 1, 9, 8, 5, 3, 0, 7 }
            // field: { 7, 3, 2, 4, 0, 9, 8, 5 }
        }


        // call algorithm
        result = std::reduce
        (
            oneapi::dpl::execution::dpcpp_default,
            oneapi::dpl::make_transform_iterator
            (
                oneapi::dpl::make_zip_iterator
                (
                    rand_it + 2,
                    field_it + 2
                ),
                multiply()
            ),
            oneapi::dpl::make_transform_iterator
            (
                oneapi::dpl::make_zip_iterator
                (
                    rand_it + 6,
                    field_it + 6
                ),
                multiply()
            ),
            0.0,
            std::plus<int>()
        );

        test_name = "std::reduce with make_transform_iterator 3";
        failed_tests += ASSERT_EQUAL(test_name, result, 77);
    }

    std::cout << std::endl << failed_tests << " failing test(s) detected." << std::endl;
    if (failed_tests == 0) {
        return 0;
    }
    return 1;
}
