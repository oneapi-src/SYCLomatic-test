// ====------ onedpl_test_transform_if.cpp---------- -*- C++ -* ----===////
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

struct is_odd
{
    bool operator()(const int64_t &x) const {
        return x % 2;
    }
};

struct is_even
{
    bool operator()(const int64_t &x) const {
        return (x % 2) == 0;
    }
};

struct add_two
{
    int operator()(int x) const {
        return x+2;
    }
};

struct add_tuple_components {
    template <typename Tuple>
    Tuple operator()(const Tuple& t, const Tuple& u) const {
        return Tuple(std::get<0>(t), std::get<0>(u));
    }
};

struct add_tuple_components2 {
    template <typename Scalar, typename Tuple>
    Scalar operator()(const Scalar s, const Tuple t) const {
        return s + std::get<0>(t) + std::get<1>(t);
    }
};

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

class Transform1 {};     // name for policy
class Transform2 {};     // name for policy
class Transform3 {};     // name for policy

int main() {

    // used to detect failures
    int failed_tests = 0;
    int num_failing = 0;
    std::string test_name = "";

    // #24 TRANSFORM IF TEST //

    {
        // create buffer
        sycl::buffer<int64_t, 1> dst_buf{ sycl::range<1>(8) };

        auto dst_it = oneapi::dpl::begin(dst_buf);
        auto dst_end_it = oneapi::dpl::end(dst_buf);

        {
            auto dst = dst_it.get_buffer().template get_access<sycl::access::mode::write>();
            dst[0] = -1; dst[1] = 2; dst[2] = -3; dst[3] = 4; dst[4] = -5; dst[5] = 6; dst[6] = -7; dst[7] = 8;
        }

        // create named policy from existing one
        auto new_policy = oneapi::dpl::execution::make_device_policy<Transform1>(oneapi::dpl::execution::dpcpp_default);

        // call algorithm:
        dpct::transform_if(new_policy, dst_it, dst_end_it, dst_it, std::negate<int64_t>(), is_odd());

        {
            test_name = "transform_if test 1";
            auto dst = dst_it.get_buffer().template get_access<sycl::access::mode::read>();
            num_failing += ASSERT_EQUAL(test_name, dst[0], 1);
            num_failing += ASSERT_EQUAL(test_name, dst[1], 2);
            num_failing += ASSERT_EQUAL(test_name, dst[2], 3);
            num_failing += ASSERT_EQUAL(test_name, dst[3], 4);
            num_failing += ASSERT_EQUAL(test_name, dst[4], 5);
            num_failing += ASSERT_EQUAL(test_name, dst[5], 6);
            num_failing += ASSERT_EQUAL(test_name, dst[6], 7);
            num_failing += ASSERT_EQUAL(test_name, dst[7], 8);

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
    }

    {
        // create buffer
        sycl::buffer<int64_t, 1> dst_buf{ sycl::range<1>(8) };
        sycl::buffer<int64_t, 1> stencil_buf{ sycl::range<1>(8) };

        auto dst_it = oneapi::dpl::begin(dst_buf);
        auto dst_end_it = oneapi::dpl::end(dst_buf);
        auto stn_it = oneapi::dpl::begin(stencil_buf);

        {
            auto dst = dst_it.get_buffer().template get_access<sycl::access::mode::write>();
            auto stn = stn_it.get_buffer().template get_access<sycl::access::mode::write>();
            dst[0] = -1; dst[1] = 2; dst[2] = -3; dst[3] = 4; dst[4] = -5; dst[5] = 6; dst[6] = -7; dst[7] = 8;
            stn[0] = 1; stn[1] = 1; stn[2] = 0; stn[3] = 0; stn[4] = 1; stn[5] = 1; stn[6] = 0; stn[7] = 0;
        }

        // create named policy from existing one
        auto new_policy = oneapi::dpl::execution::make_device_policy<Transform2>(oneapi::dpl::execution::dpcpp_default);

        // call algorithm:
        dpct::transform_if(new_policy, dst_it, dst_end_it, stn_it, dst_it, std::negate<int64_t>(),
                           oneapi::dpl::identity());

        {
            test_name = "transform_if test 2";
            auto dst = dst_it.get_buffer().template get_access<sycl::access::mode::read>();
            num_failing += ASSERT_EQUAL(test_name, dst[0], 1);
            num_failing += ASSERT_EQUAL(test_name, dst[1], -2);
            num_failing += ASSERT_EQUAL(test_name, dst[2], -3);
            num_failing += ASSERT_EQUAL(test_name, dst[3], 4);
            num_failing += ASSERT_EQUAL(test_name, dst[4], 5);
            num_failing += ASSERT_EQUAL(test_name, dst[5], -6);
            num_failing += ASSERT_EQUAL(test_name, dst[6], -7);
            num_failing += ASSERT_EQUAL(test_name, dst[7], 8);

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
    }

    {
        // create buffer
        sycl::buffer<int64_t, 1> dst_buf{ sycl::range<1>(8) };
        sycl::buffer<int64_t, 1> stencil_buf{ sycl::range<1>(8) };

        auto dst_it = oneapi::dpl::begin(dst_buf);
        auto dst_end_it = oneapi::dpl::end(dst_buf);
        auto stn_it = oneapi::dpl::begin(stencil_buf);

        {
            auto dst = dst_it.get_buffer().template get_access<sycl::access::mode::write>();
            auto stn = stn_it.get_buffer().template get_access<sycl::access::mode::write>();
            dst[0] = -1; dst[1] = 2; dst[2] = -3; dst[3] = 4; dst[4] = -5; dst[5] = 6; dst[6] = -7; dst[7] = 8;
            stn[0] = 1; stn[1] = 1; stn[2] = 0; stn[3] = 0; stn[4] = 1; stn[5] = 1; stn[6] = 0; stn[7] = 0;
        }

        // create named policy from existing one
        auto new_policy = oneapi::dpl::execution::make_device_policy<Transform3>(oneapi::dpl::execution::dpcpp_default);

        // call algorithm:
        dpct::transform_if(new_policy, dst_it, dst_end_it, dst_it, stn_it, dst_it,
                           std::multiplies<int64_t>(), oneapi::dpl::identity());

        {
            test_name = "transform_if test 3";
            auto dst = dst_it.get_buffer().template get_access<sycl::access::mode::read>();
            num_failing += ASSERT_EQUAL(test_name, dst[0], 1);
            num_failing += ASSERT_EQUAL(test_name, dst[1], 4);
            num_failing += ASSERT_EQUAL(test_name, dst[2], -3);
            num_failing += ASSERT_EQUAL(test_name, dst[3], 4);
            num_failing += ASSERT_EQUAL(test_name, dst[4], 25);
            num_failing += ASSERT_EQUAL(test_name, dst[5], 36);
            num_failing += ASSERT_EQUAL(test_name, dst[6], -7);
            num_failing += ASSERT_EQUAL(test_name, dst[7], 8);

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
    }

    // add 3 std::transform_if tests

    // test 1/4

    {
        sycl::buffer<int64_t, 1> src1_buf{ sycl::range<1>(8) };
        sycl::buffer<int64_t, 1> src2_buf{ sycl::range<1>(8) };
        sycl::buffer<int64_t, 1> src3_buf{ sycl::range<1>(8) };
        sycl::buffer<int64_t, 1> map_buf{ sycl::range<1>(4) };

        {
            auto src1 = src1_buf.template get_access<sycl::access::mode::write>();
            auto src2 = src2_buf.template get_access<sycl::access::mode::write>();
            auto src3 = src3_buf.template get_access<sycl::access::mode::write>();
            auto map = map_buf.template get_access<sycl::access::mode::write>();

            for (int i = 0; i != 8; ++i) {
                src1[i] = i;
                src2[i] = 7-i;
                src3[i] = i+5;
            }
            map[0] = 1; map[1] = 3; map[2] = 0; map[3] = 2;
        }

        auto src1_it = oneapi::dpl::begin(src1_buf);
        auto src2_it = oneapi::dpl::begin(src2_buf);
        auto src3_it = oneapi::dpl::begin(src3_buf);
        auto map_it = oneapi::dpl::begin(map_buf);

        {
            auto perm1 = oneapi::dpl::make_permutation_iterator(src1_it, map_it);
            auto perm2 = oneapi::dpl::make_permutation_iterator(src2_it, map_it);

            auto zip1 = oneapi::dpl::make_zip_iterator(perm1, perm2);
            auto zip2 = oneapi::dpl::make_zip_iterator(src3_it, src3_it+1);

            dpct::transform_if(oneapi::dpl::execution::dpcpp_default, zip1, zip1 + 4, zip2, map_it, zip1, add_tuple_components(), is_even());
        }

        auto src1 = src1_it.get_buffer().template get_access<sycl::access::mode::read>();
        auto src2 = src2_it.get_buffer().template get_access<sycl::access::mode::read>();
        auto map = map_it.get_buffer().template get_access<sycl::access::mode::write>();
        test_name = "transform_if with fancy iterators 1";

        for (int i = 0; i != 8; ++i) {
            num_failing += ASSERT_EQUAL(test_name, src1[i], i);
            if (i<4 && map[i]%2 == 0) {
                num_failing += ASSERT_EQUAL(test_name, src2[map[i]], i+5);
            }
            else {
                num_failing += ASSERT_EQUAL(test_name, src2[i], 7-i);
            }
        }

        failed_tests += test_passed(num_failing, test_name);
        num_failing = 0;
    }

// Comment out test using permutation_iterator until oneDPL is updated to better support use of
// transform_iterator as a component of the permutation_iterator
#if 0
    // test 2/4

    {
        sycl::buffer<int64_t, 1> src1_buf{ sycl::range<1>(8) };
        sycl::buffer<int64_t, 1> src2_buf{ sycl::range<1>(8) };
        sycl::buffer<int64_t, 1> map_buf{ sycl::range<1>(4) };

        {
            auto src1 = src1_buf.template get_access<sycl::access::mode::write>();
            auto src2 = src2_buf.template get_access<sycl::access::mode::write>();
            auto map = map_buf.template get_access<sycl::access::mode::write>();

            for (int i = 0; i != 8; ++i) {
                src1[i] = i;
                src2[i] = i+5;
            }
            map[0] = 2; map[1] = 5; map[2] = 4; map[3] = 3;
        }

        auto src1_it = oneapi::dpl::begin(src1_buf);
        auto src2_it = oneapi::dpl::begin(src2_buf);
        auto map_it = oneapi::dpl::begin(map_buf);

        {
            auto trf_it = oneapi::dpl::make_transform_iterator(map_it, add_two());
            auto perm_begin = oneapi::dpl::make_permutation_iterator(src1_it, trf_it);
            auto zip = oneapi::dpl::make_zip_iterator(src2_it, src2_it + 1);

            dpct::transform_if(oneapi::dpl::execution::dpcpp_default, perm_begin, perm_begin + 4, zip, map_it, perm_begin, add_tuple_components2(), is_even());
        }

        auto src1 = src1_it.get_buffer().template get_access<sycl::access::mode::read>();
        test_name = "transform_if with fancy iterators 2";

        for (int i = 0; i != 8; ++i) {
            if (i < 4 || i%2 == 0)
                num_failing += ASSERT_EQUAL(test_name, src1[i], i);
            else
                num_failing += ASSERT_EQUAL(test_name, src1[i], 11 + i*3);
        }

        failed_tests += test_passed(num_failing, test_name);
        num_failing = 0;

    }
#endif

    // test 3/4

    {
        sycl::buffer<int64_t, 1> src1_buf{ sycl::range<1>(8) };
        sycl::buffer<int64_t, 1> src2_buf{ sycl::range<1>(8) };
        sycl::buffer<int64_t, 1> map_buf{ sycl::range<1>(4) };

        {
            auto src1 = src1_buf.template get_access<sycl::access::mode::write>();
            auto src2 = src2_buf.template get_access<sycl::access::mode::write>();
            auto map = map_buf.template get_access<sycl::access::mode::write>();

            for (int i = 0; i != 8; ++i) {
                src1[i] = i;
                src2[i] = i+5;
            }
            map[0] = 7; map[1] = 5; map[2] = 4; map[3] = 3;
        }

        auto src1_it = oneapi::dpl::begin(src1_buf);
        auto src2_it = oneapi::dpl::begin(src2_buf);
        auto map_it = oneapi::dpl::begin(map_buf);

        {
            auto perm_begin = oneapi::dpl::make_permutation_iterator(src1_it, map_it);
            auto zip = oneapi::dpl::make_zip_iterator(src2_it, src2_it + 1);

            dpct::transform_if(oneapi::dpl::execution::dpcpp_default, perm_begin, perm_begin + 4, zip, map_it, perm_begin, add_tuple_components2(), is_odd());
        }

        auto src1 = src1_it.get_buffer().template get_access<sycl::access::mode::read>();
        auto map = map_it.get_buffer().template get_access<sycl::access::mode::read>();
        test_name = "transform_if with fancy iterators 3";

        for (int i = 0; i != 8; ++i) {
            if (i < 4 && map[i]%2 == 1) {
                num_failing += ASSERT_EQUAL(test_name, src1[map[i]], 11 + i*2 + map[i]);
            }
            else if (i < 4 && map[i]%2 == 0) {
                num_failing += ASSERT_EQUAL(test_name, src1[map[i]], map[i]);
            }
        }

        failed_tests += test_passed(num_failing, test_name);
        num_failing = 0;
    }

    // test 4/4

    {
        sycl::buffer<int64_t, 1> src1_buf{ sycl::range<1>(8) };
        sycl::buffer<int64_t, 1> src2_buf{ sycl::range<1>(8) };
        sycl::buffer<int64_t, 1> map_buf{ sycl::range<1>(4) };

        {
            auto src1 = src1_buf.template get_access<sycl::access::mode::write>();
            auto src2 = src2_buf.template get_access<sycl::access::mode::write>();
            auto map = map_buf.template get_access<sycl::access::mode::write>();

            for (int i = 0; i != 8; ++i) {
                src1[i] = i;
                src2[i] = 7-i;
            }
            map[0] = 6; map[1] = 4; map[2] = 5; map[3] = 3;
        }

        auto src1_it = oneapi::dpl::begin(src1_buf);
        auto src2_it = oneapi::dpl::begin(src2_buf);
        auto map_it = oneapi::dpl::begin(map_buf);

        {
            dpct::transform_if
            (
                oneapi::dpl::execution::dpcpp_default,
                src1_it,
                src1_it + 4,
                src1_it + 1,
                map_it,
                oneapi::dpl::make_permutation_iterator(src2_it, map_it),
                std::plus<int>(),
                is_even()
            );
        }

        {
            auto src2 = src2_it.get_buffer().template get_access<sycl::access::mode::read>();
            test_name = "transform_if with fancy iterators 4";

            for (int i = 0; i != 8; ++i) {
                num_failing += ASSERT_EQUAL(test_name, src2[i], 7-i);
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }

    }

    std::cout << std::endl << failed_tests << " failing test(s) detected." << std::endl;
    if (failed_tests == 0) {
        return 0;
    }
    return 1;
}
