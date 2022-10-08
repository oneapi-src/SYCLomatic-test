// ====------ onedpl_test_copy.cpp---------- -*- C++ -* ----===////
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

struct calcFaceDeltaFunctor
{
    template <typename T>
    typename std::remove_reference<typename std::tuple_element<0, typename std::decay<T>::type>::type>::type
    operator()(T&& t) const
    {
        return std::get<0>(t) - std::get<1>(t);
    }
};

int main() {
    // used to detect failures
    int failed_tests = 0;
    int num_failing = 0;

    // First Group: Testing regular calls to std::copy
    {
        // test 1/4

        // create buffers
        sycl::buffer<uint64_t, 1> src_buf { sycl::range<1>(8) };
        sycl::buffer<uint64_t, 1> dst_buf { sycl::range<1>(8) };

        auto src_it = oneapi::dpl::begin(src_buf);
        auto src_end_it = oneapi::dpl::end(src_buf);
        auto dst_it = oneapi::dpl::begin(dst_buf);
        auto dst_end_it = oneapi::dpl::end(dst_buf);

        iota_buffer(src_buf, 0, 8);
        fill_buffer(dst_buf, 0, 8, 10);

        // call algorithm
        std::copy(oneapi::dpl::execution::dpcpp_default, src_it, src_it + 4, dst_it);

        {
            std::string test_name = "Regular call to std::copy 1/4";
            auto dst = dst_it.get_buffer().template get_access<sycl::access::mode::read>();

            for (int i = 0; i != 8; ++i) {
                if (i < 4)
                    num_failing += ASSERT_EQUAL(test_name, dst[i], i);
                else
                    num_failing += ASSERT_EQUAL(test_name, dst[i], 10);
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }

        // test 2/4

        iota_buffer(src_buf, 0, 8);
        fill_buffer(dst_buf, 0, 8, 10);

        // call algorithm
        std::copy(oneapi::dpl::execution::dpcpp_default, src_it + 4, src_end_it, dst_it);

        {
            std::string test_name = "Regular call to std::copy 2/4";
            auto dst = dst_it.get_buffer().template get_access<sycl::access::mode::read>();

            for (int i = 0; i != 8; ++i) {
                if (i < 4)
                    num_failing += ASSERT_EQUAL(test_name, dst[i], i + 4);
                else
                    num_failing += ASSERT_EQUAL(test_name, dst[i], 10);
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }

        // test 3/4

        iota_buffer(src_buf, 0, 8);
        fill_buffer(dst_buf, 0, 8, 10);

        // call algorithm
        std::copy(oneapi::dpl::execution::dpcpp_default, src_it + 2, src_it + 6, dst_it);

        {
            std::string test_name = "Regular call to std::copy 3/4";
            auto dst = dst_it.get_buffer().template get_access<sycl::access::mode::read>();

            for (int i = 0; i != 8; ++i) {
                if (i < 4)
                    num_failing += ASSERT_EQUAL(test_name, dst[i], i + 2);
                else
                    num_failing += ASSERT_EQUAL(test_name, dst[i], 10);
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }

        // test 4/4

        iota_buffer(src_buf, 0, 8);
        fill_buffer(dst_buf, 0, 8, 10);

        // call algorithm
        std::copy(oneapi::dpl::execution::dpcpp_default, src_it, src_it + 4, dst_it + 4);

        {
            std::string test_name = "Regular call to std::copy 4/4";
            auto dst = dst_it.get_buffer().template get_access<sycl::access::mode::read>();

            for (int i = 0; i != 8; ++i) {
                if (i > 3)
                    num_failing += ASSERT_EQUAL(test_name, dst[i], i - 4);
                else
                    num_failing += ASSERT_EQUAL(test_name, dst[i], 10);
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
    }

    // Second Group: Testing calls to std::copy using make_permutation_iterator

    {
        // test 1/2

        // create buffers
        sycl::buffer<uint64_t, 1> src_buf{ sycl::range<1>(8) };
        sycl::buffer<uint64_t, 1> map_buf{ sycl::range<1>(4) };
        sycl::buffer<uint64_t, 1> dst_buf{ sycl::range<1>(8) };

        auto src_it = oneapi::dpl::begin(src_buf);
        auto map_it = oneapi::dpl::begin(map_buf);
        auto dst_it = oneapi::dpl::begin(dst_buf);

        {
            iota_buffer(src_buf, 0, 8);
            fill_buffer(dst_buf, 0, 8, 0);
            auto map = map_buf.template get_access<sycl::access::mode::write>();

            map[0] = 7; map[1] = 6; map[2] = 5; map[3] = 4;
            // src buffer: { 0, 1, 2, 3, 4, 5, 6, 7 }
            // map buffer: { 7, 6, 5, 4 }
            // dst buffer: { 0, 0, 0, 0, 0, 0, 0, 0 }
        }

        {
            auto perm_begin = oneapi::dpl::make_permutation_iterator(src_it, map_it);
            auto perm_end = perm_begin + 4;

            // call algorithm
            std::copy(oneapi::dpl::execution::dpcpp_default, perm_begin, perm_end, dst_it);
        }

        {
            std::string test_name = "std::copy using make_perm_it 1/2";
            auto dst = dst_it.get_buffer().template get_access<sycl::access::mode::read>();
            auto map = map_it.get_buffer().template get_access<sycl::access::mode::read>();

            for (int i = 0; i != 8; ++i) {
                if (i < 4)
                    num_failing += ASSERT_EQUAL(test_name, dst[i], map[i]);
                else
                    num_failing += ASSERT_EQUAL(test_name, dst[i], 0);
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }

        // test 2/2

        {
            auto map = map_buf.template get_access<sycl::access::mode::write>();
            auto src = src_buf.template get_access<sycl::access::mode::write>();

            src[0] = 3; src[1] = 2; src[2] = 6; src[3] = 20; src[4] = 5; src[5] = 0; src[6] = 3; src[7] = 55;
            map[0] = 1; map[1] = 5; map[2] = 7; map[3] = 2;
            fill_buffer(dst_buf, 0, 8, 0);

            // src buffer: { 3, 2, 6, 20, 5, 0, 3, 55 }
            // map buffer: { 1, 5, 7, 2 }
            // dst buffer: { 0, 0, 0, 0, 0, 0, 0, 0 }
        }

        {
            auto perm_begin = oneapi::dpl::make_permutation_iterator(src_it, map_it);
            auto perm_end = perm_begin + 4;

            // call algorithm
            std::copy(oneapi::dpl::execution::dpcpp_default, perm_begin, perm_end, dst_it + 4);
        }

        {
            std::string test_name = "std::copy using make_perm_it 2/2";
            auto dst = dst_it.get_buffer().template get_access<sycl::access::mode::read>();

            for (int i = 0; i != 8; ++i) {
                if (i == 4)
                    num_failing += ASSERT_EQUAL(test_name, dst[i], 2);
                else if (i == 6)
                    num_failing += ASSERT_EQUAL(test_name, dst[i], 55);
                else if (i == 7)
                    num_failing += ASSERT_EQUAL(test_name, dst[i], 6);
                else
                    num_failing += ASSERT_EQUAL(test_name, dst[i], 0);
            }

            // post-condition: dst buffer = { 0, 0, 0, 0, 2, 0, 55, 6 }
            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
    }

    // Third Group: Testing call to std::copy using make_counting_iterator

    {
        // test 1/2

        // create buffer
        sycl::buffer<uint64_t, 1> dst_buf{ sycl::range<1>(8) };

        fill_buffer(dst_buf, 0, 8, 10);

        auto dst_it = oneapi::dpl::begin(dst_buf);

        // call algorithm
        std::copy(oneapi::dpl::execution::dpcpp_default, dpct::make_counting_iterator(0), dpct::make_counting_iterator(4), dst_it);

        {
            std::string test_name = "std::copy using make_counting_it 1/2";
            auto dst = dst_it.get_buffer().template get_access<sycl::access::mode::read>();

            for (int i = 0; i != 8; ++i) {
                if (i < 4)
                    num_failing += ASSERT_EQUAL(test_name, dst[i], i);
                else
                    num_failing += ASSERT_EQUAL(test_name, dst[i], 10);
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }

        // test 2/2

        fill_buffer(dst_buf, 0, 8, 10);

        // call algorithm
        std::copy(oneapi::dpl::execution::dpcpp_default, dpct::make_counting_iterator(0), dpct::make_counting_iterator(4), dst_it + 4);

        {
            std::string test_name = "std::copy using make_counting_it 2/2";
            auto dst = dst_it.get_buffer().template get_access<sycl::access::mode::read>();

            for (int i = 0; i != 8; ++i) {
                if (i < 4)
                    num_failing += ASSERT_EQUAL(test_name, dst[i], 10);
                else
                    num_failing += ASSERT_EQUAL(test_name, dst[i], i - 4);
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
    }

    // Fourth Group: Testing call to std::copy using counting_iterator<T>

    {
        // test 1/1

        // create buffer
        sycl::buffer<uint64_t, 1> dst_buf{ sycl::range<1>(8) };

        fill_buffer(dst_buf, 0, 8, 10);

        auto dst_it = oneapi::dpl::begin(dst_buf);

        // call algorithm
        std::copy(oneapi::dpl::execution::dpcpp_default, oneapi::dpl::counting_iterator<int>(0), oneapi::dpl::counting_iterator<int>(4), dst_it);

        {
            std::string test_name = "std::copy using counting_it<T>";
            auto dst = dst_it.get_buffer().template get_access<sycl::access::mode::read>();

            for (int i = 0; i != 8; ++i) {
                if (i < 4)
                    num_failing += ASSERT_EQUAL(test_name, dst[i], i);
                else
                    num_failing += ASSERT_EQUAL(test_name, dst[i], 10);
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
    }

    // Fifth Group: Testing call to std::copy using make_transform_iterator

    {
        // test 1/1

        // create buffers
        sycl::buffer<uint64_t, 1> src_buf{ sycl::range<1>(8) };
        sycl::buffer<uint64_t, 1> map_buf{ sycl::range<1>(4) };
        sycl::buffer<uint64_t, 1> dst_buf{ sycl::range<1>(8) };
        sycl::buffer<uint64_t, 1> map2_buf{ sycl::range<1>(4) };

        {
            auto map = map_buf.template get_access<sycl::access::mode::write>();
            auto map2 = map2_buf.template get_access<sycl::access::mode::write>();

            iota_buffer(src_buf, 0, 8);
            map[0] = 3; map[1] = 0; map[2] = 2; map[3] = 1;
            map2[0] = 7; map2[1] = 6; map2[2] = 5; map2[3] = 4;
            fill_buffer(dst_buf, 0, 8, 10);

            // src buffer { 0, 1, 2, 3, 4, 5, 6, 7 }
            // map buffer { 3, 0, 2, 1 }
            // map2 buffer { 7, 6, 5, 4 }
        }

        auto src_it = oneapi::dpl::begin(src_buf);
        auto map_it = oneapi::dpl::begin(map_buf);
        auto dst_it = oneapi::dpl::begin(dst_buf);
        auto map2_it = oneapi::dpl::begin(map2_buf);
        auto map_end_it = oneapi::dpl::end(map_buf);
        auto map2_end_it = oneapi::dpl::end(map2_buf);

        {
            auto perm_begin = oneapi::dpl::make_permutation_iterator(src_it, map_it);
            auto perm_end = perm_begin + 4;
            auto perm_begin2 = oneapi::dpl::make_permutation_iterator(src_it, map2_it);
            auto perm_end2 = perm_begin2 + 4;

            std::copy
            (
                oneapi::dpl::execution::dpcpp_default,
                oneapi::dpl::make_transform_iterator
                (
                    oneapi::dpl::make_zip_iterator
                    (
                        perm_begin,
                        perm_begin2
                    ),
                    calcFaceDeltaFunctor()
                ),
                oneapi::dpl::make_transform_iterator
                (
                    oneapi::dpl::make_zip_iterator
                    (
                        perm_end,
                        perm_end2
                    ),
                    calcFaceDeltaFunctor()
                ),
                dst_it
            );
        }

        {
            std::string test_name = "std::copy using make_trf_it";
            auto dst = dst_buf.template get_access<sycl::access::mode::read>();

            for (int i = 0; i != 8; ++i) {
                if (i == 0) {
                    num_failing += ASSERT_EQUAL(test_name, dst[i], -4);
                }
                else if (i == 1) {
                    num_failing += ASSERT_EQUAL(test_name, dst[i], -6);
                }
                else if (i == 2 || i == 3) {
                    num_failing += ASSERT_EQUAL(test_name, dst[i], -3);
                }
                else {
                    num_failing += ASSERT_EQUAL(test_name, dst[i], 10);
                }
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
    }

    // Sixth Group: Testing call to std::copy using device vector

    {
        // test 1/1

        // create host arrays and device vector
        std::vector<int> hostArray(8);
        std::vector<int64_t> src(8);
        src[0] = -1; src[1] = 2; src[2] = -3; src[3] = 4; src[4] = -5; src[5] = 6; src[6] = -7; src[7] = 8;
        dpct::device_vector<int64_t> dv = src;

        // fill hostArray with 10s, dv with index values
        for (int i = 0; i !=8; ++i) {
            hostArray[i] = 10;
        }

        {
            // call algorithm
            std::copy(oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()), dv.begin()+2, dv.begin()+6, hostArray.begin());
        }
        dpct::get_default_queue().wait();

        std::string test_name = "std::copy using device_vector";

        for (int i = 0; i != 8; ++i) {
            if (i < 4)
                num_failing += ASSERT_EQUAL(test_name, hostArray[i], i%2 == 0 ? -(i + 3) : i + 3);
            else
                num_failing += ASSERT_EQUAL(test_name, hostArray[i], 10);
        }

        failed_tests += test_passed(num_failing, test_name);
        num_failing = 0;
    }

    std::cout << std::endl << failed_tests << " failing test(s) detected." << std::endl;
    if (failed_tests == 0) {
        return 0;
    }
    return 1;
}
