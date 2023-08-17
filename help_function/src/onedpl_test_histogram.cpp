// ====------ onedpl_test_histogram.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include "dpct/dpct.hpp"
#include "dpct/dpl_utils.hpp"
#include <iostream>

template <typename _T>
using Vector = dpct::device_vector<_T>;

template <typename String, typename _T1, typename _T2>
int
ASSERT_EQUAL(String msg, _T1&& X, _T2&& Y)
{
    if (X != Y)
    {
        std::cout << "FAIL: " << msg << " - (" << X << "," << Y << ")" << std::endl;
        return 1;
    }
    else
    {
        std::cout << "PASS: " << msg << std::endl;
        return 0;
    }
}

template <typename String, typename _T1, typename _T2, typename _OffsetT>
int
ASSERT_ARRAY_EQUAL(String msg, _T1&& X, _T2&& Y, _OffsetT idx)
{
    if (X != Y)
    {
        std::cout << "FAIL: " << msg << " - (" << X << "," << Y << ") idx = " << idx << std::endl;
        return 1;
    }
    return 0;
}

int
test_passed(int failing_elems, std::string test_name)
{
    if (failing_elems == 0)
    {
        std::cout << "PASS: " << test_name << std::endl;
        return 0;
    }
    return 1;
}

int
main()
{

    // used to detect failures
    int failed_tests = 0;
    int num_failing = 0;
    std::string test_name = "";

    {
        std::vector<int> vec({1, 2, 3, 5, 15, 22, 23, 24, 25, 99});
        dpct::device_vector<int> dvec(vec.begin(), vec.end());
        dpct::device_vector<int> bins(10, 0);
        dpct::histogram_even(oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()), dvec.begin(),
                            bins.begin(), bins.size() + 1, 0, 100, dvec.size());

        std::vector<int> expected_bins = {4, 1, 4, 0, 0, 0, 0, 0, 0, 1};

        std::string test_name = "histogram_even";
        for (int i = 0; i < bins.size(); i++)
        {
            num_failing += ASSERT_ARRAY_EQUAL(test_name, expected_bins[i], bins[i], i);
        }
        failed_tests += test_passed(num_failing, test_name);
    }

    {
        std::vector<int> vec({1, 2, 3, 5, 15, 22, 23, 24, 25, 99});
        dpct::device_vector<int> dvec(vec.begin(), vec.end());
        std::vector<int> levels({0, 4, 20, 55, 100});
        dpct::device_vector<int> dlevels(levels.begin(), levels.end());
        dpct::device_vector<int> bins(4, 0);
        dpct::histogram_range(oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()), dvec.begin(),
                             bins.begin(), dlevels.size(), dlevels.begin(), dvec.size());

        std::vector<int> expected_bins = {3, 2, 4, 1};

        std::string test_name = "histogram_range";
        for (int i = 0; i < bins.size(); i++)
        {
            num_failing += ASSERT_ARRAY_EQUAL(test_name, expected_bins[i], bins[i], i);
        }
        failed_tests += test_passed(num_failing, test_name);
    }

    {
        std::vector<int> vec({1, 2, 3, 5, 15,
                              22, 23, 24, 25, 99,
                              33, 34, 35, 35, 99});
        dpct::device_vector<int> dvec(vec.begin(), vec.end());
        dpct::device_vector<int> bins(10, 0);

        int num_cols = 3;
        int num_rows = 2;
        int row_stride_bytes = 5 * sizeof(int);

        dpct::histogram_even_roi(oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()), dvec.begin(),
                            bins.begin(), bins.size() + 1, 0, 100, num_cols, num_rows, row_stride_bytes);

        std::vector<int> expected_bins = {3, 0, 3, 0, 0, 0, 0, 0, 0, 0};

        std::string test_name = "histogram_even_roi";
        for (int i = 0; i < bins.size(); i++)
        {
            num_failing += ASSERT_ARRAY_EQUAL(test_name, expected_bins[i], bins[i], i);
        }
        failed_tests += test_passed(num_failing, test_name);
    }

    {
        std::vector<int> vec({1, 2, 3, 5, 15,
                              22, 23, 24, 25, 99,
                              33, 34, 35, 35, 99});
        dpct::device_vector<int> dvec(vec.begin(), vec.end());
        std::vector<int> levels({0, 4, 20, 55, 100});
        dpct::device_vector<int> dlevels(levels.begin(), levels.end());

        dpct::device_vector<int> bins(4, 0);

        int num_cols = 4;
        int num_rows = 3;
        int row_stride_bytes = 5 * sizeof(int);

        dpct::histogram_range_roi(oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()), dvec.begin(),
                             bins.begin(), 5, dlevels.begin(), num_cols, num_rows, row_stride_bytes);

        std::vector<int> expected_bins = {3, 1, 8, 0};

        std::string test_name = "histogram_range_roi";
        for (int i = 0; i < bins.size(); i++)
        {
            num_failing += ASSERT_ARRAY_EQUAL(test_name, expected_bins[i], bins[i], i);
        }
        failed_tests += test_passed(num_failing, test_name);
    }

    {
        std::vector<int> vec({1,  1,  0, 1,
                              2,  1,  0, 2,
                              3,  1,  0, 3,
                              5,  88, 0, 5,
                              15, 55, 0, 15,
                              22, 66, 0, 22,
                              23, 77, 0, 23,
                              24, 88, 0, 24,
                              25, 99, 0, 25,
                              99, 99, 0, 25});
        dpct::device_vector<int> dvec(vec.begin(), vec.end());
        dpct::device_vector<int> bins[3];

        bins[0].resize(10, 0);
        bins[1].resize(5, 0);
        bins[2].resize(9, 0);

        dpct::device_pointer<int> bin_pointers[3] = {bins[0].begin(), bins[1].begin(), bins[2].begin()};

        int sizes[3] = {11, 6, 10};
        int lower_levels[3] = {0, 0, 0};
        int upper_levels[3] = {100, 100, 100};

        dpct::multi_histogram_even<4, 3>(oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()),
                                       dvec.begin(), bin_pointers, sizes, lower_levels, upper_levels, dvec.size() / 4);

        std::vector<int> expected_bins[3] = {
            {4, 1, 4, 0, 0, 0, 0, 0, 0, 1}, {3, 0, 1, 2, 4}, {10, 0, 0, 0, 0, 0, 0, 0, 0}};

        std::string test_name = "multi_histogram_even";
        int index = 0;
        for (int b = 0; b < 3; b++)
        {
            for (int i = 0; i < sizes[b] - 1; i++)
            {
                num_failing += ASSERT_ARRAY_EQUAL(test_name, expected_bins[b][i], bins[b][i], index);
                index++;
            }
        }
        failed_tests += test_passed(num_failing, test_name);
    }

    {
        std::vector<int> vec({1,  1,  0, 1,
                              2,  1,  0, 2,
                              3,  1,  0, 3,
                              5,  88, 0, 5,
                              15, 55, 0, 15,
                              22, 66, 0, 22,
                              23, 77, 0, 23,
                              24, 88, 0, 24,
                              25, 99, 0, 25,
                              99, 99, 0, 25});
        dpct::device_vector<int> dvec(vec.begin(), vec.end());
        dpct::device_vector<int> bins[3];

        bins[0].resize(10, 0);
        bins[1].resize(10, 0);
        bins[2].resize(10, 0);

        dpct::device_pointer<int> bin_pointers[3] = {bins[0].begin(), bins[1].begin(), bins[2].begin()};

        std::vector<int> levels_0({0, 4, 20, 55, 100});
        std::vector<int> levels_1({0, 10, 20, 30, 40, 100});
        std::vector<int> levels_2({0, 12, 44, 100});

        dpct::device_vector<int> dlevels_0(levels_0.begin(), levels_0.end());
        dpct::device_vector<int> dlevels_1(levels_1.begin(), levels_1.end());
        dpct::device_vector<int> dlevels_2(levels_2.begin(), levels_2.end());

        int sizes[3];
        sizes[0] = levels_0.size();
        sizes[1] = levels_1.size();
        sizes[2] = levels_2.size();

        dpct::device_pointer<int> dlevel_pointers[3] = {dlevels_0.begin(), dlevels_1.begin(), dlevels_2.begin()};

        dpct::multi_histogram_range<4, 3>(oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()),
                                        dvec.begin(), bin_pointers, sizes, dlevel_pointers, dvec.size() / 4);

        std::vector<int> expected_bins[3] = {{3, 2, 4, 1}, {3, 0, 0, 0, 7}, {10, 0, 0}};

        std::string test_name = "multi_histogram_range";
        int index = 0;
        for (int b = 0; b < 3; b++)
        {
            for (int i = 0; i < sizes[b] - 1; i++)
            {
                num_failing += ASSERT_ARRAY_EQUAL(test_name, expected_bins[b][i], bins[b][i], index);
                index++;
            }
        }
        failed_tests += test_passed(num_failing, test_name);
    }

    {
        std::vector<int> vec({ /*row0*/
                              1, 1, 0, 1, 
                              2, 1, 0, 2, 
                              3, 1, 0, 3,     // skip
                              5, 88, 0, 5,    // skip
                               /*row1*/
                              15, 55, 0, 15,
                              22, 66, 0, 22,
                              23, 77, 0, 23,  // skip
                              24, 88, 0, 24,  // skip
                               /*row2*/
                              25, 99, 0, 25,  // skip
                              99, 99, 0, 25,  // skip
                              99, 99, 0, 25,  // skip
                              99, 99, 0, 25});// skip

        dpct::device_vector<int> dvec(vec.begin(), vec.end());

        dpct::device_vector<int> bins[3];

        bins[0].resize(10, 0);
        bins[1].resize(5, 0);
        bins[2].resize(9, 0);

        int num_cols = 2;
        int num_rows = 2;
        int row_stride_bytes = 4 * 4 * sizeof(int);

        dpct::device_pointer<int> bin_pointers[3] = {bins[0].begin(), bins[1].begin(), bins[2].begin()};

        int sizes[3] = {11, 6, 10};

        int lower_levels[3] = {0, 0, 0};
        int upper_levels[3] = {100, 100, 100};

        dpct::multi_histogram_even_roi<4, 3>(oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()),
                                       dvec.begin(), bin_pointers, sizes, lower_levels, upper_levels, num_cols,
                                       num_rows, row_stride_bytes);

        std::vector<int> expected_bins[3] = {
            {2, 1, 1, 0, 0, 0, 0, 0, 0, 0}, {2, 0, 1, 1, 0}, {4, 0, 0, 0, 0, 0, 0, 0, 0}};

        std::string test_name = "multi_histogram_even_roi";
        int index = 0;
        for (int b = 0; b < 3; b++)
        {
            for (int i = 0; i < sizes[b] - 1; i++)
            {
                num_failing += ASSERT_ARRAY_EQUAL(test_name, expected_bins[b][i], bins[b][i], index);
                index++;
            }
        }
        failed_tests += test_passed(num_failing, test_name);
    }

    {
        std::vector<int> vec({ /*row0*/
                              1, 1, 0, 1,
                              2, 1, 0, 2,
                              3, 1, 0, 3,     // skip
                              5, 88, 0, 5,    // skip
                               /*row1*/
                              15, 55, 0, 15,
                              22, 66, 0, 22,
                              23, 77, 0, 23,  // skip
                              24, 88, 0, 24,  // skip
                               /*row2*/
                              25, 99, 0, 25,  // skip
                              99, 99, 0, 25,  // skip
                              99, 99, 0, 25,  // skip
                              99, 99, 0, 25});// skip

        dpct::device_vector<int> dvec(vec.begin(), vec.end());

        dpct::device_vector<int> bins[3];

        bins[0].resize(10, 0);
        bins[1].resize(5, 0);
        bins[2].resize(9, 0);

        int num_cols = 2;
        int num_rows = 2;
        int row_stride_bytes = 4 * 4 * sizeof(int);

        dpct::device_pointer<int> bin_pointers[3] = {bins[0].begin(), bins[1].begin(), bins[2].begin()};

        std::vector<int> levels_0({0, 4, 20, 55, 100});
        std::vector<int> levels_1({0, 10, 20, 30, 40, 100});
        std::vector<int> levels_2({0, 12, 44, 100});

        dpct::device_vector<int> dlevels_0(levels_0.begin(), levels_0.end());
        dpct::device_vector<int> dlevels_1(levels_1.begin(), levels_1.end());
        dpct::device_vector<int> dlevels_2(levels_2.begin(), levels_2.end());

        int sizes[3];
        sizes[0] = levels_0.size();
        sizes[1] = levels_1.size();
        sizes[2] = levels_2.size();

        dpct::device_pointer<int> dlevel_pointers[3] = {dlevels_0.begin(), dlevels_1.begin(), dlevels_2.begin()};

        dpct::multi_histogram_range_roi<4, 3>(oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()),
                                        dvec.begin(), bin_pointers, sizes, dlevel_pointers, num_cols, num_rows,
                                        row_stride_bytes);

        std::vector<int> expected_bins[3] = {{2, 1, 1, 0}, {2, 0, 0, 0, 2}, {4, 0, 0}};

        std::string test_name = "multi_histogram_range_roi";
        int index = 0;
        for (int b = 0; b < 3; b++)
        {
            for (int i = 0; i < sizes[b] - 1; i++)
            {
                num_failing += ASSERT_ARRAY_EQUAL(test_name, expected_bins[b][i], bins[b][i], index);
                index++;
            }
        }
        failed_tests += test_passed(num_failing, test_name);
    }

    std::cout << std::endl << failed_tests << " failing test(s) detected." << std::endl;
    if (failed_tests == 0)
    {
        return 0;
    }
    return 1;
}