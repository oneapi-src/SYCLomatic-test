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

//#define LONG_TEST

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

#ifdef LONG_TEST

template <typename _T1, bool _IsFloatingPoint>
struct evenly_divided_binhash_impl{};

template <typename _T>
struct evenly_divided_binhash_impl<_T, /* is_floating_point = */ true>
{
    _T __minimum;
    _T __scale;
    evenly_divided_binhash_impl(const _T& min, const _T& max, const ::std::size_t& num_bins)
        : __minimum(min), __scale(_T(num_bins) / (max - min))
    {
    }
    template <typename _T2>
    ::std::uint32_t
    operator()(_T2&& value) const
    {
        return ::std::uint32_t((::std::forward<_T2>(value) - __minimum) * __scale);
    }

};

// non floating point type
template <typename _T>
struct evenly_divided_binhash_impl<_T, /* is_floating_point= */ false>
{
    _T __minimum;
    ::std::size_t __num_bins;
    _T __range_size;
    evenly_divided_binhash_impl(const _T& min, const _T& max, const ::std::size_t& num_bins)
        : __minimum(min), __num_bins(num_bins), __range_size(max - min)
    {
    }
    template <typename _T2>
    ::std::uint32_t
    operator()(_T2&& value) const
    {
        return ::std::uint32_t(((std::uint64_t(::std::forward<_T2>(value)) - __minimum) * std::uint64_t(__num_bins)) / __range_size);
    }
};


template <typename _T1>
using evenly_divided_binhash = evenly_divided_binhash_impl<_T1, std::is_floating_point_v<_T1>>;

//TODO: must ensure iterators here are device copy-able (protect with metaprogramming)
template <typename _ForwardIterator>
struct custom_range_binhash
{
    _ForwardIterator __first;
    _ForwardIterator __last;
    custom_range_binhash(_ForwardIterator first, _ForwardIterator last)
        : __first(first), __last(last)
    {
    }

    template <typename _T>
    ::std::uint32_t
    operator()(_T&& value) const
    {
        return (::std::upper_bound(__first, __last, ::std::forward<_T>(value)) - __first) - 1;

    }

};

template <typename _ForwardIterator, typename _RandomAccessIterator, typename _Size, typename _IdxHashFunc>
_RandomAccessIterator
histogram_general_sequential(_ForwardIterator __first, _ForwardIterator __last, _RandomAccessIterator __histogram_first,
                  _Size num_bins, _IdxHashFunc __func)
{
    for (auto tmp = __first; tmp != __last; ++tmp)
    {
        std::int64_t selected_bin = __func(*tmp);
        if ( selected_bin >=0 && selected_bin < num_bins)
    		++(__histogram_first[__func(*tmp)]);
    }
    return __histogram_first + num_bins;
}


template <typename _ForwardIterator, typename _RandomAccessIterator, typename _Size, typename _T>
_RandomAccessIterator
histogram_sequential(_ForwardIterator __first, _ForwardIterator __last, _RandomAccessIterator __histogram_first, _Size num_bins,
          _T __first_bin_min_val, _T __last_bin_max_val)
{
    return histogram_general_sequential(__first, __last, __histogram_first, num_bins,
                             evenly_divided_binhash<_T>(__first_bin_min_val, __last_bin_max_val, num_bins));
}

template <typename _ForwardIterator, typename _RandomAccessIterator1, typename _RandomAccessIterator2>
_RandomAccessIterator1
histogram_sequential(_ForwardIterator __first, _ForwardIterator __last, _RandomAccessIterator1 __histogram_first,
          _RandomAccessIterator2 __boundary_first, _RandomAccessIterator2 __boundary_last)
{
    return histogram_general_sequential(__first, __last, __histogram_first, (__boundary_last - __boundary_first) - 1,
                             custom_range_binhash{__boundary_first, __boundary_last});
}



enum DataGeneration{
    UniformDistribution = 0,
    NormalDistribution = 1,
    ConstantData = 2
};


template <typename ForwardIterator, typename ValType>
void
generate_data_helper(ForwardIterator first, ForwardIterator last, enum DataGeneration data_gen, ValType min, ValType max)
{
    std::default_random_engine gen{123};
    if (data_gen == DataGeneration::UniformDistribution)
    {
        if constexpr (std::is_integral_v<ValType>)
        {
            std::uniform_int_distribution<ValType> dist(min, max);
            std::generate(first, last, [&] { return dist(gen); });
        }
        else
        {
            std::uniform_real_distribution<ValType> dist(min, max);
            std::generate(first, last, [&]{ return dist(gen); });
        }
    }
    else if (data_gen == DataGeneration::NormalDistribution)
    {
        double center = ((double)max - (double)min) / 2.0 + (double)min;
        double std_dev = ((double)max - (double)min) / 8.0;

        std::normal_distribution<double> dist{center, std_dev};

        if constexpr (std::is_integral_v<ValType>)
        {
            std::generate(first, last, [&]{return ValType(::std::round(::std::max((double)min, ::std::min((double)max, dist(gen)))));});
        }
        else
        {
            std::generate(first, last, [&]{return ValType(::std::max((double)min, ::std::min((double)max-0.001, dist(gen))));});
        }
    }
    else if (data_gen == DataGeneration::ConstantData)
    {
        ValType center = (max - min) / 2 + min;
        std::fill(first, last, center);
    }
    else
    {
        std::cout << "ERROR, unsupported data generation!" << std::endl;
    }
}

#endif // LONG_TEST

int
main()
{

    // used to detect failures
    int failed_tests = 0;
    std::string test_name = "";

    {
        int num_failing = 0;

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

#ifdef LONG_TEST
    {
        int num_failing = 0;

        size_t max_size = 100000000;
        size_t max_bins = 16384;
        int min = 10;
        int max = 3141592;
        std::string test_name = "histogram_even_long_test_int";

        for (size_t n = 0; n <= max_size; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
        {
            for (size_t num_bins = 2; num_bins <= max_bins; num_bins = num_bins <= 16 ? num_bins + 1 : size_t(3.1415 * num_bins))
            {
                std::vector<int> vec(n);
                generate_data_helper(vec.begin(), vec.end(), DataGeneration::UniformDistribution, min, max-1);
                dpct::device_vector<int> dvec(vec.begin(), vec.end());
                dpct::device_vector<::std::uint64_t> bins(num_bins);
                std::vector<::std::uint64_t> expected_bins(num_bins, 0);
                dpct::histogram_even(oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()), dvec.begin(),
                                    bins.begin(), bins.size() + 1, min, max, dvec.size());
                histogram_sequential(vec.begin(), vec.end(), expected_bins.begin(), expected_bins.size(), min, max);
                for (int i = 0; i < bins.size(); i++)
                {
                    num_failing += ASSERT_ARRAY_EQUAL(test_name, expected_bins[i], bins[i], i);
                }
            }
        }
        failed_tests += test_passed(num_failing, test_name);

    }
    {
        int num_failing = 0;

        size_t max_size = 100000000;
        size_t max_bins = 16384;
        float min = 10.0f;
        float max = 3141592.0f;
        std::string test_name = "histogram_even_long_test_float";

        for (size_t n = 0; n <= max_size; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
        {
            for (size_t num_bins = 2; num_bins <= max_bins; num_bins = num_bins <= 16 ? num_bins + 1 : size_t(3.1415 * num_bins))
            {
                std::vector<float> vec(n);
                generate_data_helper(vec.begin(), vec.end(), DataGeneration::UniformDistribution, min, max);
                dpct::device_vector<float> dvec(vec.begin(), vec.end());
                dpct::device_vector<::std::uint64_t> bins(num_bins);
                std::vector<::std::uint64_t> expected_bins(num_bins, 0);
                dpct::histogram_even(oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()), dvec.begin(),
                                    bins.begin(), bins.size() + 1, min, max, dvec.size());
                histogram_sequential(vec.begin(), vec.end(), expected_bins.begin(), expected_bins.size(), min, max);
                for (int i = 0; i < bins.size(); i++)
                {
                    num_failing += ASSERT_ARRAY_EQUAL(test_name, expected_bins[i], bins[i], i);
                }
            }
        }
        failed_tests += test_passed(num_failing, test_name);

    }


#endif //LONG_TEST


    {
        int num_failing = 0;

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
#ifdef LONG_TEST
    {
        int num_failing = 0;

        size_t max_size = 100000000;
        size_t max_bins = 16384;
        int min = 10;
        int max = 3141592;
        std::string test_name = "histogram_range_long_test_int";

        for (size_t n = 0; n <= max_size; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
        {
            for (size_t num_bins = 2; num_bins <= max_bins; num_bins = num_bins <= 16 ? num_bins + 1 : size_t(3.1415 * num_bins))
            {
                std::vector<int> vec(n);
                generate_data_helper(vec.begin(), vec.end(), DataGeneration::UniformDistribution, min, max-1);

                dpct::device_vector<int> dvec(vec.begin(), vec.end());
                
                size_t scale = max - min;
                std::int64_t jitter;
                std::vector<int> levels(num_bins);
                levels[0] = min;
                for (size_t b = 1; b < num_bins -1; b++)
                {
                    jitter = ::std::rand() % (scale / 4) - scale/8;
                    levels[b] = scale + jitter;
                }
                levels[num_bins-1] = max;
                dpct::device_vector<int> dlevels(levels.begin(), levels.end());

                dpct::device_vector<::std::uint64_t> bins(num_bins);
                std::vector<::std::uint64_t> expected_bins(num_bins, 0);
                dpct::histogram_range(oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()), dvec.begin(),
                             bins.begin(), dlevels.size(), dlevels.begin(), dvec.size());

                histogram_sequential(vec.begin(), vec.end(), expected_bins.begin(), levels.begin(), levels.end());
                for (int i = 0; i < bins.size(); i++)
                {
                    num_failing += ASSERT_ARRAY_EQUAL(test_name, expected_bins[i], bins[i], i);
                }
            }
        }
        failed_tests += test_passed(num_failing, test_name);

    }

    {
        int num_failing = 0;

        size_t max_size = 100000000;
        size_t max_bins = 16384;
        float min = 10.0f;
        float max = 3141592.0f;
        std::string test_name = "histogram_range_long_test_float";

        for (size_t n = 0; n <= max_size; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
        {
            for (size_t num_bins = 2; num_bins <= max_bins; num_bins = num_bins <= 16 ? num_bins + 1 : size_t(3.1415 * num_bins))
            {
                std::vector<int> vec(n);
                generate_data_helper(vec.begin(), vec.end(), DataGeneration::UniformDistribution, min, max);

                dpct::device_vector<float> dvec(vec.begin(), vec.end());
                
                size_t scale = max - min;
                std::int64_t jitter;
                std::vector<float> levels(num_bins);
                levels[0] = min;
                for (size_t b = 1; b < num_bins -1; b++)
                {
                    jitter = ::std::rand() % (scale / 4) - scale/8;
                    levels[b] = scale + jitter;
                }
                levels[num_bins-1] = max;
                dpct::device_vector<float> dlevels(levels.begin(), levels.end());

                dpct::device_vector<::std::uint64_t> bins(num_bins);
                std::vector<::std::uint64_t> expected_bins(num_bins, 0);
                dpct::histogram_range(oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()), dvec.begin(),
                             bins.begin(), dlevels.size(), dlevels.begin(), dvec.size());

                histogram_sequential(vec.begin(), vec.end(), expected_bins.begin(), levels.begin(), levels.end());
                for (int i = 0; i < bins.size(); i++)
                {
                    num_failing += ASSERT_ARRAY_EQUAL(test_name, expected_bins[i], bins[i], i);
                }
            }
        }
        failed_tests += test_passed(num_failing, test_name);

    }
#endif //LONG_TEST


    {
        int num_failing = 0;

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
        int num_failing = 0;

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
        int num_failing = 0;

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
        int num_failing = 0;

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
        int num_failing = 0;

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
        int num_failing = 0;

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

    {
        int num_failing = 0;
        std::string test_name = "2D_ROI_functor";

        int cols = 59;
        int rows_stride = 147;

        dpct::internal::__roi_2d_index_functor custom_size_roi(cols, rows_stride);

        ASSERT_ARRAY_EQUAL(test_name, 0, custom_size_roi(0), 0);
        ASSERT_ARRAY_EQUAL(test_name, 147, custom_size_roi(59), 1);
        ASSERT_ARRAY_EQUAL(test_name, 21206, custom_size_roi(8534), 2);
        failed_tests += test_passed(num_failing, test_name);
    }

    {
        int num_failing = 0;
        std::string test_name = "interleaved_index_functor";
        {
            int total = 3;
            int active = 1;
            dpct::internal::__interleaved_index_functor intlv(total, active);
            for (int i = 0; i < 1000; i ++)
            {
                ASSERT_ARRAY_EQUAL(test_name, i*total+active, intlv(i), i);
            }
        }
        {
            int total = 10;
            int active = 9;
            dpct::internal::__interleaved_index_functor intlv(total, active);
            for (int i = 0; i < 1000; i ++)
            {
                ASSERT_ARRAY_EQUAL(test_name, i*total+active, intlv(i), i);
            }
        }
        {
            int total = 100;
            int active = 0;
            dpct::internal::__interleaved_index_functor intlv(total, active);
            for (int i = 0; i < 1000; i ++)
            {
                ASSERT_ARRAY_EQUAL(test_name, i*total+active, intlv(i), i);
            }
        }

        failed_tests += test_passed(num_failing, test_name);
    }

    {
        int num_failing = 0;
        std::string test_name = "composition_functor";
        {
            dpct::internal::__composition_functor triple_then_mod10([](auto a){return a*3;}, [](auto a){return a%10;});

            for (int i = 0; i < 1000; i ++)
            {
                ASSERT_ARRAY_EQUAL(test_name, (i*3)%10, triple_then_mod10(i), i);
            }
        }
        {
            dpct::internal::__composition_functor mod10_then_triple([](auto a){return a%10;}, [](auto a){return a*3;});

            for (int i = 0; i < 1000; i ++)
            {
                ASSERT_ARRAY_EQUAL(test_name, (i%10) * 3, mod10_then_triple(i), i);
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