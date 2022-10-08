// ====------ onedpl_test_constant_iterator.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/numeric>

#include "dpct/dpct.hpp"
#include "dpct/dpl_utils.hpp"

#include <CL/sycl.hpp>

#include <iostream>
#include <chrono>

template<typename String, typename _T1, typename _T2>
int ASSERT_EQUAL(String msg, _T1&& X, _T2&& Y) {
    if(X!=Y) {
        std::cout << "FAIL: " << msg << " - (" << X << "," << Y << ")" << std::endl;
        return 1;
    }
    return 0;
}

// Collect statistics of the times sampled, including the 95% confidence interval using the t-distribution
struct statistics {
    statistics(std::vector<size_t> const& samples)
        : min(std::numeric_limits<size_t>::max()), max(0), mean(0.), stddev(0.), confint(0.)
    {
        if (samples.size() > 10)
            std::cout << "Warning: statistics reported using first 10 samples\n";

        for (int i = 0; i != 10; ++i) {
            if (samples[i] < min)
                min = samples[i];
            if (samples[i] > max)
                max = samples[i];
            mean += samples[i];
        }

        mean /= samples.size();

        for (int i = 0; i != 10; ++i) {
            stddev += (samples[i] - mean) * (samples[i] - mean);
        }
        stddev /= samples.size() - 1;
        stddev = std::sqrt(stddev);

        // value for 95% confidence interval with 10 samples (9 degrees of freedom) is 2.262
        confint = 2.262 * stddev/std::sqrt(10.0);
    }

    size_t min;
    size_t max;
    float mean;
    float stddev;
    float confint;
};

template<typename _RefIt, typename _ConstIt>
int evaluate(_RefIt ref_begin, _RefIt ref_end, _ConstIt const_begin, _ConstIt const_end, std::string test) {
    int return_value;
    using clock = std::chrono::high_resolution_clock;
    using value_type = typename std::iterator_traits<_RefIt>::value_type;

    value_type zero = 0;

    // Collect 10 samples and compute statistics using t-distribution
    std::vector<size_t> ref_times;
    std::vector<size_t> const_times;

    ref_times.reserve(10);
    const_times.reserve(10);

    const size_t n = std::distance(ref_begin, ref_end);
    for (int i = 0; i != 10; ++i) {
        auto start = clock::now();
        auto ref_sum = oneapi::dpl::reduce(oneapi::dpl::execution::make_device_policy<class Reduce1>(oneapi::dpl::execution::dpcpp_default), ref_begin, ref_end, zero);
        auto stop = clock::now();
        ref_times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count());

        start = clock::now();
        auto const_sum = oneapi::dpl::reduce(oneapi::dpl::execution::make_device_policy<class Reduce2>(oneapi::dpl::execution::dpcpp_default), const_begin, const_end, zero);
        stop = clock::now();
        const_times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count());

        return_value = ASSERT_EQUAL("Statistics test", ref_sum, const_sum);
    }
    statistics ref_stats(ref_times);
    statistics const_stats(const_times);
#if PRINT_STATISTICS
    std::cout << test << " (ns): " << const_stats.mean << " conf. int. " << const_stats.confint
              << ", min " << const_stats.min << ", max " << const_stats.max << "\n"
              << "reference (ns): " << ref_stats.mean << " conf. int. " << ref_stats.confint
              << ", min " << ref_stats.min << ", max " << ref_stats.max << "\n"
              << "slowdown (const / ref): " << const_stats.mean / ref_stats.mean << std::endl;
#endif

    return return_value;
}

struct constant_it_test {
    int var1;
    int var2;
    char var3;

    constant_it_test(int v1, int v2, char v3) : var1(v1), var2(v2), var3(v3) {}
};

struct combine_components {
    template <typename Scalar>
    Scalar operator()(const constant_it_test& t, const Scalar& s) const {
        return ((t.var1 * t.var2) / (t.var3 - 'a') * s);
    }
};

int main(int argc, char** argv)
{
    // used to detect failures
    int failed_tests = 0;
    int num_failing = 0;

    int n = 10000;
    if (argc == 2)
        n = std::atoi(argv[1]);

    uint64_t value = 7;

    // create buffers
    sycl::buffer<uint64_t, 1> ref_buf{ sycl::range<1>(n) };

    auto ref_begin = oneapi::dpl::begin(ref_buf);
    auto ref_end = oneapi::dpl::end(ref_buf);
    std::fill(oneapi::dpl::execution::dpcpp_default, ref_begin, ref_end, value);

    auto const_begin = dpct::make_constant_iterator<uint64_t>(value);
    dpct::constant_iterator<uint64_t> const_end = const_begin + n;

    failed_tests += evaluate(ref_begin, ref_end, const_begin, const_end, std::string("GPU_const_iterator"));

    // Adding test for make_constant_iterator

    // create buffer
    sycl::buffer<uint64_t, 1> src_buf { sycl::range<1>(8) };
    auto src_it = oneapi::dpl::begin(src_buf);

    constant_it_test c1(3, 4, 'd');

    {
        auto src = src_it.get_buffer().template get_access<sycl::access::mode::write>();
        for (int i = 0; i != 8; ++i) {
            src[i] = 10;
        }
    }

    // call algorithm
    std::transform(oneapi::dpl::execution::dpcpp_default, dpct::make_constant_iterator(c1), dpct::make_constant_iterator(c1) + 5, dpct::make_counting_iterator(0), src_it, combine_components());

    {
        std::string test_name = "dpct::make_constant_iterator with std::transform";
        auto src = src_it.get_buffer().template get_access<sycl::access::mode::read>();
        for (int i = 0; i != 8; ++i) {
            if (i < 5)
                num_failing += ASSERT_EQUAL(test_name, src[i], i*4);
            else
                num_failing += ASSERT_EQUAL(test_name, src[i], 10);
        }

        if (num_failing == 0) {
            std::cout << "PASS: " << test_name << std::endl;
            num_failing = 0;
        }
        else {
            ++failed_tests;
        }
    }

    std::cout << std::endl << failed_tests << " failing test(s) detected." << std::endl;
    if (failed_tests == 0) {
        return 0;
    }
    return 1;
}
