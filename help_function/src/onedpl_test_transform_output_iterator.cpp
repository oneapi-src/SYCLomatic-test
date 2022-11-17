// ====------ onedpl_test_transform_output_iterator.cpp-- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include "oneapi/dpl/algorithm"
#include "oneapi/dpl/execution"
#include "oneapi/dpl/iterator"

#include "dpct/dpct.hpp"
#include "dpct/dpl_utils.hpp"

#include <CL/sycl.hpp>

#include <iomanip>
#include <iostream>

// verification utilities

template <typename String, typename _T1, typename _T2>
int ASSERT_EQUAL(_T1 &&X, _T2 &&Y, String msg) {
  if (X != Y) {
    std::cout << "FAIL: " << msg << " - (" << X << "," << Y << ")" << std::endl;
    return 1;
  }
  return 0;
}

template <typename String, typename _T1, typename _T2>
int ASSERT_EQUAL_N(_T1 &&X, _T2 &&Y, ::std::size_t n, String msg) {
  int failed_tests = 0;
  for (size_t i = 0; i < n; i++) {
    failed_tests += ASSERT_EQUAL(*X, *Y, msg);
    X++;
    Y++;
  }
  return failed_tests;
}

int test_passed(int failing_elems, std::string test_name) {
  if (failing_elems == 0) {
    std::cout << "PASS: " << test_name << std::endl;
    return 0;
  }
  return 1;
}

template <typename ExecutionPolicy, typename Iterator1, typename Iterator2,
          typename Size, typename Iterator3, typename Iterator4>
int test_copy(ExecutionPolicy &&exec, Iterator1 first1, Iterator1 last1,
              Iterator2 first2, Iterator3 result, Size n,
              Iterator4 expected_values) {

  ::std::copy(::std::forward<ExecutionPolicy>(exec), first1, last1, first2);
  // result is the base iterator for the transform_output_iterator
  // first2
  return ASSERT_EQUAL_N(
      expected_values, result, n,
      "Wrong result from copy with transform_output_iterator");
}

struct test_simple_copy {
  int operator()(size_t buffer_size) {

    sycl::queue q{};

    auto sycl_source_begin = sycl::malloc_shared<int>(buffer_size, q);
    auto sycl_result_begin = sycl::malloc_shared<int>(buffer_size, q);

    auto transformation = [](int item) { return item + 1; };

    int identity = 0;
    ::std::fill_n(sycl_source_begin, buffer_size, identity);
    auto tr_host_result_begin =
        dpct::make_transform_output_iterator(sycl_result_begin, transformation);

    ::std::vector<int> expected_res(buffer_size, identity + 1);

    return test_copy(oneapi::dpl::execution::make_device_policy(q),
                     sycl_source_begin, sycl_source_begin + buffer_size,
                     tr_host_result_begin, sycl_result_begin, buffer_size,
                     expected_res.begin());
  }
};

struct test_multi_transform_copy {
  int operator()(size_t buffer_size) {

    sycl::queue q{};

    auto sycl_source_begin = sycl::malloc_shared<int>(buffer_size, q);
    auto sycl_result_begin = sycl::malloc_shared<int>(buffer_size, q);

    auto transformation1 = [](int item) { return item + 1; };
    auto transformation2 = [](int item) { return item * 2; };
    auto transformation3 = [](int item) { return item + 3; };

    int identity = 0;
    ::std::fill_n(sycl_source_begin, buffer_size, identity);
    auto tr_sycl_result_begin = dpct::make_transform_output_iterator(
        sycl_result_begin, transformation1);
    auto tr2_sycl_result_begin = dpct::make_transform_output_iterator(
        tr_sycl_result_begin, transformation2);
    auto tr3_sycl_result_begin = dpct::make_transform_output_iterator(
        tr2_sycl_result_begin, transformation3);

    ::std::vector<int> expected_res(buffer_size, ((identity + 3) * 2) + 1);

    return test_copy(oneapi::dpl::execution::make_device_policy(q),
                     sycl_source_begin, sycl_source_begin + buffer_size,
                     tr3_sycl_result_begin, sycl_result_begin, buffer_size,
                     expected_res.begin());
  }
};

struct test_fill_transform {
  int operator()(size_t buffer_size) {

    sycl::queue q{};
    auto sycl_source_begin = sycl::malloc_shared<int>(buffer_size, q);
    auto sycl_result_begin = sycl::malloc_shared<int>(buffer_size, q);

    auto transformation = [](int item) { return item + 1; };

    int identity = 0;
    {
      auto tr_source_begin = dpct::make_transform_output_iterator(
          sycl_source_begin, transformation);
      ::std::fill_n(tr_source_begin, buffer_size, identity);
    }

    ::std::vector<int> expected_res(buffer_size, identity + 1);

    return ASSERT_EQUAL_N(
        expected_res.begin(), sycl_source_begin, buffer_size,
        "Wrong result from fill with transform_output_iterator");
  }
};

struct test_type_shift {
  int operator()(size_t buffer_size) {

    sycl::queue q{};

    auto sycl_source_begin = sycl::malloc_shared<double>(buffer_size, q);
    auto sycl_result_begin = sycl::malloc_shared<int>(buffer_size, q);

    // 3. run algorithms
    auto transformation1 = [](float item) { return (int)(item * 2.0f); };
    auto transformation2 = [](double item) { return (float)(item + 1.0); };

    double init = 0.5;

    ::std::fill_n(sycl_source_begin, buffer_size, init);

    auto tr1_host_result_begin = oneapi::dpl::make_transform_output_iterator(
        sycl_result_begin, transformation1);
    auto tr2_host_result_begin = oneapi::dpl::make_transform_output_iterator(
        tr1_host_result_begin, transformation2);

    ::std::vector<int> expected_res(buffer_size,
                                    (int)((float)(init + 1.0) * 2.0f));

    return test_copy(oneapi::dpl::execution::make_device_policy(q),
                     sycl_source_begin, sycl_source_begin + buffer_size,
                     tr2_host_result_begin, sycl_result_begin, buffer_size,
                     expected_res.begin());
  }
};

struct test_zip_iterator {
  int operator()(size_t buffer_size) {

    sycl::queue q{};

    auto sycl_source_begin = sycl::malloc_shared<float>(buffer_size, q);
    auto sycl_result_begin1 = sycl::malloc_shared<float>(buffer_size, q);
    auto sycl_result_begin2 = sycl::malloc_shared<float>(buffer_size, q);

    auto zip =
        oneapi::dpl::make_zip_iterator(sycl_result_begin1, sycl_result_begin2);

    auto transformation1 = [](const auto &item) {
      return ::std::make_tuple(item - 1.0f, item * item);
    };
    auto tr1_host_result_begin =
        dpct::make_transform_output_iterator(zip, transformation1);

    float init = 2.0f;

    ::std::fill_n(sycl_source_begin, buffer_size, init);

    ::std::vector<float> expected_res1(buffer_size, init - 1.0f);
    ::std::vector<float> expected_res2(buffer_size, init * init);
    auto zip_res = oneapi::dpl::make_zip_iterator(expected_res1.begin(),
                                                  expected_res2.begin());

    ::std::copy(oneapi::dpl::execution::make_device_policy(q),
                sycl_source_begin, sycl_source_begin + buffer_size,
                tr1_host_result_begin);

    int failed_tests = 0;
    failed_tests +=
        ASSERT_EQUAL_N(expected_res1.begin(), sycl_result_begin1, buffer_size,
                       "Wrong result from copy with transform_output_iterator");
    failed_tests +=
        ASSERT_EQUAL_N(expected_res2.begin(), sycl_result_begin2, buffer_size,
                       "Wrong result from copy with transform_output_iterator");

    return failed_tests;
  }
};

template <typename OutputIterator1, typename UnaryFunc>
auto attempt_to_dangle(OutputIterator1 out1, UnaryFunc unary) {
  auto toi = oneapi::dpl::make_transform_output_iterator(out1, unary);
  // Leave scope of the transform output iterator, then assign data
  // to the wrapper after the fact.
  return *toi;
}

template <typename ExecutionPolicy, typename OutputIterator1,
          typename InputIterator, typename Size, typename Iterator3,
          typename UnaryFunc>
int dangle_test(ExecutionPolicy &&exec, OutputIterator1 first1,
                OutputIterator1 last1, InputIterator input, Size n,
                Iterator3 expected_values, UnaryFunc unary) {
  auto num_eles = last1 - first1;
  oneapi::dpl::counting_iterator<::std::size_t> count_first(0UL);
  oneapi::dpl::counting_iterator<::std::size_t> count_last(num_eles);
  std::for_each(::std::forward<ExecutionPolicy>(exec), count_first, count_last,
                [=](const auto &elem) {
                  auto wrapper = attempt_to_dangle(first1 + elem, unary);
                  // assigning to wrapper object after
                  // transform_iterator leaves
                  // scope with zip iter
                  wrapper = input[elem];
                });

  return ASSERT_EQUAL_N(
      expected_values, first1, n,
      "Wrong result from usage of transform_output_iterator (dangle_test) ");
}

struct test_dangling_ref {
  int operator()(size_t buffer_size) {

    sycl::queue q{};

    auto sycl_source_begin = sycl::malloc_shared<float>(buffer_size, q);
    auto sycl_result_begin = sycl::malloc_shared<float>(buffer_size, q);

    float init = 2.0f;

    ::std::fill_n(sycl_source_begin, buffer_size, init);

    ::std::vector<float> expected_res(buffer_size, init * init);

    auto transformation1 = [](const auto &item) { return item * item; };

    return dangle_test(oneapi::dpl::execution::make_device_policy(q),
                       sycl_result_begin, sycl_result_begin + buffer_size,
                       sycl_source_begin, buffer_size, expected_res.begin(),
                       transformation1);
  }
};

template <typename OutputIterator1, typename UnaryFunc1, typename UnaryFunc2>
auto attempt_to_dangle_chain(OutputIterator1 out1, UnaryFunc1 unary1,
                             UnaryFunc2 unary2) {
  auto toi1 = oneapi::dpl::make_transform_output_iterator(out1, unary1);
  auto toi2 = oneapi::dpl::make_transform_output_iterator(toi1, unary2);
  // leave scope of the transform output iterator, then assign
  // data to the wrapper after the fact.
  return *toi2;
}

template <typename ExecutionPolicy, typename OutputIterator1,
          typename InputIterator, typename Size, typename Iterator3,
          typename UnaryFunc1, typename UnaryFunc2>
int chain_dangle_test(ExecutionPolicy &&exec, OutputIterator1 first1,
                      OutputIterator1 last1, InputIterator input, Size n,
                      Iterator3 expected_values, UnaryFunc1 unary1,
                      UnaryFunc2 unary2) {
  auto num_eles = last1 - first1;
  oneapi::dpl::counting_iterator<::std::size_t> count_first(0UL);
  oneapi::dpl::counting_iterator<::std::size_t> count_last(num_eles);
  std::for_each(::std::forward<ExecutionPolicy>(exec), count_first, count_last,
                [=](const auto &elem) {
                  auto wrapper =
                      attempt_to_dangle_chain(first1 + elem, unary1, unary2);
                  // assigning to wrapper object after transform_iterators
                  // leaves scope
                  wrapper = input[elem];
                });

  return ASSERT_EQUAL_N(expected_values, first1, n,
                        "Wrong result from usage of transform_output_iterator "
                        "(chain_dangle_test) ");
}

struct test_dangling_chain_ref {
  int operator()(size_t buffer_size) {

    sycl::queue q{};

    auto sycl_source_begin = sycl::malloc_shared<float>(buffer_size, q);
    auto sycl_result_begin = sycl::malloc_shared<float>(buffer_size, q);

    float init = 2.0f;

    ::std::fill_n(sycl_source_begin, buffer_size, init);

    ::std::vector<float> expected_res(buffer_size, (init * init) - 2.0f);

    auto transformation1 = [](const auto &item) { return item - 2.0f; };
    auto transformation2 = [](const auto &item) { return item * item; };

    return chain_dangle_test(
        oneapi::dpl::execution::make_device_policy(q), sycl_result_begin,
        sycl_result_begin + buffer_size, sycl_source_begin, buffer_size,
        expected_res.begin(), transformation1, transformation2);
  }
};

template <typename OutputIterator1, typename OutputIterator2,
          typename UnaryFunc>
auto attempt_to_dangle_zip(OutputIterator1 out1, OutputIterator2 out2,
                           UnaryFunc unary) {
  auto zip = oneapi::dpl::make_zip_iterator(out1, out2);
  auto toi = oneapi::dpl::make_transform_output_iterator(zip, unary);
  // leave scope of the transform output iterator, then assign data to the
  // wrapper after the fact.
  return *toi;
}

template <typename ExecutionPolicy, typename OutputIterator1,
          typename OutputIterator2, typename InputIterator, typename Size,
          typename Iterator3, typename Iterator4, typename UnaryFunc>
int zip_dangle_test(ExecutionPolicy &&exec, OutputIterator1 first1,
                    OutputIterator1 last1, OutputIterator2 first2,
                    InputIterator input, Size n, Iterator3 expected_values1,
                    Iterator4 expected_values2, UnaryFunc unary) {
  auto num_eles = last1 - first1;
  oneapi::dpl::counting_iterator<::std::size_t> count_first(0UL);
  oneapi::dpl::counting_iterator<::std::size_t> count_last(num_eles);
  std::for_each(::std::forward<ExecutionPolicy>(exec), count_first, count_last,
                [=](const auto &elem) {
                  auto wrapper = attempt_to_dangle_zip(first1 + elem,
                                                       first2 + elem, unary);
                  // assigning to wrapper object after transform_iterator leaves
                  // scope
                  wrapper = input[elem];
                });
  auto zip_output = oneapi::dpl::make_zip_iterator(first1, first2);

  int falied_tests = 0;
  falied_tests +=
      ASSERT_EQUAL_N(expected_values1, first1, n,
                     "Wrong result from usage of transform_output_iterator "
                     "(zip_dangle_test) ");
  falied_tests +=
      ASSERT_EQUAL_N(expected_values2, first2, n,
                     "Wrong result from usage of transform_output_iterator "
                     "(zip_dangle_test) ");
  return falied_tests;
}

struct test_dangling_zip_ref {
  int operator()(size_t buffer_size) {

    sycl::queue q{};

    auto sycl_source_begin = sycl::malloc_shared<float>(buffer_size, q);
    auto sycl_result_begin1 = sycl::malloc_shared<float>(buffer_size, q);
    auto sycl_result_begin2 = sycl::malloc_shared<float>(buffer_size, q);

    float init = 2.0f;

    ::std::fill_n(sycl_source_begin, buffer_size, init);

    ::std::vector<float> expected_res1(buffer_size, init - 1.0f);
    ::std::vector<float> expected_res2(buffer_size, init * init);
    auto zip_res = oneapi::dpl::make_zip_iterator(expected_res1.begin(),
                                                  expected_res2.begin());

    auto transformation1 = [](const auto &item) {
      return ::std::make_tuple(item - 1.0f, item * item);
    };

    return zip_dangle_test(oneapi::dpl::execution::make_device_policy(q),
                           sycl_result_begin1, sycl_result_begin1 + buffer_size,
                           sycl_result_begin2, sycl_source_begin, buffer_size,
                           expected_res1.begin(), expected_res2.begin(),
                           transformation1);
  }
};

template <typename TestFunctor>
int RunTest(TestFunctor test, size_t max_n, ::std::string msg) {
  int failed_tests = 0;
  for (size_t n = 1; n <= max_n; n = n <= 16 ? n + 1 : size_t(3.1415 * n)) {
    failed_tests += test(n);
  }
  return test_passed(failed_tests, msg);
}

int main() {

  // used to detect failures

  int test_suites_failed = 0;
  size_t max_n = 10000;

  test_suites_failed += RunTest(test_simple_copy(), max_n, "Simple Copy");
  test_suites_failed +=
      RunTest(test_multi_transform_copy(), max_n, "Multiple Transforms");
  test_suites_failed +=
      RunTest(test_fill_transform(), max_n, "Simple Host Fill");
  test_suites_failed +=
      RunTest(test_type_shift(), max_n, "Type Shifting Transform");
  test_suites_failed +=
      RunTest(test_zip_iterator(), max_n, "Zip Iterator Transform");
  test_suites_failed +=
      RunTest(test_dangling_ref(), max_n, "Checking Reference Not Dangled");
  test_suites_failed += RunTest(test_dangling_chain_ref(), max_n,
                                "Checking Reference Not Dangled Chain");
  test_suites_failed += RunTest(test_dangling_zip_ref(), max_n,
                                "Checking Reference Not Dangled Zip");

  std::cout << std::endl
            << test_suites_failed << " failing test(s) detected." << std::endl;
  if (test_suites_failed == 0) {
    return 0;
  }
  return 1;
}
