// ====------ onedpl_test_sort_keys.cpp---------- -*- C++ -* ----===////
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

int test_passed(int failing_elems, std::string test_name) {
  if (failing_elems == 0) {
    std::cout << "PASS: " << test_name << std::endl;
    return 0;
  }
  return 1;
}

template <typename KeyTp, bool descending> class VerifySortKeys {
private:
  using uint_type_t = typename dpct::internal::uint_map<KeyTp>::type;
  int64_t num_checks;
  dpct::internal::translate_key<KeyTp, uint_type_t> trans_key;

public:
  // by default, check all elements
  VerifySortKeys(int begin_bit = 0, int end_bit = sizeof(KeyTp) * 8,
                 int64_t num_checks_ = -1)
      : num_checks(num_checks_), trans_key(begin_bit, end_bit) {}

  bool operator()(::std::vector<KeyTp> &input_keys,
                  ::std::vector<KeyTp> &output_keys) const {
    bool ret = true;
    int n = output_keys.size();
    int64_t check_values = (num_checks == -1) ? n : num_checks;
    int64_t step = fmax(1, (n - 1) / check_values);

    uint_type_t prev_key = trans_key(output_keys[0]);
    uint_type_t cur_key;
    for (int i = 1; i < n; i += step) {
      cur_key = trans_key(output_keys[i]);
      if (descending) {
        bool loc_ret = (prev_key >= cur_key);
        if (!loc_ret) {
          printf("(output_keys[%d] = %f) > (prev_key = %f) ... list not sorted "
                 "in descending order\n",
                 i, (float)output_keys[i], (float)(prev_key));
        }
        ret &= loc_ret;
      } else {
        bool loc_ret = (prev_key <= cur_key);
        if (!loc_ret) {
          printf("(output_keys[%d] = %f) < (prev_key = %f) ... list not sorted "
                 "in ascending order\n",
                 i, (float)output_keys[i], (float)(prev_key));
        }
        ret &= loc_ret;
      }

      prev_key = cur_key;
    }
    return ret;
  }
};

template <typename KeyTp> class VerifySequencesMatch {
private:
  int64_t num_checks;
  int64_t num_elements;

public:
  // by default, check all elements
  VerifySequencesMatch(int64_t num_elements_ = -1, int64_t num_checks_ = -1)
      : num_checks(num_checks_), num_elements(num_elements_) {}

  bool operator()(::std::vector<KeyTp> &input_keys,
                  ::std::vector<KeyTp> &output_keys) const {
    bool ret = true;
    int n = (num_elements == -1) ? output_keys.size() : num_elements;
    int64_t check_values = (num_checks == -1) ? n : num_checks;
    int64_t step = fmax(1, (n - 1) / check_values);

    for (int i = 1; i < n; i += step) {
      bool loc_ret =
          (input_keys[i] == output_keys[i]) &&
          (::std::signbit(input_keys[i]) == ::std::signbit(output_keys[i]));
      if (!loc_ret) {
        printf("(input_keys[%d] = %f) != (output_keys[%d] = %f) ... sequences "
               "do not match\n",
               i, (float)input_keys[i], i, (float)(output_keys[i]));
      }

      ret &= loc_ret;
    }
    return ret;
  }
};

// Data setup

template <typename T> T GetRandVal() {
  return (T)(::std::rand() % (uint64_t)::std::numeric_limits<T>::max());
}

template <> float GetRandVal<float>() {
  return (float)(static_cast<float>(::std::rand()) /
                 (float)::std::numeric_limits<int>::max());
}

template <> double GetRandVal<double>() {
  return (double)(static_cast<double>(::std::rand()) /
                  (double)::std::numeric_limits<int>::max());
}

template <typename T> void GetRandVector(::std::vector<T> &data, int64_t size) {
  data.resize(size);
  for (int64_t i = 0; i < size; ++i) {
    data[i] = GetRandVal<T>();
  }
}

template <typename T> struct GetRandVectorFunc {
  void operator()(::std::vector<T> &vec, int64_t n) {
    GetRandVector<T>(vec, n);
  }
};

template <typename T> struct GetRandomSignedZerosFunc {
  void operator()(::std::vector<T> &vec, int64_t n) {
    vec.resize(n);
    for (int i = 0; i < n; i++) {
      vec[i] = ((::std::rand() % 2) == 0) ? T(-0.0f) : T(0.0f);
    }
  }
};

// test code
template <typename key_t>
inline void sort_keys_wrapper(sycl::queue queue, ::std::string testname,
                              const key_t *keys_in, key_t *keys_out, int64_t n,
                              bool descending, int begin_bit, int end_bit) {
  if (n >= std::numeric_limits<int>::max()) {
    printf("limit sort to INT_MAX elements");
    return;
  }

  if (keys_out == nullptr) {
    // This works as a "return" because the function is inlined
    keys_out = sycl::malloc_device<key_t>(n, queue);
  }

  if (begin_bit == 0 && end_bit == sizeof(key_t) * 8) {
    // test out default in api
    dpct::sort_keys(oneapi::dpl::execution::make_device_policy(queue), keys_in,
                    keys_out, n, descending);
  } else {
    dpct::sort_keys(oneapi::dpl::execution::make_device_policy(queue), keys_in,
                    keys_out, n, descending, begin_bit, end_bit);
  }
}

template <typename KeyTp, typename SetupDataOpT, typename VerifyOpT>
int setup_and_run(int64_t n, SetupDataOpT setup_data, VerifyOpT verify,
                  const bool descending, int begin_bit = 0,
                  int end_bit = sizeof(KeyTp) * 8) {
  ::std::ostringstream test_name_stream;
  test_name_stream << "sorting " << n << " elements of type "
                   << typeid(KeyTp).name() << " in "
                   << (descending ? "descending" : "ascending")
                   << " order, with bits: [" << begin_bit << ", " << end_bit
                   << ")" << ::std::endl;
  ::std::string test_name = std::move(test_name_stream).str();
  std::vector<KeyTp> input_keys;
  setup_data(input_keys, n);

  ::std::vector<KeyTp> output_keys(n, KeyTp(0));

  dpct::device_ext &device = dpct::get_current_device();
  sycl::queue queue = device.default_queue();

  KeyTp *dev_input_keys = sycl::malloc_device<KeyTp>(n, queue);

  KeyTp *dev_output_keys = sycl::malloc_device<KeyTp>(n, queue);

  queue.memcpy(dev_input_keys, input_keys.data(), n * sizeof(KeyTp)).wait();

  sort_keys_wrapper(queue, test_name, dev_input_keys, dev_output_keys, n,
                    descending, begin_bit, end_bit);

  queue.memcpy(output_keys.data(), dev_output_keys, n * sizeof(KeyTp)).wait();
  bool ret = verify(input_keys, output_keys);
  sycl::free(dev_input_keys, queue);
  sycl::free(dev_output_keys, queue);
  return ASSERT_EQUAL(true, ret, test_name.c_str());
}

template <typename KeyTp, typename SetupDataOpT, typename VerifyOp1T,
          typename VerifyOp2T>
bool setup_and_run_pingpong(int64_t n, SetupDataOpT setup_data,
                            VerifyOp1T verify1, VerifyOp2T verify2,
                            int begin_bit = 0,
                            int end_bit = sizeof(KeyTp) * 8) {
  ::std::ostringstream test_name_stream;
  test_name_stream << "sorting " << n << " elements of type "
                   << typeid(KeyTp).name() << " in "
                   << "descending followed by ascending order, with bits: ["
                   << begin_bit << ", " << end_bit << ")" << ::std::endl;
  ::std::string test_name = std::move(test_name_stream).str();
  std::vector<KeyTp> input_keys;
  setup_data(input_keys, n);

  ::std::vector<KeyTp> output_keys(n, KeyTp(0));

  dpct::device_ext &device = dpct::get_current_device();
  sycl::queue queue = device.default_queue();

  KeyTp *dev_input_keys = sycl::malloc_device<KeyTp>(n, queue);

  KeyTp *dev_output_keys = sycl::malloc_device<KeyTp>(n, queue);

  queue.memcpy(dev_input_keys, input_keys.data(), n * sizeof(KeyTp)).wait();

  dpct::io_iterator_pair<decltype(dev_input_keys)> pingpong(dev_input_keys,
                                                            dev_output_keys);

  if (begin_bit == 0 && end_bit == sizeof(KeyTp) * 8)
  {
    // testing defaults for bit range
    dpct::sort_keys(oneapi::dpl::execution::make_device_policy(queue), pingpong,
                  n, true, true);
  }
  else
  {
    dpct::sort_keys(oneapi::dpl::execution::make_device_policy(queue), pingpong,
                    n, true, true, begin_bit, end_bit);
  }


  queue.memcpy(output_keys.data(), pingpong.first(), n * sizeof(KeyTp)).wait();
  bool ret = verify1(input_keys, output_keys);

  dpct::sort_keys(oneapi::dpl::execution::make_device_policy(queue), pingpong,
                  n, false, true, begin_bit, end_bit);

  queue.memcpy(input_keys.data(), pingpong.first(), n * sizeof(KeyTp)).wait();
  ret &= verify2(output_keys, input_keys);

  sycl::free(dev_input_keys, queue);
  sycl::free(dev_output_keys, queue);
  return ASSERT_EQUAL(true, ret, test_name.c_str());
}

int main() {
  int test_suites_failed = 0;
  {
    ::std::string test_name = "sort integer keys";
    int tests_failed = 0;
    VerifySortKeys<int, true> verify_int_descending;
    GetRandVectorFunc<int> get_rand_int;
    tests_failed +=
        setup_and_run<int>(100, get_rand_int, verify_int_descending, true);

    VerifySortKeys<int, false> verify_int_ascending;
    tests_failed +=
        setup_and_run<int>(100, get_rand_int, verify_int_ascending, false);

    tests_failed += setup_and_run_pingpong<int>(
        100, get_rand_int, verify_int_descending, verify_int_ascending);

    test_suites_failed += test_passed(tests_failed, test_name);
  }

  {
    ::std::string test_name = "sort float keys";

    int tests_failed = 0;
    VerifySortKeys<float, true> verify_float_descending;
    GetRandVectorFunc<float> get_rand_float;
    tests_failed +=
        setup_and_run<float>(10, get_rand_float, verify_float_descending, true);
    tests_failed += setup_and_run<float>(100, get_rand_float,
                                         verify_float_descending, true);

    VerifySortKeys<float, false> verify_float_ascending;
    tests_failed +=
        setup_and_run<float>(10, get_rand_float, verify_float_ascending, false);
    tests_failed += setup_and_run<float>(100, get_rand_float,
                                         verify_float_ascending, false);

    tests_failed += setup_and_run_pingpong<float>(
        100, get_rand_float, verify_float_descending, verify_float_ascending);

    test_suites_failed += test_passed(tests_failed, test_name);
  }
  {
    ::std::string test_name = "sort float keys for zero stability";

    int tests_failed = 0;

    VerifySequencesMatch<float> verify_float_match;
    GetRandomSignedZerosFunc<float> get_signed_zeros_float;
    tests_failed += setup_and_run<float>(100, get_signed_zeros_float,
                                         verify_float_match, true);
    tests_failed += setup_and_run<float>(100, get_signed_zeros_float,
                                         verify_float_match, false);
    tests_failed += setup_and_run_pingpong<float>(
        100, get_signed_zeros_float, verify_float_match, verify_float_match);

    test_suites_failed += test_passed(tests_failed, test_name);
  }
  {
    ::std::string test_name = "sort double keys for zero stability";
    int tests_failed = 0;
    VerifySequencesMatch<double> verify_double_match;
    GetRandomSignedZerosFunc<double> get_signed_zeros_double;
    tests_failed += setup_and_run<double>(100, get_signed_zeros_double,
                                          verify_double_match, true);
    tests_failed += setup_and_run<double>(100, get_signed_zeros_double,
                                          verify_double_match, false);
    tests_failed += setup_and_run_pingpong<double>(
        100, get_signed_zeros_double, verify_double_match,
        verify_double_match);

    test_suites_failed += test_passed(tests_failed, test_name);
  }

  {
    ::std::string test_name = "sort int keys with bitrange";

    int tests_failed = 0;

    // partial bits sort
    int begin_bit = 8;
    int end_bit = 24;
    VerifySortKeys<int, true> verify_int_descending_subset(begin_bit, end_bit);
    GetRandVectorFunc<int> get_rand_int;
    tests_failed +=
        setup_and_run<int>(100, get_rand_int, verify_int_descending_subset,
                           true, begin_bit, end_bit);

    VerifySortKeys<int, false> verify_int_ascending_subset(begin_bit, end_bit);
    tests_failed +=
        setup_and_run<int>(100, get_rand_int, verify_int_ascending_subset,
                           false, begin_bit, end_bit);
    tests_failed += setup_and_run_pingpong<int>(
        100, get_rand_int, verify_int_descending_subset,
        verify_int_ascending_subset, begin_bit, end_bit);
    test_suites_failed += test_passed(tests_failed, test_name);
  }
  {
    ::std::string test_name = "sort double keys with bitrange";

    int tests_failed = 0;

    // partial bits sort
    int begin_bit = 10;
    int end_bit = 17;
    GetRandVectorFunc<double> get_rand_double;
    VerifySortKeys<double, true> verify_double_descending_subset(begin_bit,
                                                                 end_bit);
    tests_failed += setup_and_run<double>(100, get_rand_double,
                                          verify_double_descending_subset, true,
                                          begin_bit, end_bit);

    VerifySortKeys<double, false> verify_double_ascending_subset(begin_bit,
                                                                 end_bit);
    tests_failed += setup_and_run<double>(100, get_rand_double,
                                          verify_double_ascending_subset, false,
                                          begin_bit, end_bit);

    tests_failed += setup_and_run_pingpong<double>(
        100, get_rand_double, verify_double_descending_subset,
        verify_double_ascending_subset, begin_bit, end_bit);

    test_suites_failed += test_passed(tests_failed, test_name);
  }

  // test ping pong buffer
  {
    dpct::io_iterator_pair<int *> pp_default;

    oneapi::dpl::zip_iterator<int *, oneapi::dpl::counting_iterator<int>> zip;
    dpct::io_iterator_pair<decltype(zip)> pp_zip_default;

    auto input = pp_default.first();
    auto output = pp_zip_default.second();

    pp_zip_default.swap();

    auto swapped_input = pp_zip_default.first();

    ASSERT_EQUAL(true, swapped_input == output, "default ping pong buffer");

    std::vector<int> a_vec(100);
    std::vector<int> b_vec(100);

    dpct::io_iterator_pair<int *> pp_integer_ptr(a_vec.data(), b_vec.data());

    auto test_input1 = pp_integer_ptr.first();
    auto test_output1 = pp_integer_ptr.second();
    pp_integer_ptr.swap();
    auto test_input2 = pp_integer_ptr.first();
    auto test_output2 = pp_integer_ptr.second();
    ASSERT_EQUAL(true, test_input1 == test_output2, "input == swapped output");
    ASSERT_EQUAL(true, test_input2 == test_output1, "swapped input == output");
  }

  std::cout << std::endl
            << test_suites_failed << " failing test(s) detected." << std::endl;
  if (test_suites_failed == 0) {
    return 0;
  }
  return 1;
}