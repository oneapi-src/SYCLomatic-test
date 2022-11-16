// ====------ onedpl_test_sort_pairs.cpp---------- -*- C++ -* ----===////
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

// Assumes that the values are a integer type initialized to match the index of
// the original position
//  This allows us to check that the key value pairs move together appropriately
template <typename KeyTp, typename ValueTp, bool descending>
class VerifyOrderedValues {
private:
  using uint_type_t = typename dpct::internal::uint_map<KeyTp>::type;
  int64_t num_checks;
  dpct::internal::translate_key<KeyTp, uint_type_t> trans_key;

public:
  // by default, check all elements
  VerifyOrderedValues(int begin_bit = 0, int end_bit = sizeof(KeyTp) * 8,
                      int64_t num_checks_ = -1)
      : num_checks(num_checks_), trans_key(begin_bit, end_bit) {}

  bool operator()(::std::vector<KeyTp> &input_keys,
                  ::std::vector<ValueTp> &input_values,
                  ::std::vector<KeyTp> &output_keys,
                  ::std::vector<ValueTp> &output_values) const {
    bool ret = true;
    int n = output_values.size();
    int64_t check_values = (num_checks == -1) ? n : num_checks;
    int64_t step = fmax(1, (n - 1) / check_values);

    uint_type_t prev_key = trans_key(output_keys[0]);
    ValueTp prev_value = output_values[0];
    for (int i = 1; i < n; i += step) {
      uint_type_t cur_key = trans_key(output_keys[i]);

      if (descending) {
        bool loc_ret = (prev_key >= cur_key);
        if (!loc_ret) {
          printf("(output_keys[%d] = %f) > (prev_key = %f) ... list not sorted "
                 "in descending order\n",
                 i, (float)cur_key, (float)(prev_key));
        }
        ret &= loc_ret;
      } else {
        bool loc_ret = (prev_key <= cur_key);
        if (!loc_ret) {
          printf("(output_keys[%d] = %f) < (prev_key = %f) ... list not sorted "
                 "in ascending order\n",
                 i, (float)cur_key, (float)(prev_key));
        }
        ret &= loc_ret;
      }

      {
        // This ensures that the key and value are moved together and relies on
        // the fact that input_values are set up
        // to equal their index.
        bool loc_ret = (input_keys[output_values[i]] == output_keys[i]);
        if (!loc_ret) {
          printf("(input_keys[%d] = %f) == (output_keys[%d] = %f) ... value "
                 "pair not copied with key\n",
                 output_values[i], (float)input_keys[output_values[i]], i,
                 (float)output_keys[i]);
        }
        ret &= loc_ret;
      }
      {
        // This is to check that the stability of the sort, relies on the fact
        // that input values are set up to
        // be equal to their index (ascending), therefore groups of equal keys
        // should have ascending output_value values as that means that their
        // relative positions are preserved.
        // * Requires using repeat values for keys, using a high n value with
        // limited key type like uint8_t will
        //   force this.
        if (cur_key == prev_key) {
          bool loc_ret = (prev_value < output_values[i]);
          if (!loc_ret) {
            printf("(output_values[%d] = %f) > (prev_value = %f) ... sort is "
                   "not stable\n",
                   i, (float)output_values[i], (float)prev_value);
          }
          ret &= loc_ret;
        }
      }
      prev_key = cur_key;
      prev_value = output_values[i];
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

template <typename T>
void GetCountingVector(::std::vector<T> &data, int64_t size) {
  data.resize(size);
  for (int64_t i = 0; i < size; ++i) {
    data[i] = i;
  }
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

// test code

template <typename key_t, typename value_t>
inline void sort_pairs(sycl::queue queue, const key_t *keys_in, key_t *keys_out,
                       const value_t *values_in, value_t *values_out, int64_t n,
                       bool descending = false, int begin_bit = 0,
                       int end_bit = sizeof(keys_in) * 8) {
  if (n >= std::numeric_limits<int>::max()) {
    printf("limit sort to INT_MAX elements");
    return;
  }

  if (keys_out == nullptr) {
    // This works as a "return" because the function is inlined
    keys_out = sycl::malloc_device<key_t>(n, queue);
  }

  if (begin_bit == 0 && end_bit == sizeof(keys_in) * 8) {
    // test out defaults in api
    dpct::sort_pairs(oneapi::dpl::execution::make_device_policy(queue), keys_in,
                     keys_out, values_in, values_out, n, descending);
  } else {
    dpct::sort_pairs(oneapi::dpl::execution::make_device_policy(queue), keys_in,
                     keys_out, values_in, values_out, n, descending, begin_bit,
                     end_bit);
  }
}

template <typename KeyTp, typename ValueTp, typename VerifyOpT>
bool setup_and_run(int64_t n, VerifyOpT verify, const bool descending,
                   int begin_bit = 0, int end_bit = sizeof(KeyTp) * 8) {
  ::std::ostringstream test_name_stream;
  test_name_stream << "sorting pairs " << n << " elements of type "
                   << typeid(KeyTp).name() << " in "
                   << (descending ? "descending" : "ascending")
                   << " order, with bits: [" << begin_bit << ", " << end_bit
                   << ")" << ::std::endl;
  ::std::string test_name = std::move(test_name_stream).str();

  ::std::vector<KeyTp> input_keys;
  GetRandVector<KeyTp>(input_keys, n);
  ::std::vector<ValueTp> input_values;
  GetCountingVector<ValueTp>(input_values, n);

  ::std::vector<KeyTp> output_keys(n, KeyTp(0));
  ::std::vector<ValueTp> output_values(n, ValueTp(0));

  dpct::device_ext &device = dpct::get_current_device();
  sycl::queue queue = device.default_queue();

  KeyTp *dev_input_keys = sycl::malloc_device<KeyTp>(n, queue);

  ValueTp *dev_input_values = sycl::malloc_device<ValueTp>(n, queue);

  KeyTp *dev_output_keys = sycl::malloc_device<KeyTp>(n, queue);

  ValueTp *dev_output_values = sycl::malloc_device<ValueTp>(n, queue);

  queue.memcpy(dev_input_keys, input_keys.data(), n * sizeof(KeyTp)).wait();
  queue.memcpy(dev_input_values, input_values.data(), n * sizeof(ValueTp))
      .wait();

  sort_pairs(queue, dev_input_keys, dev_output_keys, dev_input_values,
             dev_output_values, n, descending, begin_bit, end_bit);

  queue.memcpy(output_keys.data(), dev_output_keys, n * sizeof(KeyTp)).wait();
  queue.memcpy(output_values.data(), dev_output_values, n * sizeof(ValueTp))
      .wait();
  bool ret = verify(input_keys, input_values, output_keys, output_values);

  sycl::free(dev_input_keys, queue);
  sycl::free(dev_input_values, queue);
  sycl::free(dev_output_keys, queue);
  sycl::free(dev_output_values, queue);
  return ASSERT_EQUAL(true, ret, test_name.c_str());
}

int main() {

  int test_suites_failed = 0;
  {
    ::std::string test_name = "sort <int, int> keys";
    int tests_failed = 0;
    VerifyOrderedValues<int, int, true> verify_int_int_descending;
    tests_failed +=
        setup_and_run<int, int>(10, verify_int_int_descending, true);
    tests_failed +=
        setup_and_run<int, int>(1000, verify_int_int_descending, true);

    VerifyOrderedValues<int, int, false> verify_int_int_ascending;
    tests_failed +=
        setup_and_run<int, int>(10, verify_int_int_ascending, false);
    tests_failed +=
        setup_and_run<int, int>(1000, verify_int_int_ascending, false);
    test_suites_failed += test_passed(tests_failed, test_name);
  }
  {
    ::std::string test_name = "sort <float, int> pairs keys";
    int tests_failed = 0;
    VerifyOrderedValues<float, int, true> verify_float_int_descending;
    tests_failed +=
        setup_and_run<float, int>(10, verify_float_int_descending, true);
    tests_failed +=
        setup_and_run<float, int>(1000, verify_float_int_descending, true);

    VerifyOrderedValues<float, int, false> verify_float_int_ascending;
    tests_failed +=
        setup_and_run<float, int>(10, verify_float_int_ascending, false);
    tests_failed +=
        setup_and_run<float, int>(1000, verify_float_int_ascending, false);
    test_suites_failed += test_passed(tests_failed, test_name);
  }
  {
    ::std::string test_name = "sort <uint8_t, int> pairs keys";
    int tests_failed = 0;
    VerifyOrderedValues<uint8_t, int, true> verify_uint8t_int_descending;
    tests_failed +=
        setup_and_run<uint8_t, int>(10, verify_uint8t_int_descending, true);
    tests_failed +=
        setup_and_run<uint8_t, int>(1000, verify_uint8t_int_descending, true);

    VerifyOrderedValues<uint8_t, int, false> verify_uint8t_int_ascending;
    tests_failed +=
        setup_and_run<uint8_t, int>(10, verify_uint8t_int_ascending, false);
    tests_failed +=
        setup_and_run<uint8_t, int>(1000, verify_uint8t_int_ascending, false);
    test_suites_failed += test_passed(tests_failed, test_name);
  }
  {
    ::std::string test_name = "sort <int, int> pairs with bitrange";
    int tests_failed = 0;
    // partial bits sort
    int begin_bit = 8;
    int end_bit = 24;
    VerifyOrderedValues<int, int, true> verify_descending(begin_bit, end_bit);
    tests_failed += setup_and_run<int, int>(10, verify_descending, true,
                                            begin_bit, end_bit);
    tests_failed += setup_and_run<int, int>(1000, verify_descending, true,
                                            begin_bit, end_bit);

    VerifyOrderedValues<int, int, false> verify(begin_bit, end_bit);
    tests_failed +=
        setup_and_run<int, int>(10, verify, false, begin_bit, end_bit);
    tests_failed +=
        setup_and_run<int, int>(1000, verify, false, begin_bit, end_bit);
    test_suites_failed += test_passed(tests_failed, test_name);
  }
  {
    ::std::string test_name = "sort <double, int> pairs with bitrange";
    int tests_failed = 0;
    // partial bits sort
    int begin_bit = 10;
    int end_bit = 17;
    VerifyOrderedValues<double, int, true> verify_descending(begin_bit,
                                                             end_bit);
    tests_failed += setup_and_run<double, int>(10, verify_descending, true,
                                               begin_bit, end_bit);
    tests_failed += setup_and_run<double, int>(1000, verify_descending, true,
                                               begin_bit, end_bit);

    VerifyOrderedValues<double, int, false> verify(begin_bit, end_bit);
    tests_failed +=
        setup_and_run<double, int>(10, verify, false, begin_bit, end_bit);
    tests_failed +=
        setup_and_run<double, int>(1000, verify, false, begin_bit, end_bit);
    test_suites_failed += test_passed(tests_failed, test_name);
  }
}