// ====------ onedpl_test_arg_index_input_iterator.cpp---------- -*- C++ -*
// ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>

#include "dpct/dpct.hpp"
#include "dpct/dpl_utils.hpp"

#include <CL/sycl.hpp>

#include <chrono>
#include <iostream>

#define EPSILON 0.0001

template <typename String, typename _T1, typename _T2>
int ASSERT_EQUAL(String msg, _T1 &&X, _T2 &&Y) {
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

template <typename T> bool check_eq(T a, T b) {
  return fabs((double)a - (double)b) < EPSILON;
}

template <typename T>
bool check_ele_sum(T fill_val, ::std::vector<T> &output, int64_t idx_start,
                   int64_t idx, bool inclusive) {
  bool ret = check_eq(output[idx_start + idx],
                      (T)(idx + (inclusive ? 1 : 0)) * fill_val);
  return ret;
}

template <typename T>
bool check_ele_max(T fill_val, ::std::vector<T> &output, int64_t idx_start,
                   int64_t idx) {
  bool ret = check_eq(output[idx_start + idx], fill_val);
  return ret;
}

template <typename T> class VerifySumInclusive {
private:
  T fill_val;
  int64_t n;
  int64_t num_checks;

public:
  VerifySumInclusive(T fill_val_, int64_t num_checks_ = 100)
      : fill_val(fill_val_), num_checks(num_checks_) {}

  bool operator()(::std::vector<T> &output, int64_t idx_start = 0,
                  int64_t idx_end = -1) const {
    if (idx_end == -1) {
      idx_end = output.size();
    }
    bool ret = true;
    int64_t size = idx_end - idx_start;
    int64_t step = fmax(1, size / num_checks);

    for (int i = 0; i < size; i += step) {
      ret &= check_ele_sum(fill_val, output, idx_start, i, true);
    }

    if (size > 0) {
      ret &= check_ele_sum(fill_val, output, idx_start, size - 1, true);
    }
    return ret;
  }
};

class VerifyRangeEqual {
private:
  int64_t num_checks;

public:
  VerifyRangeEqual(int64_t num_checks_ = 100) : num_checks(num_checks_) {}

  template <typename T>
  bool operator()(T veca_begin, T veca_end, T vecb_begin) const {
    bool ret = true;
    int64_t size = veca_end - veca_begin;
    int64_t step = fmax(1, size / num_checks);
    T a_iter = veca_begin;
    T b_iter = vecb_begin;

    for (int i = 0; i < size; i += step, a_iter++, b_iter++) {
      ret &= check_ele_eq(*a_iter, *b_iter);
    }

    return ret;
  }
};

template <typename T> class VerifyMaxInclusive {
private:
  T fill_val;
  T second_val;
  int64_t num_checks;

public:
  VerifyMaxInclusive(T fill_val_, T second_val_, int64_t num_checks_ = 100)
      : fill_val(fill_val_), second_val(second_val_), num_checks(num_checks_) {}

  bool operator()(::std::vector<T> &output, int64_t idx_start = 0,
                  int64_t idx_end = -1) const {
    if (idx_end == -1) {
      idx_end = output.size();
    }

    bool ret = true;
    int64_t size = idx_end - idx_start;
    int64_t step = fmax(1, size / num_checks);

    ret &= check_ele_max(fill_val, output, idx_start, 0);

    for (int i = 1; i < size; i += step) {
      ret &= check_ele_max(second_val, output, idx_start, i);
    }

    return ret;
  }
};

template <typename OutputItrTp>
int check_output(OutputItrTp output, OutputItrTp ref, int64_t offset,
                 ::std::string msg) {
  return ASSERT_EQUAL(msg.c_str(), output[offset], ref[offset]);
}

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

int std_host_create_host_use(::std::vector<float> &input,
                             ::std::vector<float> &output, int offset) {

  auto itr = dpct::arg_index_input_iterator(input.data());
  typedef decltype(itr)::value_type Tuple;
  Tuple item_offset_pair = *itr;

  output[(int)item_offset_pair.key] = (float)item_offset_pair.value;
  itr = itr + offset;
  item_offset_pair = *itr;
  output[(int)item_offset_pair.key] = (float)item_offset_pair.value;

  return check_output(output.data(), input.data(), offset,
                      "host create host copied");
}

int usm_device_create_device_use(sycl::queue queue, ::std::vector<float> &input,
                                 ::std::vector<float> &output, int offset) {
  float *input_shared = sycl::malloc_shared<float>(input.size(), queue);
  float *output_shared = sycl::malloc_shared<float>(output.size(), queue);
  ::std::memcpy(input_shared, input.data(), input.size());
  ::std::memcpy(output_shared, output.data(), output.size());

  auto event = queue.submit([&](sycl::handler &h) {
    h.single_task([=]() {
      auto itr = dpct::arg_index_input_iterator(input_shared);
      typedef decltype(itr)::value_type Tuple;
      Tuple item_offset_pair = *itr;
      output_shared[(int)item_offset_pair.key] = (float)item_offset_pair.value;
      itr = itr + offset;
      item_offset_pair = *itr;
      output_shared[(int)item_offset_pair.key] = (float)item_offset_pair.value;
    });
  });

  return check_output(output_shared, input_shared, offset,
                      "host create host copied");
}

int ArgIndexInputIteratorMain(sycl::queue queue) {

  ::std::vector<float> input;
  const uint64_t n = 100;
  GetRandVector(input, n);

  ::std::vector<float> output(n, 0.0f);
  const int offset = 6;

  int failed_tests = 0;
  failed_tests += std_host_create_host_use(input, output, offset);
  failed_tests += usm_device_create_device_use(queue, input, output, offset);
  return test_passed(failed_tests, "Simple usage on host and device");
}

template <typename OrigIterTp, typename KeyValTp>
int AssertCheck(OrigIterTp orig, KeyValTp kv, int64_t expected_index) {
  ::std::string out =
      "(" + ::std::to_string(kv.key) + ", " + ::std::to_string(kv.value) + ") ";
  bool ret = true;
  if (kv.key == expected_index) {
    out += "[Matches expected index] ";
  } else {
    out += "[Does *NOT* match expected index, " +
           ::std::to_string(expected_index) + "]";
    ret = false;
  }
  if (kv.value == orig[kv.key]) {
    out += "[Key value pair grouped correctly] ";
  } else {
    out += "[Key value pair *NOT* grouped correctly] ";
    ret = false;
  }
  return ASSERT_EQUAL(out.c_str(), ret, true);
}

int CheckOperator(bool expression, ::std::string oper_string, bool expected) {
  return ASSERT_EQUAL(oper_string.c_str(), expression, expected);
}

template <typename InputIteratorTp>
bool TestArgIndexItr(InputIteratorTp itr_in, int64_t size) {
  int failed_tests = 0;
  if (size < 8) {
    ::std::cout << "Need at least 8 elements to test, returning" << ::std::endl;
    return 0;
  }
  auto itr = dpct::arg_index_input_iterator(itr_in);

  auto itr_begin = itr;
  failed_tests += AssertCheck(itr_in, *itr, 0);
  failed_tests += AssertCheck(itr_in, *(itr++), 0);
  failed_tests += AssertCheck(itr_in, *(itr++), 1);
  failed_tests += AssertCheck(itr_in, *(++itr), 3);
  failed_tests += AssertCheck(itr_in, *(itr += 4), 7);
  failed_tests += AssertCheck(itr_in, *(itr -= 4), 3);
  failed_tests += AssertCheck(itr_in, *(itr--), 3);
  failed_tests += AssertCheck(itr_in, *(--itr), 1);

  failed_tests += AssertCheck(itr_in, *(itr += 2), 3);
  auto normalized_itr = itr.create_normalized();
  failed_tests += CheckOperator((*normalized_itr).key == 0,
                                "create_normalized() key", true);
  failed_tests += CheckOperator((*normalized_itr).value == (*itr).value,
                                "create_normalized() value", true);

  failed_tests += CheckOperator(itr == itr_begin, "==", false);
  failed_tests += CheckOperator(itr - 3 == itr_begin, "==", true);
  failed_tests += CheckOperator(itr != itr_begin, "!=", true);
  failed_tests += CheckOperator(itr - 3 != itr_begin, "!=, -(int)", false);
  failed_tests += CheckOperator(itr - itr_begin == 3, "-(Itr)", true);
  failed_tests +=
      CheckOperator(itr[3].key == 6 && itr[4].value == itr_in[7], "[]", true);
  failed_tests += CheckOperator((itr + 2) == (itr_begin + 5), "+, ==", true);
  failed_tests +=
      CheckOperator((*(itr + 2)) == (itr_begin[5]), "+, [], ==", true);
  failed_tests += CheckOperator(itr_begin < itr, "<", true);
  failed_tests += CheckOperator(itr < itr_begin, "<", false);
  failed_tests += CheckOperator(itr > itr_begin, ">", true);
  failed_tests += CheckOperator(itr_begin > itr, ">", false);
  failed_tests += CheckOperator(itr_begin <= itr, "<=", true);
  failed_tests += CheckOperator(itr <= itr_begin, "<=", false);
  failed_tests += CheckOperator(itr >= itr_begin, ">=", true);
  failed_tests += CheckOperator(itr_begin >= itr, ">=", false);
  failed_tests += CheckOperator(itr_begin <= itr - 3, "<=", true);
  failed_tests +=
      CheckOperator(-3 + itr >= itr_begin, ">=, friend int +", true);

  auto iter_deref = *itr;
  auto kvp = iter_deref;
  failed_tests +=
      CheckOperator(iter_deref.key == kvp.key && iter_deref.value == kvp.value,
                    "(key_value_pair) assignment operator", true);
  kvp.key = kvp.key + 1;
  failed_tests +=
      CheckOperator(iter_deref.key != kvp.key,
                    "key is represented by value, not reference", true);

  kvp.value = kvp.value + 1;
  failed_tests +=
      CheckOperator(iter_deref.value != kvp.value,
                    "value is represented by value, not reference", true);

  failed_tests += CheckOperator(
      &(iter_deref.key) != &(kvp.key) && &(iter_deref.value) != &(kvp.value),
      "(key_value_pair) assignment operator properly assigns reference members",
      true);

  decltype(iter_deref) kvp2(iter_deref);
  failed_tests += CheckOperator(iter_deref.key == kvp2.key &&
                                    iter_deref.value == kvp2.value,
                                "(key_value_pair) copy constructor", true);

  failed_tests += CheckOperator(
      &(iter_deref.key) != &(kvp2.key) && &(iter_deref.value) != &(kvp2.value),
      "(key_value_pair) copy constructor properly assigns reference members",
      true);

  failed_tests +=
      CheckOperator(::std::is_same<decltype(iter_deref.key), ptrdiff_t>::value,
                    "(key_value_pair) key type is ptrdiff_t", true);

  failed_tests += CheckOperator(
      ::std::is_same<
          decltype(iter_deref.value),
          typename ::std::iterator_traits<InputIteratorTp>::value_type>::value,
      "(key_value_pair) value type is the value type of InputIteratorTp", true);

  failed_tests += CheckOperator(
      ::std::is_same<
          decltype(kvp.value),
          typename ::std::iterator_traits<InputIteratorTp>::value_type>::value,
      "(key_value_pair) value type is the value type of InputIteratorTp after "
      "assignment",
      true);

  return failed_tests;
}

int TestArgIndexItrMain() {
  ::std::vector<float> input;
  GetRandVector(input, 10);
  int failed_tests = 0;
  int ret = TestArgIndexItr(input.begin(), input.size());
  failed_tests +=
      test_passed(ret, "iterator to beginning of vector of 10 floats");
  ret = TestArgIndexItr(input.data(), input.size());
  failed_tests += test_passed(ret, "raw pointer to the first of 10 floats");
  ::std::vector<int> input_int;
  GetRandVector(input_int, 50);
  ret = TestArgIndexItr(input_int.begin(), input_int.size());
  failed_tests +=
      test_passed(ret, "iterator to beginning of vector of 50 ints");
  ret = TestArgIndexItr(input_int.data(), input_int.size());
  failed_tests += test_passed(ret, "raw pointer to the first of 50 ints");
  auto counting = oneapi::dpl::counting_iterator(10);
  ret = TestArgIndexItr(counting, 20);
  failed_tests += test_passed(ret, "iterator to counting_iterator of ints");
}

#define EPSILON 0.0001

template <typename InputIteratorT, typename OutputIteratorT, typename ScanOpT>
inline void inclusive_scan_(InputIteratorT input, OutputIteratorT output,
                            ScanOpT scan_op, int64_t num_items,
                            sycl::queue queue) {

  constexpr int max_size = ::std::numeric_limits<int>::max() / 2 + 1;
  int my_size = ::std::min<int64_t>(num_items, max_size);

  using input_t = ::std::remove_reference_t<decltype(*input)>;
  oneapi::dpl::inclusive_scan(oneapi::dpl::execution::make_device_policy(queue),
                              input, input + my_size, output, scan_op);
  queue.wait();

  input_t *first_elem_ptr = sycl::malloc_device<input_t>(1, queue);

  for (int64_t i = max_size; i < num_items; i += max_size) {

    my_size = ::std::min<int64_t>(num_items - i, max_size);

    auto event = queue.submit([&](sycl::handler &h) {
      h.single_task([=]() {
        *first_elem_ptr = scan_op(*(output + i - 1), *(input + i));
      });
    });
    queue.wait();

    using ArgIndexInputIterator =
        dpct::arg_index_input_iterator<InputIteratorT>;
    using tuple = typename ArgIndexInputIterator::value_type;

    auto input_iter_transform = [=](const tuple &x) -> input_t {
      if (x.key == 0) {
        return *first_elem_ptr;
      } else {
        return x.value;
      }
    };

    auto input_ = oneapi::dpl::make_transform_iterator(
        ArgIndexInputIterator(input + i), input_iter_transform);

    oneapi::dpl::inclusive_scan(
        oneapi::dpl::execution::make_device_policy(queue), input_,
        input_ + my_size, output + i, scan_op);
    queue.wait();
  }
  sycl::free(first_elem_ptr, queue);
}

template <typename T, typename ScanOpT, typename VerifyOpT>
int setup_and_run(T fill_val, T second_val, ScanOpT scanop, int64_t n,
                  VerifyOpT verify) {

  ::std::ostringstream test_name_stream;
  test_name_stream << "testing pytorch usage of " << n << " elements of type "
                   << typeid(T).name() << "." << std::endl;
  ::std::string test_name = std::move(test_name_stream).str();

  ::std::vector<T> input(n, fill_val);
  input[1] = second_val;
  ::std::vector<T> output(n, T(0));

  sycl::queue queue;

  T *dev_input = sycl::malloc_device<T>(n, queue);
  T *dev_output = sycl::malloc_device<T>(n, queue);

  queue.memcpy(dev_input, input.data(), n * sizeof(T)).wait();

  inclusive_scan_(dev_input, dev_output, scanop, n, queue);

  queue.memcpy(output.data(), dev_output, n * sizeof(T)).wait();
  bool ret = verify(output);
  sycl::free(dev_input, queue);
  sycl::free(dev_output, queue);
  return ASSERT_EQUAL(test_name.c_str(), ret, true);
}

int TestWithInclusiveScan() {
  float fill_val_f = 0.5f;
  int64_t num_floats = 10;
  int failed_tests = 0;
  VerifySumInclusive<float> verify_float(fill_val_f);
  failed_tests += setup_and_run(fill_val_f, fill_val_f, sycl::plus<void>(),
                                num_floats, verify_float);

  int64_t fill_val_int64 = 2;
  int64_t num_int64 = 50000;
  VerifySumInclusive<int64_t> verify_int64t(fill_val_int64);
  failed_tests += setup_and_run(fill_val_int64, fill_val_int64,
                                sycl::plus<void>(), num_int64, verify_int64t);

  double fill_val_double = 0.1;
  int64_t num_doubles = 100000;
  VerifySumInclusive<double> verify_double(fill_val_double);
  failed_tests += setup_and_run(fill_val_double, fill_val_double,
                                sycl::plus<void>(), num_doubles, verify_double);

  uint8_t fill_val_uint8t = 1;
  uint8_t end_val_uint8t = 2;
  int64_t num_uint8t = 100000;
  VerifyMaxInclusive<uint8_t> verify_uint_max(fill_val_uint8t, end_val_uint8t);

  failed_tests +=
      setup_and_run(fill_val_uint8t, end_val_uint8t, sycl::maximum<uint8_t>(),
                    num_uint8t, verify_uint_max);
  return test_passed(failed_tests, "Test with inclusive scan");
}

int main(int argc, char *argv[]) {

  sycl::queue queue;
  int failed_tests = 0;

  ::std::cout << "Target device: "
              << queue.get_info<::sycl::info::queue::device>()
                     .get_info<::sycl::info::device::name>()
              << ::std::endl;

  failed_tests += ArgIndexInputIteratorMain(queue);

  failed_tests += TestArgIndexItrMain();
  failed_tests += TestWithInclusiveScan();

  std::cout << std::endl
            << failed_tests << " failing test(s) detected." << std::endl;
  if (failed_tests == 0) {
    return 0;
  }
  return 1;
}
