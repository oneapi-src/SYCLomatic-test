// ====------ onedpl_test_segmented_sort_pairs.cpp---------- -*- C++ -*
// ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include "oneapi/dpl/iterator"
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>

#include "dpct/dpct.hpp"
#include "dpct/dpl_utils.hpp"

#include <CL/sycl.hpp>

#include <iomanip>
#include <iostream>

template <typename _T1, typename _T2, typename String>
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

// Verification utilities

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

template <typename scalar_t>
inline void
segmented_sort_keys(sycl::queue queue, int64_t nsegments, int64_t nsort,
                     int64_t n, bool descending, scalar_t *self_ptr,
                     scalar_t *values_ptr, int algorithm,
                     bool use_io_iterator_pair = false, int begin_bit = 0,
                     int end_bit = sizeof(self_ptr) * 8) {
  if (use_io_iterator_pair && algorithm != 3)
  {
    //only public API can use io_iterator_pair
    return;
  }
  const auto numel = nsort * nsegments;

  auto offset_generator = [](auto stride, auto offset) {
    return oneapi::dpl::make_transform_iterator(
        oneapi::dpl::counting_iterator(offset),
        [stride](auto a) { return a * stride; });
  };
  auto offset_generator_starts = offset_generator((int)nsort, 0);
  auto offset_generator_ends = offset_generator((int)nsort, 1);

  if (algorithm == 0) {
    dpct::internal::segmented_sort_keys_by_parallel_for_of_sorts(
        oneapi::dpl::execution::make_device_policy(queue), self_ptr, values_ptr,
        n, nsegments, offset_generator_starts, offset_generator_ends,
        descending, begin_bit, end_bit);
  } else if (algorithm == 1) {
    dpct::internal::segmented_sort_keys_by_parallel_sorts(
        oneapi::dpl::execution::make_device_policy(queue), self_ptr, values_ptr,
        n, nsegments, offset_generator_starts, offset_generator_ends,
        descending, begin_bit, end_bit);
  } else if (algorithm == 2) {
    dpct::internal::segmented_sort_keys_by_two_pair_sorts(
        oneapi::dpl::execution::make_device_policy(queue), self_ptr, values_ptr,
        n, nsegments, offset_generator_starts, offset_generator_ends,
        descending, begin_bit, end_bit);
  } else if (algorithm == 3) // this will be the one used for the mapping,
                             // others are for timing purposes
  {
    if (use_io_iterator_pair)
    {
      dpct::io_iterator_pair<scalar_t*> keys(self_ptr, values_ptr);

      dpct::segmented_sort_keys(
          oneapi::dpl::execution::make_device_policy(queue), keys, n, 
          nsegments, offset_generator_starts, offset_generator_ends,
          descending, begin_bit, end_bit);

    }
    else
    {
      dpct::segmented_sort_keys(
          oneapi::dpl::execution::make_device_policy(queue), self_ptr,
          values_ptr, n, nsegments, offset_generator_starts,
          offset_generator_ends, descending, begin_bit, end_bit);
    }
  }
}

// Algorithm:
//   0: parallel_for of serial sorts
//   1: for loop of parallel sorts
//   2: segmented sort with 2 sorts
//   3: let the device characteristics decide
template <typename scalar_t>
int test_with_generated_offsets(const int64_t nsegments, const int64_t nsort,
                                const int64_t n, const bool descending,
                                int algorithm, 
                                bool use_io_iterator_pair = false) {
  ::std::ostringstream test_name_stream;
  test_name_stream << "testing sorting with " << nsegments
                   << " segments of size " << nsort << " of type "
                   << typeid(scalar_t).name() << " in "
                   << (descending ? "descending" : "ascending") << " order"
                   << std::endl;

  if (use_io_iterator_pair && algorithm != 3)
  {
    std::cout<<"io_iterator_pair interface only available with public dpct API"<<std::endl;
    return 1;
  }

  ::std::string test_name = std::move(test_name_stream).str();
  // setup data
  int64_t numel = nsort * nsegments;
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue queue = dev_ct1.default_queue();

  std::vector<scalar_t> input_keys;
  GetRandVectorFunc<scalar_t> setup_data;
  setup_data(input_keys, n);

  std::vector<scalar_t> output_keys(n, scalar_t(0));

  scalar_t *dev_input_keys = sycl::malloc_device<scalar_t>(numel, queue);

  scalar_t *dev_output_keys = sycl::malloc_device<scalar_t>(numel, queue);

  queue.memcpy(dev_input_keys, input_keys.data(), n * sizeof(scalar_t)).wait();

  segmented_sort_keys(queue, nsegments, nsort, n, descending, dev_input_keys,
                       dev_output_keys, algorithm, use_io_iterator_pair);

  queue.memcpy(output_keys.data(), dev_output_keys, n * sizeof(scalar_t))
      .wait();

  // sort each segment individually

  std::vector<scalar_t> output_keys_indiv(n, scalar_t(0));

  scalar_t *dev_output_keys_indiv = sycl::malloc_device<scalar_t>(n, queue);

  queue
      .memcpy(dev_output_keys_indiv, output_keys_indiv.data(),
              n * sizeof(scalar_t))
      .wait();

  for (int seg = 0; seg < nsegments; seg++) {
    int64_t sort_n = std::min(nsort, n - seg * nsort);
    dpct::sort_keys(oneapi::dpl::execution::make_device_policy(queue),
                    dev_input_keys + (seg * nsort),
                    dev_output_keys_indiv + (seg * nsort), sort_n, descending);
  }
  queue
      .memcpy(output_keys_indiv.data(), dev_output_keys_indiv,
              n * sizeof(scalar_t))
      .wait();

  // compare lists to verify
  VerifySequencesMatch<scalar_t> verify(n);
  bool ret = verify(output_keys, output_keys_indiv);
  if (!ret) {
    std::cout << "Individually sorted segments dont match segmented_sort"
              << std::endl;
  }

  sycl::free(dev_input_keys, queue);
  sycl::free(dev_output_keys, queue);
  sycl::free(dev_output_keys_indiv, queue);
  return ASSERT_EQUAL(true, ret, test_name.c_str());
}

//   0: parallel_for of serial sorts
//   1: for loop of parallel sorts
//   2: full pair sort index mapping
//   3: Selected alg based on device characteristics and nsegments
::std::string GetAlgorithmName(int alg, bool descending) {
  ::std::ostringstream test_name_stream;
  test_name_stream << "Testing " << (descending ? "descending" : "ascending")
                   << " order sort with ";

  if (alg == 0) {
    test_name_stream << "parallel_for of serial sorts.";
  } else if (alg == 1) {
    test_name_stream << "for loop of parallel sorts.";
  } else if (alg == 2) {
    test_name_stream << "two full pair sorts.";
  } else if (alg == 3) {
    test_name_stream << "algorithm based on device and data.";
  }

  ::std::string test_name = std::move(test_name_stream).str();
  return test_name;
}

int test_with_device_offsets(bool descending, int algorithm) {
  int num_items = 7;
  int num_segs = 3;
  int num_offs = 4;
  int offsets[] = {0, 3, 3, 7};
  int keys[] = {8, 6, 7, 5, 3, 0, 9};
  int out_keys[7];

  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue q = dev_ct1.default_queue();
  int *d_offsets = sycl::malloc_device<int>(num_offs, q);
  int *d_keys = sycl::malloc_device<int>(num_items, q);
  int *d_out_keys = sycl::malloc_device<int>(num_items, q);


  q.memcpy(d_keys, keys, sizeof(int) * num_items).wait();
  q.memcpy(d_offsets, offsets, sizeof(int) * num_offs).wait();

  if (algorithm == 0) {
    dpct::internal::segmented_sort_keys_by_parallel_for_of_sorts(
        oneapi::dpl::execution::make_device_policy(q), d_keys, d_out_keys,
        num_items, num_segs, d_offsets, d_offsets + 1, descending);
  } else if (algorithm == 1) {
    dpct::internal::segmented_sort_keys_by_parallel_sorts(
        oneapi::dpl::execution::make_device_policy(q), d_keys, d_out_keys,
        num_items, num_segs, d_offsets, d_offsets + 1, descending);
  } else if (algorithm == 2) {
    dpct::internal::segmented_sort_keys_by_two_pair_sorts(
        oneapi::dpl::execution::make_device_policy(q), d_keys, d_out_keys,
        num_items, num_segs, d_offsets, d_offsets + 1, descending);
  } else if (algorithm == 3) // this will be the one used for the mapping,
                             // others are for timing purposes
  {
    dpct::segmented_sort_keys(oneapi::dpl::execution::make_device_policy(q),
                               d_keys, d_out_keys, num_items, num_segs,
                               d_offsets, d_offsets + 1, descending);
  }

  q.memcpy(out_keys, d_out_keys, sizeof(int) * num_items).wait();

  ::std::string descending_str;
  if (descending) {
    descending_str = "error in descending device segmented sort";
  } else {
    descending_str = "error in ascending device segmented sort";
  }

  auto check_expected = [&descending_str](auto expected, auto actual) {
    return ASSERT_EQUAL(expected, actual, descending_str.c_str());
  };

  int tests_failed = 0;
  if (!descending) {
    tests_failed += check_expected(6, out_keys[0]);
    tests_failed += check_expected(7, out_keys[1]);
    tests_failed += check_expected(8, out_keys[2]);
    tests_failed += check_expected(0, out_keys[3]);
    tests_failed += check_expected(3, out_keys[4]);
    tests_failed += check_expected(5, out_keys[5]);
    tests_failed += check_expected(9, out_keys[6]);
  } else {
    tests_failed += check_expected(8, out_keys[0]);
    tests_failed += check_expected(7, out_keys[1]);
    tests_failed += check_expected(6, out_keys[2]);
    tests_failed += check_expected(9, out_keys[3]);
    tests_failed += check_expected(5, out_keys[4]);
    tests_failed += check_expected(3, out_keys[5]);
    tests_failed += check_expected(0, out_keys[6]);
  }
  sycl::free(d_offsets, q);
  sycl::free(d_keys, q);
  sycl::free(d_out_keys, q);
  return tests_failed;
}

int main() {

  int alg_start = 0;
  int alg_end = 4;
  int test_suites_failed = 0;
    int64_t nsegments = 100;
    int64_t nsort = 100;
    int64_t n = nsegments * nsort;

  for (int alg = alg_start; alg < alg_end; alg++) {

    for (int descending = 0; descending < 2; descending++) {
      int tests_failed = 0;

      tests_failed += test_with_generated_offsets<float>(nsegments, nsort, n,
                                                         descending, alg);

      tests_failed += test_with_device_offsets(descending, alg);

      test_suites_failed +=
          test_passed(tests_failed, GetAlgorithmName(alg, descending));
    }
  }
  {
    //test with io_iterator_pair
    int tests_failed = 0;
    tests_failed += test_with_generated_offsets<float>(nsegments, nsort, n, false, 3, true);
  
      test_suites_failed +=
          test_passed(tests_failed, "Test segmented sort with io_iterator_pair");
  }

  std::cout << std::endl
            << test_suites_failed << " failing test(s) detected." << std::endl;
  if (test_suites_failed == 0) {
    return 0;
  }
  return 1;
}
