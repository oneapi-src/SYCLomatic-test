// ====------ thrust_detail_types.cu--------------- *- CUDA -*-------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <iostream>
#include <thrust/detail/allocator/allocator_traits.h>
#include <thrust/detail/type_traits.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/iterator_traits.h>

template <typename T>
typename thrust::detail::enable_if<std::is_integral<T>::value, void>::type
print_info(T value) {
  std::cout << "Integral value: " << value << std::endl;
}

template <typename T>
typename thrust::detail::enable_if<std::is_floating_point<T>::value, void>::type
print_info(T value) {
  std::cout << "Floating-point value: " << value << std::endl;
}

template <typename T> struct is_integer {
  typedef thrust::detail::false_type type;
};

template <> struct is_integer<int> { typedef thrust::detail::true_type type; };
template <> struct is_integer<long> { typedef thrust::detail::true_type type; };

template <std::size_t Len, std::size_t Align>
void test_aligned_storage_instantiation() {
  typedef thrust::detail::integral_constant<bool, false> ValidAlign;
}

template <typename T, typename U> bool check_types_same() {
  if (thrust::detail::is_same<T, U>::value) {
    std::cout << "Types T and U are the same." << std::endl;
    return true;
  } else {
    std::cout << "Types T and U are different." << std::endl;
    return false;
  }
}

template <class ExampleVector, typename NewType, typename new_alloc>
struct vector_like {
  typedef thrust::detail::vector_base<NewType, new_alloc> type;
};

int main() {
  int integer_val = 1;
  float float_val = 3.14f;
  double double_val = 2.71;

  print_info(integer_val); // Output: Integral value: 42
  print_info(float_val);   // Output: Floating-point value: 3.14
  print_info(double_val);  // Output: Floating-point value: 2.71828

  thrust::device_vector<int> A(1);
  thrust::device_vector<int> B(1);
  A[0] = 0;
  B[0] = 0;
  bool ret = thrust::detail::vector_equal(A.begin(), A.end(), B.begin());
  if (!ret) {
    printf("test failed\n");
    exit(-1);
  }

  std::cout << "is_integer<int>: " << is_integer<int>::type::value << std::endl;
  std::cout << "is_integer<long>: " << is_integer<long>::type::value
            << std::endl;
  std::cout << "is_integer<float>: " << is_integer<float>::type::value
            << std::endl;

  typedef thrust::detail::integral_constant<bool, false> ValidAlign;
  ValidAlign tt();

  typedef thrust::detail::true_type is_always_equal;

  is_always_equal foo_t();

  if (!check_types_same<int, int>() || check_types_same<float, int>() ||
      !check_types_same<double, double>()) {
    printf("test failed\n");
    exit(-1);
  }

  typedef thrust::device_vector<int>::iterator Iterator;

  // Use thrust::iterator_traits to get information about the iterator
  typedef thrust::iterator_traits<Iterator>::value_type value_type;
  typedef thrust::iterator_traits<Iterator>::difference_type difference_type;

  // Output the information obtained from iterator_traits
  std::cout << "value_type: " << typeid(value_type).name() << std::endl;
  std::cout << "difference_type: " << typeid(difference_type).name()
            << std::endl;

  printf("test passed!\n");
  return 0;
}
