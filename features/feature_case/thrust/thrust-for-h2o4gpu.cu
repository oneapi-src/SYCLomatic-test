// ====------ thrust-for-h2o4gpu.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <algorithm>
#include <thrust/inner_product.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <thrust/tuple.h>
#include <thrust/device_ptr.h>


template <typename T> struct is_even {
  __host__ __device__ bool operator()(T x) const {
    return (static_cast<unsigned int>(x) & 1) == 0;
  }
};

template <typename T> struct absolute_value {
  __host__ __device__ void operator()(T &x) const { x = (x > 0 ? x : -x); }
};

template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Predicate, typename Iterator3>
__global__ void copy_if_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result1, Predicate pred, Iterator3 result2)
{
  *result2 = thrust::copy_if(exec, first, last, result1, pred);
}

template<typename T>
struct is_foo_test {
    __host__ __device__ bool operator()(const T a) const {
        return true;
    }
};

void foo() {

  thrust::host_vector<int> h_data(10, 1);
  thrust::host_vector<int> h_result(10);
  thrust::device_vector<int> *data[10];
  thrust::device_vector<int> d_new_potential_centroids(10);
  auto range = thrust::make_counting_iterator(0);

  thrust::copy_if(h_data.begin(), h_data.end(), h_result.begin(), is_even<int>());
  thrust::copy_if(thrust::seq, h_data.begin(), h_data.end(), h_result.begin(), is_even<int>());
  thrust::copy_if((*data[0]).begin(), (*data[0]).end(), range, d_new_potential_centroids.begin(),[=] __device__(int idx) { return true; });

  std::vector<thrust::device_vector<int>> d(10);
  auto t = thrust::make_counting_iterator(0);
  auto min_costs_ptr = thrust::raw_pointer_cast(d[0].data());
  int pot_cent_num = thrust::count_if(t, t + 10, [=] __device__(int idx) { return true;});

  {
  float *_de = NULL;
  float fill_value = 0.0;

  thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(static_cast<float *>(&_de[0]));
  thrust::fill(dev_ptr, dev_ptr + 10, fill_value);
  thrust::fill_n(dev_ptr, 10, fill_value);
  float M_inner = thrust::inner_product(dev_ptr, dev_ptr + 10, dev_ptr, 0.0f);
  }

 {
  thrust::device_vector<double> t;
  thrust::for_each( t.begin(), t.end(), absolute_value<double>());
 }

 {
  int min = thrust::min(1, 2);
  int max = thrust::max(1, 2);
 }

 {
  thrust::device_vector<int> a, b, c;
  thrust::sort_by_key(a.begin(), b.end(), c.begin());
 }

 {
  const int N = 1000;
  thrust::device_vector<float> t1(N);
  thrust::device_vector<float> t2(N);
  thrust::device_vector<float> t3(N);
  thrust::transform(t1.begin(), t1.end(), t2.begin(), t3.begin(), thrust::divides<float>());
  thrust::transform(t1.begin(), t1.end(), t2.begin(), t3.begin(), thrust::multiplies<float>());
  thrust::transform(t1.begin(), t1.end(), t2.begin(), t3.begin(), thrust::plus<float>());
  thrust::transform(t1.begin(), t1.end(), t2.begin(), t3.begin(), thrust::modulus<float>());
 }

 {
    thrust::device_vector<int> data(4);
    thrust::transform(data.begin(), data.end(), thrust::make_constant_iterator(10), data.begin(), thrust::divides<int>());
 }

 {
    thrust::tuple<int, const char *> t(13, "foo");
    std::cout << "The 1st value of t is " << thrust::get<0>(t) << std::endl;
    auto ret = thrust::make_tuple(3, 4);
 }

 {
   int a;
   double b;
   thrust::tie(a, b) = thrust::make_tuple(1, 2.0);
 }

 {
   using TupleTy = thrust::tuple<int, const char *>;
   using EleType_0 = typename thrust::tuple_element<0, TupleTy>::type;
   using EleType_1 = thrust::tuple_element<1, thrust::tuple<int, const char *>>::type;
   typedef typename thrust::tuple_element<0, thrust::tuple<int, typename thrust::tuple_element<1, thrust::tuple<int, const char *>>::type>>::type EleType_2;
   static_assert(std::is_same<int, EleType_0>::value, "EleType_0 should be alias of int");
   static_assert(std::is_same<const char *, EleType_1>::value, "EleType_1 should be alias of const char *");
   static_assert(std::is_same<int, EleType_2>::value, "EleType_2 should be alias of int");

   typename thrust::tuple_element<0, TupleTy>::type v0;
   extern thrust::tuple_element<0, TupleTy>::type bar1();
   extern void foo1(typename thrust::tuple_element<0, TupleTy>::type v1);

   struct
   {
     thrust::tuple_element<0, thrust::tuple<int, const char *>>::type m = 10;
     thrust::tuple_element<1, thrust::tuple<int, const char *>>::type s = "struct st";
   } st;
   std::cout << st.m << ", " << st.s << std::endl;
 }

 {
   using TupleTy = thrust::tuple<int, double, const char *>;
   const int size = thrust::tuple_size<TupleTy>::value;
   static_assert(size == 3, "TupleTy size shoud be 3");
 }

 {
  thrust::device_vector<int> int_in(3);
  thrust::device_vector<float> float_in(3);
  auto ret = thrust::make_zip_iterator(thrust::make_tuple(int_in.begin(), float_in.begin()));
  auto arg = thrust::make_tuple(int_in.begin(), float_in.begin());
  auto ret_1 = thrust::make_zip_iterator(arg);
 }

 {
  int x =  137;
  int y = -137;
  thrust::maximum<int> mx;
  int value = mx(x,y);
 }

 {
  int data[10];
  thrust::device_ptr<int> begin = thrust::device_pointer_cast(&data[0]);
  thrust::device_ptr<int> end=begin + 10;
  bool h_result = thrust::transform_reduce(begin, end, is_foo_test<int>(), 0, thrust::plus<bool>());
  bool h_result_1 = thrust::transform_reduce(thrust::seq, begin, end, is_foo_test<int>(), 0, thrust::plus<bool>());
  auto ptrs = thrust::make_tuple(begin, end);
  int num = thrust::get<1>(ptrs) - thrust::get<0>(ptrs);
  assert(num == 10);
 }
}
