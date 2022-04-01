// ====------ thrust-binary-search.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "report.h"

class lessThanFunc {
public:
  __host__ __device__ bool operator()(int a, int b) {return a < b;}
};

class greaterThanFunc {
public:
  __host__ __device__ bool operator()(int a, int b) {return a > b;}
};

class equalFunc {
public:
  __host__ __device__ bool operator()(int a, int b) {return a == b;}
};

void checkNoPolicy() {
  thrust::device_vector<int> input(3);
  input[0] = 0;
  input[1] = 2;
  input[2] = 4;
  thrust::device_vector<int> values(2);
  values[0] = 2;
  values[1] = 3;
  thrust::device_vector<bool> result(2);
  thrust::binary_search(input.begin(), input.end(), values.begin(), values.end(), result.begin());
  Report::check("No policy - No comparator - Search for 2", result[0], true);
  Report::check("No Policy - No comparator - Search for 3", result[1], false);

  thrust::binary_search(input.begin(), input.end(), values.begin(), values.end(), result.begin(), lessThanFunc());
  Report::check("No policy - lessThan comparator - Search for 2", result[0], true);
  Report::check("No Policy - lessThan comparator - Search for 3", result[1], false);

  input[0] = 4;
  input[1] = 2;
  input[2] = 0;
  thrust::binary_search(input.begin(), input.end(), values.begin(), values.end(), result.begin(), greaterThanFunc());
  Report::check("No policy - greaterThan comparator - Search for 2", result[0], true);
  Report::check("No Policy - greaterThan comparator - Search for 3", result[1], false);

  // Using a comparator that doesn't conform to the LessThanComparable:
  // thrust::binary_search returns the 'expected' results
  // oneapi::dpl::binary_search doesn't return the same results with device policy
  //thrust::binary_search(input.begin(), input.end(), values.begin(), values.end(), result.begin(), equalFunc());
  //Report::check("No policy - equal comparator - Search for 2", result[0], true);
  //Report::check("No Policy - equal comparator - Search for 3", result[1], true);
}

void checkWithPolicy() {
  thrust::host_vector<int> input(3);
  input[0] = 0;
  input[1] = 2;
  input[2] = 4;
  thrust::host_vector<int> values(2);
  values[0] = 2;
  values[1] = 3;
  thrust::host_vector<bool> result(2);
  thrust::binary_search(thrust::seq, input.begin(), input.end(), values.begin(), values.end(), result.begin());
  Report::check("With policy - No comparator - Search for 2", result[0], true);
  Report::check("With Policy - No comparator - Search for 3", result[1], false);

  thrust::binary_search(thrust::seq, input.begin(), input.end(), values.begin(), values.end(), result.begin(), lessThanFunc());
  Report::check("With policy - lessThan comparator - Search for 2", result[0], true);
  Report::check("With Policy - lessThan comparator - Search for 3", result[1], false);

  input[0] = 4;
  input[1] = 2;
  input[2] = 0;
  thrust::binary_search(thrust::seq, input.begin(), input.end(), values.begin(), values.end(), result.begin(), greaterThanFunc());
  Report::check("With policy - greaterThan comparator - Search for 2", result[0], true);
  Report::check("With Policy - greaterThan comparator - Search for 3", result[1], false);

  thrust::binary_search(thrust::seq, input.begin(), input.end(), values.begin(), values.end(), result.begin(), equalFunc());
  Report::check("With policy - equal comparator - Search for 2", result[0], true);
  Report::check("With Policy - equal comparator - Search for 3", result[1], true);
}

int main() {
  Report::start("thrust::binary_search");
  checkNoPolicy();
  checkWithPolicy();
  return Report::finish();
}
