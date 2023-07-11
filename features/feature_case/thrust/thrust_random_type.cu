// ====------ thrust_random_type.cu--------------- *- CUDA -*-------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/transform.h>

struct random_1 {
  __host__ __device__ float operator()(const unsigned int n) const {
    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<float> dist(1.0f, 2.0f);
    rng.discard(n);

    return dist(rng);
  }
};

struct random_2 {
  __device__ float operator()(const unsigned int n) {
    thrust::default_random_engine rng;
    rng.discard(n);
    return (float)rng() / thrust::default_random_engine::max;
  }
};

int main(void) {
  {
    const int N = 20;
    thrust::device_vector<float> numbers(N);
    thrust::counting_iterator<unsigned int> index_sequence_begin(0);

    thrust::transform(index_sequence_begin, index_sequence_begin + N,
                      numbers.begin(), random_1());

    for (int i = 0; i < N; i++) {
      if (numbers[i] > 2.0f || numbers[i] < 1.0f) {
        std::cout << "Test1 failed\n";
        return -1;
      }
    }

    thrust::transform(index_sequence_begin, index_sequence_begin + N,
                      numbers.begin(), random_2());
    for (int i = 0; i < N; i++) {
      if (numbers[i] > 1.0f || numbers[i] < 0.0f) {
        std::cout << "Test2 failed\n";
        return -1;
      }
    }
  }

  {

    // create a uniform_int_distribution to produce ints from [-5,10]
    thrust::uniform_int_distribution<int> dist(-5, 10);
    if (dist.min() != -5 || dist.max() < 10 || dist.a() != -5 ||
        dist.b() != 10) {
      std::cout << "Test3 failed\n";
      return -1;
    }
  }

  {
    // create a normal_distribution to produce floats from the Normal
    // distribution with mean 1.0 and standard deviation 2.0
    thrust::normal_distribution<float> dist(1.0f, 2.0f);

    if (dist.mean() != 1.0f || dist.stddev() != 2.0f) {
      std::cout << "Test4 failed\n";
      return -1;
    }
  }

  std::cout << "Test passed\n";
  return 0;
}
