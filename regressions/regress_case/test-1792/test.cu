// ====------ test.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>

template <typename T>
struct square {
  __host__ __device__  T operator()(const T& x) const { return x * x; }
};

int main() {

}

void check_transform_reduce() {
  float x[4] = {1.0, 2.0, 3.0, 4.0};
  thrust::device_vector<float> d_x(x, x + 4);
  square<float>        unary_op;
  thrust::plus<float> binary_op;
  float init = 0;

  float norm     = thrust::transform_reduce(d_x.begin(), d_x.end(), unary_op, init, binary_op);
  float normSqrt = std::sqrt(thrust::transform_reduce(d_x.begin(), d_x.end(), unary_op, init, binary_op));
}

template <typename T>
class C {
  T *data;
public:
  C() {
    this->data = 0;
  }

  inline T *raw() {
    return thrust::raw_pointer_cast(this->data);
  }
  inline const T *raw() const {
    return thrust::raw_pointer_cast(this->data + 2);
  }
};
