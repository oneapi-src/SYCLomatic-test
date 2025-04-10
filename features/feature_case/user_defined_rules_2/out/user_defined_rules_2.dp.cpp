#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <c10/xpu/XPUStream.h>
// ===------ user_defined_rules_2.cu ---------------------- *- CUDA -* ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===---------------------------------------------------------------------===//

#include "ATen/cuda/CUDATensorMethods.cuh"
#include "c10/cuda/CUDAGuard.h"
#include "c10/cuda/CUDAStream.h"

void foo1_kernel() {}
void foo1() {
  static_cast<sycl::queue &>(c10::xpu::getCurrentXPUStream())
      .parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            foo1_kernel();
          });
}

void foo2_kernel(double *d) {}

void foo2() {
  double *d;
  cudaMalloc(&d, sizeof(double));
  {
    dpct::has_capability_or_fail(
        static_cast<sycl::queue &>(c10::xpu::getCurrentXPUStream())
            .get_device(),
        {sycl::aspect::fp64});

    static_cast<sycl::queue &>(c10::xpu::getCurrentXPUStream())
        .parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
            [=](sycl::nd_item<3> item_ct1) {
              foo2_kernel(d);
            });
  }
  dpct::dpct_free(d,
                  static_cast<sycl::queue &>(c10::xpu::getCurrentXPUStream()));
}

int main() {
  std::optional<c10::Device> device;
  c10::cuda::OptionalCUDAGuard device_guard(device);
  foo1();
  foo2();
  dpct::get_current_device().queues_wait_and_throw();
  return 0;
}
