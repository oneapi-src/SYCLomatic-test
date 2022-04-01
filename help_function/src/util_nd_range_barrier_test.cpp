// ====------ util_nd_range_barrier_test.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <cstring>
#include <dpct/dpct.hpp>
#include <iostream>
#include <stdio.h>

void kernel_1(
    sycl::nd_item<3> item_ct1, const sycl::stream &stream_ct1,
    sycl::atomic_ref<unsigned int, sycl::memory_order::seq_cst,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
        &sync_ct1) {
  dpct::experimental::nd_range_barrier(item_ct1, sync_ct1);

  stream_ct1 << "kernel_1 dim3 run\n";
}

void kernel_2(
    sycl::nd_item<3> item_ct1, const sycl::stream &stream_ct1,
    sycl::atomic_ref<unsigned int, sycl::memory_order::seq_cst,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
        &sync_ct1) {
  dpct::experimental::nd_range_barrier(item_ct1, sync_ct1);

  dpct::experimental::nd_range_barrier(item_ct1, sync_ct1);
  stream_ct1 << "kernel_2 dim3 run\n";
}

void util_nd_range_barrier_dim3_test() {

  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  {
    dpct::global_memory<unsigned int, 0> d_sync_ct1;
    unsigned *sync_ct1 = d_sync_ct1.get_ptr(dpct::get_default_queue());
    dpct::get_default_queue().memset(sync_ct1, 0, sizeof(int)).wait();

    q_ct1
        .submit([&](sycl::handler &cgh) {
          sycl::stream stream_ct1(64 * 1024, 80, cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(sycl::range<3>(1, 1, 4) *
                                    sycl::range<3>(1, 1, 4),
                                sycl::range<3>(1, 1, 4)),
              [=](sycl::nd_item<3> item_ct1) {
                auto atm_sync_ct1 = sycl::atomic_ref<
                    unsigned int, sycl::memory_order::seq_cst,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>(sync_ct1[0]);
                kernel_1(item_ct1, stream_ct1, atm_sync_ct1);
              });
        })
        .wait();
  }
  dev_ct1.queues_wait_and_throw();

  {

    dpct::global_memory<unsigned int, 0> d_sync_ct1;
    unsigned *sync_ct1 = d_sync_ct1.get_ptr(dpct::get_default_queue());
    dpct::get_default_queue().memset(sync_ct1, 0, sizeof(int)).wait();

    q_ct1
        .submit([&](sycl::handler &cgh) {
          sycl::stream stream_ct1(64 * 1024, 80, cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(sycl::range<3>(1, 1, 4) *
                                    sycl::range<3>(1, 1, 4),
                                sycl::range<3>(1, 1, 4)),
              [=](sycl::nd_item<3> item_ct1) {
                auto atm_sync_ct1 = sycl::atomic_ref<
                    unsigned int, sycl::memory_order::seq_cst,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>(sync_ct1[0]);
                kernel_2(item_ct1, stream_ct1, atm_sync_ct1);
              });
        })
        .wait();
  }
  dev_ct1.queues_wait_and_throw();
  printf("util_nd_range_barrier_dim3_test 1 passed!\n");
}

void kernel_1(
    sycl::nd_item<1> item_ct1, const sycl::stream &stream_ct1,
    sycl::atomic_ref<unsigned int, sycl::memory_order::seq_cst,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
        &sync_ct1) {
  dpct::experimental::nd_range_barrier(item_ct1, sync_ct1);

  stream_ct1 << "kernel_1 dim1 run\n";
}

void kernel_2(
    sycl::nd_item<1> item_ct1, const sycl::stream &stream_ct1,
    sycl::atomic_ref<unsigned int, sycl::memory_order::seq_cst,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
        &sync_ct1) {
  dpct::experimental::nd_range_barrier(item_ct1, sync_ct1);

  dpct::experimental::nd_range_barrier(item_ct1, sync_ct1);
  stream_ct1 << "kernel_2 dim1 run\n";
}

void util_nd_range_barrier_dim1_test() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

  {
    dpct::global_memory<unsigned int, 0> d_sync_ct1;
    unsigned *sync_ct1 = d_sync_ct1.get_ptr(dpct::get_default_queue());
    dpct::get_default_queue().memset(sync_ct1, 0, sizeof(int)).wait();

    q_ct1
        .submit([&](sycl::handler &cgh) {
          sycl::stream stream_ct1(64 * 1024, 80, cgh);

          cgh.parallel_for(
              sycl::nd_range<1>(sycl::range<1>(4) * sycl::range<1>(4),
                                sycl::range<1>(4)),
              [=](sycl::nd_item<1> item_ct1) {
                auto atm_sync_ct1 = sycl::atomic_ref<
                    unsigned int, sycl::memory_order::seq_cst,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>(sync_ct1[0]);
                kernel_1(item_ct1, stream_ct1, atm_sync_ct1);
              });
        })
        .wait();
  }

  dev_ct1.queues_wait_and_throw();
  {
    dpct::global_memory<unsigned int, 0> d_sync_ct1;
    unsigned *sync_ct1 = d_sync_ct1.get_ptr(dpct::get_default_queue());
    dpct::get_default_queue().memset(sync_ct1, 0, sizeof(int)).wait();

    q_ct1
        .submit([&](sycl::handler &cgh) {
          sycl::stream stream_ct1(64 * 1024, 80, cgh);

          cgh.parallel_for(
              sycl::nd_range<1>(sycl::range<1>(4) * sycl::range<1>(4),
                                sycl::range<1>(4)),
              [=](sycl::nd_item<1> item_ct1) {
                auto atm_sync_ct1 = sycl::atomic_ref<
                    unsigned int, sycl::memory_order::seq_cst,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>(sync_ct1[0]);
                kernel_2(item_ct1, stream_ct1, atm_sync_ct1);
              });
        })
        .wait();
  }
  dev_ct1.queues_wait_and_throw();
  printf("util_nd_range_barrier_dim1_test 2 passed!\n");
}

int main() {
  util_nd_range_barrier_dim3_test();
  util_nd_range_barrier_dim1_test();
  return 0;
}
