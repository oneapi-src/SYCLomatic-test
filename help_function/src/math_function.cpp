// ====------ math_function.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
void kernel(unsigned int *input, unsigned int *result,
            sycl::nd_item<3> item_ct1){
        int i = item_ct1.get_local_id(2);
        result[i] = dpct::reverse_bits<unsigned int>(input[i]);
}

int main() {
 dpct::device_ext &dev_ct1 = dpct::get_current_device();
 sycl::queue &q_ct1 = dev_ct1.default_queue();
        unsigned int *input;
        unsigned int *result;
        int N = 10;
        input = sycl::malloc_shared<unsigned int>(N, q_ct1);
        result = sycl::malloc_shared<unsigned int>(N, q_ct1);

        for(int i = 0; i < N; i++){
                input[i] = i;
        }
        unsigned int verify[10] = {
                0x00000000,
                0x80000000,
                0x40000000,
                0xC0000000,
                0x20000000,
                0xA0000000,
                0x60000000,
                0xE0000000,
                0x10000000,
                0x90000000
        };
        q_ct1.submit([&](sycl::handler &cgh) {
                cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, N),
                                                   sycl::range<3>(1, 1, N)),
                                 [=](sycl::nd_item<3> item_ct1) {
                                         kernel(input, result, item_ct1);
                                 });
        });
        dev_ct1.queues_wait_and_throw();

        for(int i = 0; i < N; i++){
                if(result[i] != verify[i]){
                        printf("Verify Failed. result[%d] = %u, it should be %u.\n", i, result[i], verify[i]);
                        return -1;
                }
        }
        printf("Verify Succeed\n");
        return 0;
}
