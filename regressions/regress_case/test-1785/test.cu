// ====------ test.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include "stdio.h"
#include "omp.h"

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int main() {
    int tid, nthreads;

    cuda_hello<<<1,1>>>(); 

    #pragma omp parallel private(tid, nthreads)
    {
        tid = omp_get_thread_num();
        nthreads = omp_get_num_threads();

        printf("Greetings from thread %i of %i\n", tid, nthreads);
    }

    cudaDeviceSynchronize();
    return 0;
}