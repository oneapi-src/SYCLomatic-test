// ====------ test.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <cuda_runtime.h>
__global__ void kernel(){

__threadfence_block();

__threadfence();

__threadfence_system();

}

int main() {

    kernel<<<1,1>>>();

    return 0;
}