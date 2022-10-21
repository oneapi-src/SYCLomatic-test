// ====------ thrust-pinned-allocator.cu---------- -*- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

#define SIZE 4

int main(int argc, char *argv[])
{
    std::vector<float, thrust::system::cuda::experimental::pinned_allocator<float>> hVec(SIZE);
    std::fill(hVec.begin(), hVec.end(), 2);

    thrust::device_vector<float> dVec(hVec.size());
    thrust::copy(hVec.begin(), hVec.end(), dVec.begin());

    thrust::transform(dVec.begin(), dVec.end(), thrust::make_constant_iterator(2), dVec.begin(), thrust::multiplies<float>());
    thrust::copy(dVec.begin(), dVec.end(), hVec.begin());

    std::vector<float, thrust::cuda::experimental::pinned_allocator<float>> hVecCopy = hVec;
    for (const auto &v : hVecCopy)
    {
        // hVec elemenet should equal 4
        if(v != 4) {
            return -1;
        }
    }
    return 0;
}