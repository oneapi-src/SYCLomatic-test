// ====------ test.cu---------- *- CUDA -* ----===////
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
#include<cuda.h>
#include <thrust/host_vector.h>

const thrust::host_vector<float> transform(
    const thrust::host_vector<int>& src, size_t width, size_t height, size_t pitch)
{
    const thrust::host_vector<float> result(100, 0);
    return result;
}


template <typename T>
const thrust::host_vector<float> transformT(
    const thrust::host_vector<T>& src, size_t width, size_t height, size_t pitch)
{
    const thrust::host_vector<float> result(100, 0);
    return result;
}

const thrust::device_vector<float> transform(
    const thrust::device_vector<int>& src, size_t width, size_t height, size_t pitch)
{
    const thrust::device_vector<float> result(100, 0);
    return result;
}

template <typename T>
const thrust::device_vector<float> transformT(
    const thrust::device_vector<T>& src, size_t width, size_t height, size_t pitch)
{
    const thrust::device_vector<float> result(100, 0);
    return result;
}

void test(){

    const thrust::host_vector<float> d_actual;

    const thrust::device_vector<float> d_actual2;
}

int main() {
    return 0;
}