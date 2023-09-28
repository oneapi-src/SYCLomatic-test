// ====------ thrust-for-RapidCFD.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <thrust/scan.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/uninitialized_fill.h>
// for cuda 12.0
#include <thrust/extrema.h>
#include <thrust/unique.h>

int main(){
    return 0;
}
template<class T = int>
class MyClass{
    T t;
};

void foo_host(){

    thrust::host_vector<int> h_input(10);
    thrust::host_vector<int> h_input2(10);
    thrust::host_vector<int> h_output(10);
    thrust::host_vector<int> h_output2(10);

    //type
    thrust::identity<int>();
    MyClass<thrust::identity<int> > M;
    MyClass<thrust::use_default> M2;

    //iterator
    thrust::make_permutation_iterator(h_input.begin(),h_input2.begin());
    thrust::make_transform_iterator(h_input.begin(), thrust::negate<int>());

    //functor
    thrust::minus<int>();
    thrust::negate<int>();
    thrust::logical_or<bool>();

    //algo
    thrust::uninitialized_fill(h_input.begin(), h_input.end(), 10);
    thrust::unique(h_input.begin(), h_input.end());
    thrust::exclusive_scan(h_input.begin(), h_input.end(), h_output.begin());
    thrust::max_element(h_input.begin(), h_input.end());
    thrust::min_element(h_input.begin(), h_input.end());
}

