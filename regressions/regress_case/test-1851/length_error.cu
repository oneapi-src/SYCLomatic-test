// ====------ length_error.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <cstddef>
#include <memory>


using std::size_t;


template <typename ValueType>
__global__ void fill_array_kernel(int n, ValueType *__restrict__ array,
                                  ValueType val)
{
    const auto tidx = threadIdx.x;
    if (tidx < n) {
        array[tidx] = val;
    }
}

template <typename ValueType>
void fill_array(int n, ValueType *array, ValueType val)
{
    fill_array_kernel<<<1, 32>>>(n, array, val);
}


#define INSTANTIATE_FOR_EACH_VALUE_TYPE(_macro) \
    template _macro(float);                     \
    template _macro(double)

// Using DECLARE_FILL_ARRAY works
#define DECLARE_FILL_ARRAY_KERNEL(ValueType) \
    void fill_array(int num_entries, ValueType *data, ValueType val)

#define TEMPLATE_INSTANTIATE(_macro, _type) template _macro(_type)

#define INSTANTIATE_FOR_INT_TYPE(_macro) template _macro(int)
INSTANTIATE_FOR_EACH_VALUE_TYPE(DECLARE_FILL_ARRAY_KERNEL);

//error : terminate called after throwing an instance of 'std::length_error' what():  basic_string::_M_create 
//template DECLARE_FILL_ARRAY_KERNEL(int);

//error : terminate called after throwing an instance of 'std::length_error' what():  basic_string::_M_create 
//TEMPLATE_INSTANTIATE(DECLARE_FILL_ARRAY_KERNEL, int);

// correct
INSTANTIATE_FOR_INT_TYPE(DECLARE_FILL_ARRAY_KERNEL);
int main() {
	return 0;
}
