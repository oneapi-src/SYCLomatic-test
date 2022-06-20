// ====------ cudnn-memory.cu ---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <vector>

template<cudnnDataType_t T>
struct dt_trait{
    typedef void type;
};
template<>
struct dt_trait<CUDNN_DATA_FLOAT>{
    typedef float type;
};

template<>
struct dt_trait<CUDNN_DATA_INT32>{
    typedef int type;
};
template<>
struct dt_trait<CUDNN_DATA_HALF>{
    typedef float type;
};

template<typename T>
void check(std::vector<T> &expect, std::vector<T> &actual, int num) {
  for(int i = 0; i < num; i++){
      if(expect[i] != actual[i]) {
          std::cout << "test failed" << std::endl;
          std::cout << "expect:" << expect[i] << std::endl;
          std::cout << "actual:" << actual[i] << std::endl;
          exit(-1);
      }
  }
}

void test() {

    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor;

    cudnnCreate(&handle);

    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudnnSetStream(handle, stream1);

    cudnnCreateTensorDescriptor(&dataTensor);

    int on, oc, oh, ow, on_stride, oc_stride, oh_stride, ow_stride;
    size_t size;
    cudnnDataType_t odt;
    // Test 1
    cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW_VECT_C, CUDNN_DATA_INT8x4, 1, 16, 5, 5);
    cudnnGetTensor4dDescriptor(dataTensor, &odt, &on, &oc, &oh, &ow, &on_stride, &oc_stride, &oh_stride, &ow_stride);
    cudnnGetTensorSizeInBytes(dataTensor, &size);
    std::vector<int> expect1 = {1, 16, 5, 5, 100, 25, 5, 1, 400};
    std::vector<int> actual1 = {on, oc, oh, ow, on_stride, oc_stride, oh_stride, ow_stride, (int)size};
    check<int>(expect1, actual1, expect1.size());

    // Test 2
    cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, 2, 5, 5);
    cudnnGetTensor4dDescriptor(dataTensor, &odt, &on, &oc, &oh, &ow, &on_stride, &oc_stride, &oh_stride, &ow_stride);
    cudnnGetTensorSizeInBytes(dataTensor, &size);
    std::vector<int> expect2 = {1, 2, 5, 5, 50, 1, 10, 2, 200};
    std::vector<int> actual2 = {on, oc, oh, ow, on_stride, oc_stride, oh_stride, ow_stride, (int)size};
    check<int>(expect2, actual2, expect2.size());

    // Test 3
    cudnnSetTensor4dDescriptorEx(dataTensor, CUDNN_DATA_FLOAT, 1, 2, 5, 5, 50, 25, 5, 1);
    cudnnGetTensor4dDescriptor(dataTensor, &odt, &on, &oc, &oh, &ow, &on_stride, &oc_stride, &oh_stride, &ow_stride);
    cudnnGetTensorSizeInBytes(dataTensor, &size);
    std::vector<int> expect3 = {1, 2, 5, 5, 50, 25, 5, 1, 200};
    std::vector<int> actual3 = {on, oc, oh, ow, on_stride, oc_stride, oh_stride, ow_stride, (int)size};
    check<int>(expect3, actual3, expect3.size());

    int dims[4] = {1, 4, 5, 5};
    int odims[4] = {0, 0, 0, 0};
    int strides[4] = {100, 25, 5, 1};
    int ostrides[4] = {0, 0, 0 ,0};
    int ndims = 4, r_ndims = 4, ondims = 0;

    // Test 4
    cudnnSetTensorNdDescriptor(dataTensor, CUDNN_DATA_FLOAT, ndims, dims, strides);
    cudnnGetTensorNdDescriptor(dataTensor, r_ndims, &odt, &ondims, odims, ostrides);
    cudnnGetTensorSizeInBytes(dataTensor, &size);
    std::vector<int> expect4 = {4, 1, 4, 5, 5, 100, 25, 5, 1, 400};
    std::vector<int> actual4 = {ondims, odims[0], odims[1], odims[2], odims[3], ostrides[0], 
      ostrides[1], ostrides[2], ostrides[3], (int)size};
    check<int>(expect4, actual4, expect4.size());

    // Test 5
    cudnnSetTensorNdDescriptorEx(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, ndims, dims);
    cudnnGetTensorNdDescriptor(dataTensor, r_ndims, &odt, &ondims, odims, ostrides);
    cudnnGetTensorSizeInBytes(dataTensor, &size);
    std::vector<int> expect5 = {4, 1, 4, 5, 5, 100, 25, 5, 1, 400};
    std::vector<int> actual5 = {ondims, odims[0], odims[1], odims[2], odims[3], ostrides[0], 
      ostrides[1], ostrides[2], ostrides[3], (int)size};
    check<int>(expect5, actual5, expect5.size());

    // Test 6
    dims[0] = 1;
    dims[1] = 16;
    dims[2] = 5;
    dims[3] = 5;
    cudnnSetTensorNdDescriptorEx(dataTensor, CUDNN_TENSOR_NCHW_VECT_C, CUDNN_DATA_INT8x4, ndims, dims);
    cudnnGetTensorNdDescriptor(dataTensor, r_ndims, &odt, &ondims, odims, ostrides);
    cudnnGetTensorSizeInBytes(dataTensor, &size);
    std::vector<int> expect6 = {4, 1, 16, 5, 5, 100, 25, 5, 1, 400};
    std::vector<int> actual6 = {ondims, odims[0], odims[1], odims[2], odims[3], ostrides[0], 
      ostrides[1], ostrides[2], ostrides[3], (int)size};
    check<int>(expect6, actual6, expect6.size());

    // Test 7
    r_ndims = 2;
    cudnnSetTensorNdDescriptorEx(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, ndims, dims);
    cudnnGetTensorNdDescriptor(dataTensor, r_ndims, &odt, &ondims, odims, ostrides);
    cudnnGetTensorSizeInBytes(dataTensor, &size);
    std::vector<int> expect7 = {4, 1, 16, 400, 25, 1600};
    std::vector<int> actual7 = {ondims, odims[0], odims[1], ostrides[0], ostrides[1], (int)size};
    check<int>(expect7, actual7, expect7.size());
}

int main() {
    test();
    std::cout << "test passed" << std::endl;
    return 0;
}