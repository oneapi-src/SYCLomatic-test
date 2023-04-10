// ====------ vector_add.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <cuda.h>
#include <stdio.h>
#define VECTOR_SIZE 256

#include <cuda_runtime.h>
template <typename T>
class TestVirtual {
public:

    __device__ TestVirtual() {}
// CHECK: /*
// CHECK-NEXT: DPCT1109:{{[0-9]+}}: Virtual functions cannot be called in a SYCL kernel or by functions called by the kernel. You may need to adjust the code.
// CHECK-NEXT: */
    __device__ virtual ~TestVirtual() {}
// CHECK: /*
// CHECK-NEXT: DPCT1109:{{[0-9]+}}: Virtual functions cannot be called in a SYCL kernel or by functions called by the kernel. You may need to adjust the code.
// CHECK-NEXT: */
    __device__ virtual void push(const T &&e)= 0;
};
template <typename T>
class TestSeqContainer : public TestVirtual<T> {
public:
    __device__ TestSeqContainer(int size) : index_top(-1) { m_data = new T[size]; }

    __device__ ~TestSeqContainer() {
        if (m_data) delete []m_data;
    }
    // CHECK: /*
    // CHECK-NEXT: DPCT1109:{{[0-9]+}}: Virtual functions cannot be called in a SYCL kernel or by functions called by the kernel. You may need to adjust the code.
    // CHECK-NEXT: */
    __device__ virtual void push(const T &&e) {
        if (m_data) {
           int idx = atomicAdd(&this->index_top, 1);
           m_data[idx] = e;
        }
    }
private:
    T *m_data;
    int index_top;

};
__global__ void func(){

    auto seq = new TestSeqContainer<int>(10);
    seq->push(10);
    delete seq;
}

__device__ int factorial(int n) {
    if (n <= 1) {
        return 1;
    } else {
        return n * factorial(n - 1);
    }
}

__global__ void test_kernel() {
    factorial(10);
}


int main() {
func<<<1,1>>>();
test_kernel<<<1,1>>>();
cudaDeviceSynchronize();
return 0;

}
