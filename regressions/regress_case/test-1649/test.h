#ifndef LLVM_CLANG_TEST_DPCT_CUDA_ARCH_TEST
#define LLVM_CLANG_TEST_DPCT_CUDA_ARCH_TEST
#include <cuda_runtime.h>
#include <stdio.h>
__host__ __device__ static int Env_cuda_thread_in_threadblock(int axis);

template<typename T>
__host__ __device__ int test(T a, T b);
#endif //LLVM_CLANG_TEST_DPCT_CUDA_ARCH_TEST