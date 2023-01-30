//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/complex>

#include <cuda/std/complex>

template <class T>
__host__ __device__ void
test(T *res)
{
  cuda::std::complex<T> x(1.5, 2.5);
  cuda::std::complex<T> y(2.5, 3);
  T *a = (T *)&x;
  a[0] = 5;
  a[1] = 6;
  *(res) = x.real() * x.imag();
  cuda::std::complex<T> z = x / y;
  *(res +1) =z.real();
  *(res + 2) = z.imag();
  z = x + y;
  *(res +3) =z.real();
  *(res + 4) = z.imag();
  z = x - y;
  *(res +5) =z.real();
  *(res + 6) = z.imag();
  z = x * y;
  *(res +7) =z.real();
  *(res + 8) = z.imag();
}

__global__ void test_global(float * res)
{
  test<float>(res);
}

int main(int, char **)
{
  
  float *floatRes = (float *)malloc(9 * sizeof(float));
  test<float>(floatRes);
  //test<double>(doubleRes);
  float *hostRes = (float *)malloc(9 * sizeof(float));
  float *deviceRes;
  cudaMalloc((float **)&deviceRes, 9 * sizeof(float));
  test_global<<<1, 1>>>(deviceRes);
  cudaMemcpy(hostRes, deviceRes, sizeof(float) * 9, cudaMemcpyDeviceToHost);
  cudaFree(deviceRes);

  for (int i = 0;i<9;++i){
    if(hostRes[i]!=floatRes[i]){
      free(hostRes);
      free(floatRes);
      return 1;
    }
  }
  free(hostRes);
  free(floatRes);
  return 0;

}
