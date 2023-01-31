//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#include <cuda/std/tuple>

template <class T>
__host__ __device__ void
test(T *res)
{
  cuda::std::tuple<T, T, T> t = cuda::std::make_tuple(2.0f, 3.0f, 4.0f);
  *(res) = cuda::std::get<0>(t);
  *(res+1) = cuda::std::get<1>(t);
  *(res+2) = cuda::std::get<2>(t);
}

__global__ void test_global(float * res)
{
  test<float>(res);
}

int main(int, char **)
{
  
  float *floatRes = (float *)malloc(3 * sizeof(float));
  test<float>(floatRes);
  float *hostRes = (float *)malloc(3 * sizeof(float));
  float *deviceRes;
  cudaMalloc((float **)&deviceRes, 3 * sizeof(float));
  test_global<<<1, 1>>>(deviceRes);
  cudaMemcpy(hostRes, deviceRes, sizeof(float) * 3, cudaMemcpyDeviceToHost);
  cudaFree(deviceRes);

  for (int i = 0;i<3;++i){
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
