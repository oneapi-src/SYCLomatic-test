// ====------ thrust-op.cu ----------------------------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include <thrust/functional.h>
#include <iostream>

__global__ void compareop_kernel(int *res)
{
  if (thrust::greater_equal<>()(*res, *(res + 1)))
    *(res + 2) += 1;
  if (thrust::greater_equal<>()(*res + 1, *(res + 1)))
    *(res + 2) += 1;
  if (thrust::greater_equal<>()(*res, *(res + 1) + 1))
    *(res + 2) += 1;
  //*(res + 2) == 2

  if (thrust::less_equal<>()(*res, *(res + 1)))
    *(res + 2) += 1;
  if (thrust::less_equal<>()(*res + 1, *(res + 1)))
    *(res + 2) += 1;
  if (thrust::less_equal<>()(*res, *(res + 1) + 1))
    *(res + 2) += 1;
  //*(res + 2) == 4

  if (thrust::logical_and<>()(*res, *(res + 1)))
    *(res + 2) += 1;
  if (thrust::logical_and<>()(*res + 1, *(res + 1)))
    *(res + 2) += 1;
  if (thrust::logical_and<>()(*res, *(res + 1) + 1))
    *(res + 2) += 1;
  //*(res + 2) == 5

  *(res + 2) += thrust::bit_and<>()(*res, *(res + 1));
  *(res + 2) += thrust::bit_or<>()(*res, *(res + 1));
  *(res + 2) += thrust::bit_xor<>()(*res, *(res + 1));
  *(res + 2) += thrust::minimum<int>()(*res,*res);
}


int main()
{
  int *hostRes = (int *)malloc(3 * sizeof(int));
  *hostRes = 42;
  *(hostRes + 1) = 42;
  *(hostRes + 2) = 0;
  int *deviceRes;
  cudaMalloc((int **)&deviceRes, 3 * sizeof(int));
  cudaMemcpy(deviceRes, hostRes, sizeof(int) * 3, cudaMemcpyHostToDevice);
  compareop_kernel<<<1, 1>>>(deviceRes);
  cudaMemcpy(hostRes, deviceRes, sizeof(int) * 3, cudaMemcpyDeviceToHost);
  cudaFree(deviceRes);
  if (*hostRes==42&&*(hostRes+1)==42&&*(hostRes+2)==92)
  {
    free(hostRes);
    return 0;
  }
  free(hostRes);
  return 1;
}