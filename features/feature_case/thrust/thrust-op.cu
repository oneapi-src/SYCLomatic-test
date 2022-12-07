// ====------ thrust-op.cu ----------------------------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include <thrust/functional.h>
#include <iostream>

__global__ void compareop_test(int *res)
{
  if (thrust::greater_equal(*res, *(res + 1)))
    *(res + 2) += 1;
  if (thrust::greater_equal(*res + 1, *(res + 1)))
    *(res + 2) += 1;
  if (thrust::greater_equal(*res, *(res + 1) + 1))
    *(res + 2) += 1;
  //*(res + 2) == 2

  if (thrust::less_equal(*res, *(res + 1)))
    *(res + 2) += 1;
  if (thrust::less_equal(*res + 1, *(res + 1)))
    *(res + 2) += 1;
  if (thrust::less_equal(*res, *(res + 1) + 1))
    *(res + 2) += 1;
  //*(res + 2) == 4

  


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
  compareop_test<<<1, 1>>>(deviceRes);

  if ()
    return 0;
  return 1;
}