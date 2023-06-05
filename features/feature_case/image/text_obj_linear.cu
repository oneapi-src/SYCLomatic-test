// ====------------ text_obj_linear.cu---------- *- CUDA -* -------------===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <iostream>

using namespace std;

#define SIZE 4

__global__ void kernal(int *output, cudaTextureObject_t tex) {
  for (int i = 0; i < SIZE; ++i) {
    auto ret = tex1Dfetch<int4>(tex, i);
    output[4 * i] = ret.x;
    output[4 * i + 1] = ret.y;
    output[4 * i + 2] = ret.z;
    output[4 * i + 3] = ret.w;
  }
}

int main() {
  int4 expect[SIZE];
  for (int i = 0; i < SIZE; i++)
    expect[i] = {i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3};
  int4 *input;
  cudaMalloc(&input, sizeof(expect));
  cudaMemcpy(input, &expect, sizeof(expect), cudaMemcpyHostToDevice);

  int *output;
  cudaMallocManaged(&output, sizeof(expect));

  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeLinear;
  resDesc.res.linear.devPtr = input;
  resDesc.res.linear.desc = cudaCreateChannelDesc<int4>();
  resDesc.res.linear.sizeInBytes = sizeof(int4) * SIZE;

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.filterMode = cudaFilterModePoint;

  cudaTextureObject_t tex;
  cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

  kernal<<<1, 1>>>(output, tex);
  cudaDeviceSynchronize();
  cudaDestroyTextureObject(tex);
  cudaFree(input);
  for (int i = 0; i < SIZE; ++i) {
    cout << "{" << output[4 * i] << ", " << output[4 * i + 1] << ", "
         << output[4 * i + 2] << ", " << output[4 * i + 3] << "}" << endl;
  }
  for (int i = 0; i < SIZE; ++i) {
    if (output[4 * i] != expect[i].x || output[4 * i + 1] != expect[i].y ||
        output[4 * i + 2] != expect[i].z || output[4 * i + 3] != expect[i].w) {
      cout << "test failed" << endl;
      return 1;
    }
  }
  return 0;
}
