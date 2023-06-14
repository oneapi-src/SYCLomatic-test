// ====------------ text_obj_pitch2d.cu---------- *- CUDA -* ------------===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <iostream>

using namespace std;

#define WIDTH 4 // Must be multiple of 2, Need investigation.
#define HEIGHT 2

__global__ void kernal(int *output, cudaTextureObject_t tex) {
  for (int i = 0; i < HEIGHT; ++i) {
    for (int j = 0; j < WIDTH; ++j) {
      auto ret = tex2D<int4>(tex, j, i);
      output[4 * (WIDTH * i + j)] = ret.x;
      output[4 * (WIDTH * i + j) + 1] = ret.y;
      output[4 * (WIDTH * i + j) + 2] = ret.z;
      output[4 * (WIDTH * i + j) + 3] = ret.w;
    }
  }
}

int main() {
  int4 expect[WIDTH * HEIGHT];
  for (int i = 0; i < WIDTH * HEIGHT; i++)
    expect[i] = {i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3};
  int4 *input;
  cudaMalloc(&input, sizeof(expect));
  cudaMemcpy(input, &expect, sizeof(expect), cudaMemcpyHostToDevice);

  int *output;
  cudaMallocManaged(&output, sizeof(expect));

  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypePitch2D;
  resDesc.res.pitch2D.devPtr = input;
  resDesc.res.pitch2D.width = WIDTH;
  resDesc.res.pitch2D.height = HEIGHT;
  resDesc.res.pitch2D.desc = cudaCreateChannelDesc<int4>();
  resDesc.res.pitch2D.pitchInBytes = sizeof(int4) * WIDTH;

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
  for (int i = 0; i < HEIGHT; ++i) {
    for (int j = 0; j < WIDTH; ++j) {
      cout << "{" << output[4 * (WIDTH * i + j)] << ", "
           << output[4 * (WIDTH * i + j) + 1] << ", "
           << output[4 * (WIDTH * i + j) + 2] << ", "
           << output[4 * (WIDTH * i + j) + 3] << "}" << endl;
    }
  }
  for (int i = 0; i < WIDTH * HEIGHT; ++i) {
    if (output[4 * i] != expect[i].x || output[4 * i + 1] != expect[i].y ||
        output[4 * i + 2] != expect[i].z || output[4 * i + 3] != expect[i].w) {
      cout << "test failed" << endl;
      return 1;
    }
  }
  return 0;
}
