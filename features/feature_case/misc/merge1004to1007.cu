// ====------ merge1004to1007.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

int main() {
  int device1 = 0;
  int device2 = 1;
  int attribute = 0;
  cudaDeviceGetP2PAttribute(&attribute, cudaDevP2PAttrAccessSupported, device1, device2);
  return 0;
}