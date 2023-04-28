// ====------ device_info.cu---------- *- CUDA -* -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===--------------------------------------------------------------===//

#include <iostream>
#include <stdio.h>

void test0() {
  // no need to use `cudaSetDevice`
  size_t free_mem, total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);

  printf("total_mem : [%lu]\n", total_mem);
  printf("free_mem  : [%lu]\n", free_mem);
}

void test1() {
  cudaDeviceProp properties;
  cudaGetDeviceProperties(&properties, 0);

  const int id = properties.pciDeviceID;
  const cudaUUID_t uuid = properties.uuid;

  std::cout << "Device ID: " << id << std::endl;
  std::cout << "Device UUID: ";
  for (int i = 0; i < 16; i++) {
    std::cout << std::hex
              << static_cast<int>(static_cast<unsigned char>(uuid.bytes[i]))
              << " ";
  }
  std::cout << std::endl;
}

int main() {
  test0();
  test1();
}
