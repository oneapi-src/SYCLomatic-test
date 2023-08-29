// ====-------------- common.cuh ----------- *- CUDA -* -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#ifndef COMMON_H
#define COMMON_H

#define SIZE 64

void malloc1();
void free1();
void kernelWrapper1(int *d_Data);
void malloc2();
void free2();
void kernelWrapper2(int *d_Data);

#endif
