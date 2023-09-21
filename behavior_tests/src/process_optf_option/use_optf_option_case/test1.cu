// ====------ test2.cu---------------------------------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <cuda.h>

#ifdef __FOO1__
__global__ void foo1() {}
#endif

#ifdef __FOO2__
__global__ void foo2() {}
#endif

#ifdef __FOO3__
__global__ void foo3() {}
#endif

#ifdef __FOO4__
__global__ void foo4() {}
#endif

#include "foo1.h"
#include "foo2.h"
#include "foo3.h"
#include "foo4.h"

int main() { return 0; }
