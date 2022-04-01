// ====------ expected.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
__constant__ float const_angle[360];
void simple_kernel(float *d_array) {
	d_array[0] = const_angle[0];
	return;
}