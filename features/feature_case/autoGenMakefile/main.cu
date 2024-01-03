//===--- main.cu -------------------------------*- CUDA -*---===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===------------------------------------------------------ -===//


#include <foo.h>
#include <bar.h>

#include "define.c"
#include "kernel.cu"


int main() {
#ifdef __FOO__
    kernel<<<1,1>>>();
#endif
	return 0;
}