//===--- backprop.c ----------------------------------*- CUDA -*---===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------==//
#include <math.h>

float test(float x) {

    return (1.0 / (1.0 + exp(-x)));

}
