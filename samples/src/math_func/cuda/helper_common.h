//===---helper_common.h--------- --------------------------------*- C++ -*---===//
////
//// Copyright (C) Intel Corporation. All rights reserved.
////
//// The information and source code contained herein is the exclusive
//// property of Intel Corporation and may not be disclosed, examined
//// or reproduced in whole or in part without explicit written authorization
//// from the company.
////
////===-----------------------------------------------------------------===//
#ifndef HELP_DPCT_TEST
#define HELP_DPCT_TEST

#include<chrono>
#include<iostream>
#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1
#define checkCudaCapabilities(arg1, arg2) 1

#define cudaCheckLastErrors(msg)                                    \
    do {                                                        \
        cudaError_t __err = cudaGetLastError();                 \
        if (__err != cudaSuccess) {                             \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n",  \
                msg, cudaGetErrorString(__err),                 \
                __FILE__, __LINE__);                            \
            fprintf(stderr, "*** FAILED - ABORTING\n");         \
            exit(1);                                            \
        }                                                       \
    } while (0)



#define cudaCheckErrors(value)                                      \
    do {                                                            \
        cudaError_t __err = value;                                  \
        if (__err != cudaSuccess) {                                 \
            fprintf(stderr, "Fatal error: (%s at %s:%d)\n",         \
                     cudaGetErrorString(__err),                     \
                    __FILE__, __LINE__);                            \
            fprintf(stderr, "*** FAILED - ABORTING\n");             \
            exit(1);                                                \
        }                                                           \
    } while(0)

inline bool comparFloat(float f1, float f2) {
    if (abs(f1 - f2) <= 1.0e-05) {
        return true;
    }
    return false;
}

inline int findCudaDevice(int argc, const char **argv) {
    int devID = 0;
    //TODO: Support the cmd-line specfic the device.
    //devID = gpuGetMaxGflopsDeviceId();
    return devID;
}
#endif
