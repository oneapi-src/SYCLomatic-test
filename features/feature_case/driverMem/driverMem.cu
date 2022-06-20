// ====------ driverMem.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
int main(){
    size_t result1, result2;
    int size = 32;
    float* f_A;
    CUdeviceptr f_D = 0;
    CUdeviceptr f_D2 = 0;
    CUresult r;

    cuMemHostAlloc((void **)&f_A, size, CU_MEMHOSTALLOC_DEVICEMAP);

    cuMemAllocHost((void **)&f_A, size);

    cuMemAlloc(&f_D, size);

    cuMemAllocManaged(&f_D, size, CU_MEM_ATTACH_HOST);


    CUstream stream;

    cuMemcpyHtoDAsync(f_D, f_A, size, stream);

    cuMemcpyHtoDAsync(f_D, f_A, size, 0);

    cuMemcpyHtoD(f_D, f_A, size);

    cuMemcpyDtoDAsync(f_D, f_D2, size, stream);

    r = cuMemcpyDtoDAsync(f_D, f_D2, size, stream);

    cuMemcpyDtoDAsync(f_D, f_D2, size, 0);

    r = cuMemcpyDtoDAsync(f_D, f_D2, size, 0);


    cuMemcpyDtoD(f_D, f_D2, size);

    r = cuMemcpyDtoD(f_D, f_D2, size);

    cuMemcpyDtoHAsync(f_A, f_D, size, stream);

    cuMemcpyDtoHAsync(f_A, f_D, size, 0);

    cuMemcpyDtoH(f_A, f_D, size);


    cuMemHostGetDevicePointer(&f_D, f_A, 0);

    CUDA_MEMCPY2D cpy;

    cpy.dstMemoryType = CU_MEMORYTYPE_HOST;

    cpy.dstHost = f_A;

    cpy.dstPitch = 20;

    cpy.dstY = 10;

    cpy.dstXInBytes = 15;


    cpy.srcMemoryType = CU_MEMORYTYPE_DEVICE;

    cpy.srcDevice = f_D;

    cpy.srcPitch = 20;

    cpy.srcY = 10;

    cpy.srcXInBytes = 15;


    cpy.WidthInBytes = 4;

    cpy.Height = 7;


    cuMemcpy2D(&cpy);

    cuMemcpy2DAsync(&cpy, stream);

    CUDA_MEMCPY3D cpy2;

    CUarray ca;

    cpy2.dstMemoryType = CU_MEMORYTYPE_ARRAY;

    cpy2.dstArray = ca;

    cpy2.dstPitch = 5;

    cpy2.dstHeight = 4;

    cpy2.dstY = 3;

    cpy2.dstZ = 2;

    cpy2.dstXInBytes = 1;

    cpy2.dstLOD = 0;


    cpy2.srcMemoryType = CU_MEMORYTYPE_HOST;

    cpy2.srcHost = f_A;

    cpy2.srcPitch = 5;

    cpy2.srcHeight = 4;

    cpy2.srcY = 3;

    cpy2.srcZ = 2;

    cpy2.srcXInBytes = 1;

    cpy2.srcLOD = 0;


    cpy2.WidthInBytes = 3;

    cpy2.Height = 2;

    cpy2.Depth = 1;

    cuMemcpy3D(&cpy2);

    float *h_A = (float *)malloc(100);
    cuMemFreeHost(h_A);

    unsigned int* pFlags;

    cuMemAllocHost((void **)&f_A, size);

    cuMemHostGetFlags(pFlags, f_A);

    cuMemHostRegister((void *)pFlags, size, CU_MEMHOSTREGISTER_PORTABLE);

    cuMemHostUnregister((void *)pFlags);

    return 0;
}

