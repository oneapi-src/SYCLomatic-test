// ====------ test.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include "cuda_runtime.h"
#include "cuda.h"
#include "cusparse.h"
#include "nvml.h"
#include <vector>

int main(int argc, char **argv) {

    CUDA_ARRAY_DESCRIPTOR *pcad;

    cudaMemcpy3DParms *p1c3d;

    const cudaMemcpy3DParms *p2c3d;

    static cudaMemcpy3DParms *p3c3d;

    static volatile cudaMemcpy3DParms *p4c3d;

    std::vector<cudaMemcpy3DParms *> vc3dp;

    CUexternalMemory cum;

    CUexternalSemaphore cus;

    CUgraph cug;

    CUgraphExec cuge;

    CUgraphNode cugn;

    CUgraphicsResource cugr;

    nvmlDevice_t nvmld;

    nvmlReturn_t nvmlr;

    nvmlMemory_t nvmlm;

    nvmlValueType_t nvmlvt;

    nvmlValue_t nvmlv;

    nvmlInit();

    nvmlInit_v2();

    char Ver[10];

    nvmlSystemGetDriverVersion(Ver, 10);

    unsigned int dc;

    nvmlDeviceGetCount_v2(&dc);

    nvmlShutdown();
    return 0;
}

