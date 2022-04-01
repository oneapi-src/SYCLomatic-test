// ====------ memcpy.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include<cuda.h>
#include"helper_common.h"
#include"helper_report.h"

#define MAX_DATA 100
#define RAND_RANGE 1000
#define GRID_SIZE 1
#define BLOCK_SIZE 256

#define real float
// Copy the data from host to symbol and then copy back from symbol to host.
// Compare the copy result before and after data.
class structData{
public:
    real x;
    real y;
    bool compareStructData(const structData&);
};

bool structData::compareStructData(const structData& data) {
    if (comparFloat(this->x, data.x) && comparFloat(this->y, data.y)) {
        return true;
    }
    Report::fail("Compare the float failed.");
    return false;
}


static __constant__ structData d_structData[MAX_DATA];

int testMemcpy() {
    int failStatus = 0; //Default the case is pass.
    srand(static_cast<real>(time(0)));
    //structData h_structData;
    //structData o_structData;
    structData h_structData[MAX_DATA];
    structData o_structData[MAX_DATA];
    for(size_t i = 0; i < MAX_DATA; i++) {
        h_structData[i].x = static_cast<real>(rand()) / static_cast<real>(RAND_RANGE);
        h_structData[i].y = static_cast<real>(rand()) / static_cast<real>(RAND_RANGE);
    }

    cudaMemcpyToSymbol(d_structData, h_structData, MAX_DATA * sizeof(structData));
    // cudaCheckErrors("cudaMemcpyToSymbol");
    cudaMemcpyFromSymbol(o_structData, d_structData, MAX_DATA * sizeof(structData));
    // cudaCheckErrors("cudaMemcpy from symbol");
    // Compare the result of the data.
    for (int i = 0; i < MAX_DATA; i++) {
        if (! h_structData[i].compareStructData(o_structData[i])) {
            Report::fail("Case failed " + i);
            failStatus = 1;
            break;
        }
    }
    if (failStatus != 0) {
        return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if(testMemcpy() != 0) {
        Report::fail("Test math failed.\n");
        return -1;
    }
    return 0;
}