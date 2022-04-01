// ====------ mathFunc.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include<cuda.h>
#include"helper_report.h"
#include"helper_common.h"

#define MAX_DATA 100
#define RAND_RANGE 1000
#define GRID_SIZE 1
#define BLOCK_SIZE 256

#define real float
// Kernel will failed. multi-calculate the result.
__global__ void test_math(float *data) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    data[0] = exp(0.0);   //e^0
    data[1] = log(1.0);   //log(1) == 0
    data[2] = max(1, 0);
    data[3] = sqrtf(4.0);
    data[4] = sqrt(4.0);
    data[5] = __logf(1.0);
    data[6] = __mul24(1, 2);
    data[7] = __umul24(1, 2);
    data[8] = floor(3.2);
    data[9] = pow(1, 0.0);
    data[10] = __expf(0.0);
    data[11] = fabs(-1.0);
}

int testMath() {
    size_t arraySize = 12;
    float input[arraySize];    // Input 12 float value.
    float output[arraySize];
    float expected[] = {1, 0, 1, 2, 2, 0, 2, 2, 3, 1, 1, 1};
    input[0] = 0;       //exp
    input[1] = 1;       //log
    input[2] = 0;       //max
    input[3] = 4.0;     //sqrtf
    input[4] = 4;       //sqrt
    input[5] = 1.0;       //__logf
    input[6] = 0;       //__mul24
    input[7] = 0;       //_umul24
    input[8] = 3.2;       //floor
    input[9] = 0;       //pow
    input[10] = 0;      //__expf
    input[11] = -1;      //fabs

    float *d_value;
    size_t sizeOfData = sizeof(*input) * arraySize;
    cudaMalloc((void **)&d_value, sizeOfData);
    cudaMemcpy(d_value, input, sizeOfData, cudaMemcpyHostToDevice);
    //Call the kernel to do the calculate.
    test_math<<<GRID_SIZE, BLOCK_SIZE>>>(d_value);
    cudaMemcpy(output, d_value, sizeOfData, cudaMemcpyDeviceToHost);
    for (int i = 0; i < arraySize; i++) {
        if (! comparFloat(output[i], expected[i])){
            std::cout << "Index is " << i << " Value is " << output[i] << "  Expected is " << expected[i] << std::endl;
            return -1;
        }
    }
    Report::pass("Test Math is pass.\n");
    return 0;
}

int main(int argc, char **argv) {
    if(testMath() != 0) {
        Report::fail("Test math failed.\n");
        return -1;
    }
    return 0;
}
