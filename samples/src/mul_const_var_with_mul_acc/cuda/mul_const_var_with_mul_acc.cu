// ====------ mul_const_var_with_mul_acc.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <stdio.h>

#include <cuda_runtime.h>
const int array_size = 32;

__constant__ int device1[array_size];
__constant__ int device2[array_size];
__constant__ int device3[array_size];
__constant__ int device4[array_size];
__constant__ int device5[array_size];
__constant__ int device6[array_size];
__constant__ int device7[array_size];
__constant__ int device8[array_size];
__constant__ int device9[array_size];
__constant__ int device10[array_size];
__constant__ int device11[array_size];
__constant__ int device12[array_size];
__constant__ int device13[array_size];
__constant__ int device14[array_size];
__constant__ int device15[array_size];
__constant__ int device16[array_size];
__constant__ int device17[array_size];
__constant__ int device18[array_size];
__constant__ int device19[array_size];
__constant__ int device20[array_size];
__constant__ int device21[array_size];
__constant__ int device22[array_size];
__constant__ int device23[array_size];
__constant__ int device24[array_size];
__constant__ int device25[array_size];
__constant__ int device26[array_size];
__constant__ int device27[array_size];
__constant__ int device28[array_size];
__constant__ int device29[array_size];
__constant__ int device30[array_size];
__constant__ int device31[array_size];
//__constant__ int device32[array_size];

int host[array_size];
__global__
void kernel(int *out) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    out[i] = device1[i] + device2[i] + device3[i] + device4[i] + device5[i] + device6[i] + device7[i] + device8[i] + device9[i] + device10[i] + device11[i] + device12[i] + device13[i] + device14[i] + device15[i] + device16[i] + device17[i] + device18[i] + device19[i] + device20[i] + device21[i] + device22[i] + device23[i] + device24[i] + device25[i] + device26[i] + device27[i] + device28[i] + device29[i] + device30[i] + device31[i]/* + device32[i]*/;
}

int main() {
    for (unsigned i = 0; i < array_size; i++)
        host[i] = i;

    cudaMemcpyToSymbol(device1, host, array_size * sizeof(int));
    cudaMemcpyToSymbol(device2, host, array_size * sizeof(int));
    cudaMemcpyToSymbol(device3, host, array_size * sizeof(int));
    cudaMemcpyToSymbol(device4, host, array_size * sizeof(int));
    cudaMemcpyToSymbol(device5, host, array_size * sizeof(int));
    cudaMemcpyToSymbol(device6, host, array_size * sizeof(int));
    cudaMemcpyToSymbol(device7, host, array_size * sizeof(int));
    cudaMemcpyToSymbol(device8, host, array_size * sizeof(int));
    cudaMemcpyToSymbol(device9, host, array_size * sizeof(int));
    cudaMemcpyToSymbol(device10, host, array_size * sizeof(int));
    cudaMemcpyToSymbol(device11, host, array_size * sizeof(int));
    cudaMemcpyToSymbol(device12, host, array_size * sizeof(int));
    cudaMemcpyToSymbol(device13, host, array_size * sizeof(int));
    cudaMemcpyToSymbol(device14, host, array_size * sizeof(int));
    cudaMemcpyToSymbol(device15, host, array_size * sizeof(int));
    cudaMemcpyToSymbol(device16, host, array_size * sizeof(int));
    cudaMemcpyToSymbol(device17, host, array_size * sizeof(int));
    cudaMemcpyToSymbol(device18, host, array_size * sizeof(int));
    cudaMemcpyToSymbol(device19, host, array_size * sizeof(int));
    cudaMemcpyToSymbol(device20, host, array_size * sizeof(int));
    cudaMemcpyToSymbol(device21, host, array_size * sizeof(int));
    cudaMemcpyToSymbol(device22, host, array_size * sizeof(int));
    cudaMemcpyToSymbol(device23, host, array_size * sizeof(int));
    cudaMemcpyToSymbol(device24, host, array_size * sizeof(int));
    cudaMemcpyToSymbol(device25, host, array_size * sizeof(int));
    cudaMemcpyToSymbol(device26, host, array_size * sizeof(int));
    cudaMemcpyToSymbol(device27, host, array_size * sizeof(int));
    cudaMemcpyToSymbol(device28, host, array_size * sizeof(int));
    cudaMemcpyToSymbol(device29, host, array_size * sizeof(int));
    cudaMemcpyToSymbol(device30, host, array_size * sizeof(int));
    cudaMemcpyToSymbol(device31, host, array_size * sizeof(int));
   // cudaMemcpyToSymbol(device32, host, array_size * sizeof(int));
    int *global;
    cudaMalloc((void **)&global, array_size * sizeof(int));

    kernel<<<dim3(1, 1, 1), dim3(array_size, 1, 1)>>>(global);

    cudaMemcpy(host, global, array_size * sizeof(int), cudaMemcpyDeviceToHost);
    for (unsigned i = 0; i < array_size; i++) {
        if (host[i] != i * 31) {
            printf("Test Failed!\n");
            return -1;
        }
    }
    printf("Test Success!\n");

}

