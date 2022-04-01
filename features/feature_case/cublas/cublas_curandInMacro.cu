// ====------ cublas_curandInMacro.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <cstdio>
#include <cublas_v2.h>
#include <curand.h>
#include <cuda_runtime.h>


#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
    }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
    if (stat != CURAND_STATUS_SUCCESS) {
        fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
    }
}


int main() {
    cublasHandle_t handle;
    int N = 275;
    float *d_A_S = 0;
    float *d_B_S = 0;
    float *d_C_S = 0;
    float alpha_S = 1.0f;
    float beta_S = 0.0f;
    int trans0 = 0;
    int trans1 = 1;
    int fill0 = 0;
    int side0 = 0;
    int diag0 = 0;
    int *result = 0;
    const float *x_S = 0;

    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cublasErrCheck(cublasSetStream(handle, stream1));
    cublasErrCheck(cublasGetStream(handle, &stream1));

    cublasErrCheck(cublasSgemm(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N));


    cublasErrCheck(cublasIsamax(handle, N, x_S, N, result));


    cublasErrCheck(cublasSsyrkx(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans1, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N));


    cublasErrCheck(cublasStrmm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, N, N, &alpha_S, d_A_S, N, d_B_S, N, d_C_S, N));


    float2 *d_A_C = 0;
    float2 *d_B_C = 0;
    float2 *d_C_C = 0;
    float2 alpha_C;
    float2 beta_C;
    const float2 *x_C = 0;
    float **Aarray_S = 0;
    int *PivotArray = 0;
    int *infoArray = 0;
    int batchSize = 10;


    cublasErrCheck(cublasCgemm(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, N, N, N, &alpha_C, d_A_C, N, d_B_C, N, &beta_C, d_C_C, N));

    cublasErrCheck(cublasIcamax(handle, N, x_C, N, result));

    cublasErrCheck(cublasCtrmm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, N, N, &alpha_C, d_A_C, N, d_B_C, N, d_C_C, N));

    cublasErrCheck(cublasSgetrfBatched(handle, N, Aarray_S, N, PivotArray, infoArray, batchSize));



    float * __restrict__ d_data;
    curandGenerator_t rng;
    curandErrCheck(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
    curandErrCheck(curandSetPseudoRandomGeneratorSeed(rng, 1337ull));
    curandErrCheck(curandGenerateUniform(rng, d_data, (100 + 1) * (200) * 4));
    curandErrCheck(curandDestroyGenerator(rng));

}
