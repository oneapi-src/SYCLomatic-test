// ====------ cusolverDnLn_cuda10-1.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//


// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.2, cuda-10.0
// UNSUPPORTED: v8.0, v9.0, v9.2, v10.0
#include <cstdio>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>


int main(int argc, char *argv[])
{
    cusolverDnHandle_t* cusolverH = NULL;
    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
    status = CUSOLVER_STATUS_NOT_INITIALIZED;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    int m = 0;
    int n = 0;
    int nrhs = 0;
    float A_f = 0;
    double A_d = 0.0;
    cuComplex A_c = make_cuComplex(1,0);
    cuDoubleComplex A_z = make_cuDoubleComplex(1,0);
    float B_f = 0;
    double B_d = 0.0;
    cuComplex B_c = make_cuComplex(1,0);
    cuDoubleComplex B_z = make_cuDoubleComplex(1,0);

    const float C_f = 0;
    const double C_d = 0.0;
    const cuComplex C_c = make_cuComplex(1,0);
    const cuDoubleComplex C_z = make_cuDoubleComplex(1,0);

    int lda = 0;
    int ldb = 0;
    float workspace_f = 0;
    double workspace_d = 0;
    cuComplex workspace_c = make_cuComplex(1,0);
    cuDoubleComplex workspace_z = make_cuDoubleComplex(1,0);
    int Lwork = 0;
    int devInfo = 0;

    status = cusolverDnSpotri_bufferSize(*cusolverH, uplo, n, &A_f, lda, &Lwork);
    status = cusolverDnDpotri_bufferSize(*cusolverH, uplo, n, &A_d, lda, &Lwork);
    status = cusolverDnCpotri_bufferSize(*cusolverH, uplo, n, &A_c, lda, &Lwork);
    status = cusolverDnZpotri_bufferSize(*cusolverH, uplo, n, &A_z, lda, &Lwork);


    status = cusolverDnSpotri(*cusolverH, uplo, n, &A_f, lda, &workspace_f, Lwork, &devInfo);
    cusolverDnSpotri(*cusolverH, uplo, n, &A_f, lda, &workspace_f, Lwork, &devInfo);

    status = cusolverDnDpotri(*cusolverH, uplo, n, &A_d, lda, &workspace_d, Lwork, &devInfo);
    cusolverDnDpotri(*cusolverH, uplo, n, &A_d, lda, &workspace_d, Lwork, &devInfo);

    status = cusolverDnCpotri(*cusolverH, uplo, n, &A_c, lda, &workspace_c, Lwork, &devInfo);
    cusolverDnCpotri(*cusolverH, uplo, n, &A_c, lda, &workspace_c, Lwork, &devInfo);

    status = cusolverDnZpotri(*cusolverH, uplo, n, &A_z, lda, &workspace_z, Lwork, &devInfo);
    cusolverDnZpotri(*cusolverH, uplo, n, &A_z, lda, &workspace_z, Lwork, &devInfo);
}
