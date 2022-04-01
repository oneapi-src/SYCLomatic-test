// ====------ cusolverDnLn.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//


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
    cublasOperation_t trans = CUBLAS_OP_N;
    cublasSideMode_t side = CUBLAS_SIDE_LEFT;
    int m = 0;
    int n = 0;
    int k = 0;
    int nrhs = 0;
    float A_f = 0;
    double A_d = 0.0;
    cuComplex A_c = make_cuComplex(1,0);
    cuDoubleComplex A_z = make_cuDoubleComplex(1,0);

    float B_f = 0;
    double B_d = 0.0;
    cuComplex B_c = make_cuComplex(1,0);
    cuDoubleComplex B_z = make_cuDoubleComplex(1,0);

    float D_f = 0;
    double D_d = 0.0;
    cuComplex D_c = make_cuComplex(1,0);
    cuDoubleComplex D_z = make_cuDoubleComplex(1,0);

    float E_f = 0;
    double E_d = 0.0;
    cuComplex E_c = make_cuComplex(1,0);
    cuDoubleComplex E_z = make_cuDoubleComplex(1,0);

    float TAU_f = 0;
    double TAU_d = 0.0;
    cuComplex TAU_c = make_cuComplex(1,0);
    cuDoubleComplex TAU_z = make_cuDoubleComplex(1,0);

    float TAUQ_f = 0;
    double TAUQ_d = 0.0;
    cuComplex TAUQ_c = make_cuComplex(1,0);
    cuDoubleComplex TAUQ_z = make_cuDoubleComplex(1,0);

    float TAUP_f = 0;
    double TAUP_d = 0.0;
    cuComplex TAUP_c = make_cuComplex(1,0);
    cuDoubleComplex TAUP_z = make_cuDoubleComplex(1,0);

    const float C_f = 0;
    const double C_d = 0.0;
    const cuComplex C_c = make_cuComplex(1,0);
    const cuDoubleComplex C_z = make_cuDoubleComplex(1,0);

    int lda = 0;
    int ldb = 0;
    const int ldc = 0;
    float workspace_f = 0;
    double workspace_d = 0;
    cuComplex workspace_c = make_cuComplex(1,0);
    cuDoubleComplex workspace_z = make_cuDoubleComplex(1,0);
    int Lwork = 0;
    int devInfo = 0;
    int devIpiv = 0;

    status = cusolverDnSpotrf_bufferSize(*cusolverH, uplo, n, &A_f, lda, &Lwork);
    status = cusolverDnDpotrf_bufferSize(*cusolverH, uplo, n, &A_d, lda, &Lwork);
    status = cusolverDnCpotrf_bufferSize(*cusolverH, uplo, n, &A_c, lda, &Lwork);
    status = cusolverDnZpotrf_bufferSize(*cusolverH, uplo, n, &A_z, lda, &Lwork);

    status = cusolverDnSgetrf_bufferSize(*cusolverH, m, n, &A_f, lda, &Lwork);
    status = cusolverDnDgetrf_bufferSize(*cusolverH, m, n, &A_d, lda, &Lwork);
    status = cusolverDnCgetrf_bufferSize(*cusolverH, m, n, &A_c, lda, &Lwork);
    status = cusolverDnZgetrf_bufferSize(*cusolverH, m, n, &A_z, lda, &Lwork);

    status = cusolverDnSpotrf(*cusolverH, uplo, n, &A_f, lda, &workspace_f, Lwork, &devInfo);
    cusolverDnSpotrf(*cusolverH, uplo, n, &A_f, lda, &workspace_f, Lwork, &devInfo);

    status = cusolverDnDpotrf(*cusolverH, uplo, n, &A_d, lda, &workspace_d, Lwork, &devInfo);
    cusolverDnDpotrf(*cusolverH, uplo, n, &A_d, lda, &workspace_d, Lwork, &devInfo);

    status = cusolverDnCpotrf(*cusolverH, uplo, n, &A_c, lda, &workspace_c, Lwork, &devInfo);
    cusolverDnCpotrf(*cusolverH, uplo, n, &A_c, lda, &workspace_c, Lwork, &devInfo);

    status = cusolverDnZpotrf(*cusolverH, uplo, n, &A_z, lda, &workspace_z, Lwork, &devInfo);
    cusolverDnZpotrf(*cusolverH, uplo, n, &A_z, lda, &workspace_z, Lwork, &devInfo);


    status = cusolverDnSpotrs(*cusolverH, uplo, n, nrhs, &C_f, lda, &B_f, ldb, &devInfo);
    cusolverDnSpotrs(*cusolverH, uplo, n, nrhs, &C_f, lda, &B_f, ldb, &devInfo);

    status = cusolverDnDpotrs(*cusolverH, uplo, n, nrhs, &C_d, lda, &B_d, ldb, &devInfo);
    cusolverDnDpotrs(*cusolverH, uplo, n, nrhs, &C_d, lda, &B_d, ldb, &devInfo);

    status = cusolverDnCpotrs(*cusolverH, uplo, n, nrhs, &C_c, lda, &B_c, ldb, &devInfo);
    cusolverDnCpotrs(*cusolverH, uplo, n, nrhs, &C_c, lda, &B_c, ldb, &devInfo);


    status = cusolverDnZpotrs(*cusolverH, uplo, n, nrhs, &C_z, lda, &B_z, ldb, &devInfo);
    cusolverDnZpotrs(*cusolverH, uplo, n, nrhs, &C_z, lda, &B_z, ldb, &devInfo);

    status = cusolverDnSgetrf(*cusolverH, m, n, &A_f, lda, &workspace_f, &devIpiv, &devInfo);
    cusolverDnSgetrf(*cusolverH, m, n, &A_f, lda, &workspace_f, &devIpiv, &devInfo);

    status = cusolverDnDgetrf(*cusolverH, m, n, &A_d, lda, &workspace_d, &devIpiv, &devInfo);
    cusolverDnDgetrf(*cusolverH, m, n, &A_d, lda, &workspace_d, &devIpiv, &devInfo);

    status = cusolverDnCgetrf(*cusolverH, m, n, &A_c, lda, &workspace_c, &devIpiv, &devInfo);
    cusolverDnCgetrf(*cusolverH, m, n, &A_c, lda, &workspace_c, &devIpiv, &devInfo);

    status = cusolverDnZgetrf(*cusolverH, m, n, &A_z, lda, &workspace_z, &devIpiv, &devInfo);
    cusolverDnZgetrf(*cusolverH, m, n, &A_z, lda, &workspace_z, &devIpiv, &devInfo);

    status = cusolverDnZgetrf(*cusolverH, m, n, &A_z, lda, &workspace_z, &devIpiv, &devInfo);
    cusolverDnZgetrf(*cusolverH, m, n, &A_z, lda, &workspace_z, &devIpiv, &devInfo);

    status = cusolverDnSgetrs(*cusolverH, trans, n, nrhs, &A_f, lda, &devIpiv, &B_f, ldb, &devInfo);
    cusolverDnSgetrs(*cusolverH, trans, n, nrhs, &A_f, lda, &devIpiv, &B_f, ldb, &devInfo);

    status = cusolverDnDgetrs(*cusolverH, trans, n, nrhs, &A_d, lda, &devIpiv, &B_d, ldb, &devInfo);
    cusolverDnDgetrs(*cusolverH, trans, n, nrhs, &A_d, lda, &devIpiv, &B_d, ldb, &devInfo);

    status = cusolverDnCgetrs(*cusolverH, trans, n, nrhs, &A_c, lda, &devIpiv, &B_c, ldb, &devInfo);
    cusolverDnCgetrs(*cusolverH, trans, n, nrhs, &A_c, lda, &devIpiv, &B_c, ldb, &devInfo);

    status = cusolverDnZgetrs(*cusolverH, trans, n, nrhs, &A_z, lda, &devIpiv, &B_z, ldb, &devInfo);
    cusolverDnZgetrs(*cusolverH, trans, n, nrhs, &A_z, lda, &devIpiv, &B_z, ldb, &devInfo);

    status = cusolverDnSgeqrf_bufferSize(*cusolverH, m, n, &A_f, lda, &Lwork);
    cusolverDnSgeqrf_bufferSize(*cusolverH, m, n, &A_f, lda, &Lwork);
    status = cusolverDnSgeqrf(*cusolverH, m, n, &A_f, lda, &TAU_f, &workspace_f, Lwork, &devInfo);
    cusolverDnSgeqrf(*cusolverH, m, n, &A_f, lda, &TAU_f, &workspace_f, Lwork, &devInfo);

    status = cusolverDnDgeqrf_bufferSize(*cusolverH, m, n, &A_d, lda, &Lwork);
    cusolverDnDgeqrf_bufferSize(*cusolverH, m, n, &A_d, lda, &Lwork);
    status = cusolverDnDgeqrf(*cusolverH, m, n, &A_d, lda, &TAU_d, &workspace_d, Lwork, &devInfo);
    cusolverDnDgeqrf(*cusolverH, m, n, &A_d, lda, &TAU_d, &workspace_d, Lwork, &devInfo);

    status = cusolverDnCgeqrf_bufferSize(*cusolverH, m, n, &A_c, lda, &Lwork);
    cusolverDnCgeqrf_bufferSize(*cusolverH, m, n, &A_c, lda, &Lwork);
    status = cusolverDnCgeqrf(*cusolverH, m, n, &A_c, lda, &TAU_c, &workspace_c, Lwork, &devInfo);
    cusolverDnCgeqrf(*cusolverH, m, n, &A_c, lda, &TAU_c, &workspace_c, Lwork, &devInfo);

    status = cusolverDnZgeqrf_bufferSize(*cusolverH, m, n, &A_z, lda, &Lwork);
    cusolverDnZgeqrf_bufferSize(*cusolverH, m, n, &A_z, lda, &Lwork);
    status = cusolverDnZgeqrf(*cusolverH, m, n, &A_z, lda, &TAU_z, &workspace_z, Lwork, &devInfo);
    cusolverDnZgeqrf(*cusolverH, m, n, &A_z, lda, &TAU_z, &workspace_z, Lwork, &devInfo);

    status = cusolverDnSormqr_bufferSize(*cusolverH, side, trans, m, n, k, &A_f, lda, &TAU_f, &C_f, ldc, &Lwork);
    cusolverDnSormqr_bufferSize(*cusolverH, side, trans, m, n, k, &A_f, lda, &TAU_f, &C_f, ldc, &Lwork);
    status = cusolverDnSormqr(*cusolverH, side, trans, m, n, k, &A_f, lda, &TAU_f, &B_f, ldb, &workspace_f, Lwork, &devInfo);
    cusolverDnSormqr(*cusolverH, side, trans, m, n, k, &A_f, lda, &TAU_f, &B_f, ldb, &workspace_f, Lwork, &devInfo);

    status = cusolverDnDormqr_bufferSize(*cusolverH, side, trans, m, n, k, &A_d, lda, &TAU_d, &C_d, ldc, &Lwork);
    cusolverDnDormqr_bufferSize(*cusolverH, side, trans, m, n, k, &A_d, lda, &TAU_d, &C_d, ldc, &Lwork);
    status = cusolverDnDormqr(*cusolverH, side, trans, m, n, k, &A_d, lda, &TAU_d, &B_d, ldb, &workspace_d, Lwork, &devInfo);
    cusolverDnDormqr(*cusolverH, side, trans, m, n, k, &A_d, lda, &TAU_d, &B_d, ldb, &workspace_d, Lwork, &devInfo);


    status = cusolverDnCunmqr_bufferSize(*cusolverH, side, trans, m, n, k, &A_c, lda, &TAU_c, &C_c, ldc, &Lwork);
    cusolverDnCunmqr_bufferSize(*cusolverH, side, trans, m, n, k, &A_c, lda, &TAU_c, &C_c, ldc, &Lwork);
    status = cusolverDnCunmqr(*cusolverH, side, trans, m, n, k, &A_c, lda, &TAU_c, &B_c, ldb, &workspace_c, Lwork, &devInfo);
    cusolverDnCunmqr(*cusolverH, side, trans, m, n, k, &A_c, lda, &TAU_c, &B_c, ldb, &workspace_c, Lwork, &devInfo);

    status = cusolverDnZunmqr_bufferSize(*cusolverH, side, trans, m, n, k, &A_z, lda, &TAU_z, &C_z, ldc, &Lwork);
    cusolverDnZunmqr_bufferSize(*cusolverH, side, trans, m, n, k, &A_z, lda, &TAU_z, &C_z, ldc, &Lwork);
    status = cusolverDnZunmqr(*cusolverH, side, trans, m, n, k, &A_z, lda, &TAU_z, &B_z, ldb, &workspace_z, Lwork, &devInfo);
    cusolverDnZunmqr(*cusolverH, side, trans, m, n, k, &A_z, lda, &TAU_z, &B_z, ldb, &workspace_z, Lwork, &devInfo);

    status = cusolverDnSorgqr_bufferSize(*cusolverH, m, n, k, &A_f, lda, &TAU_f, &Lwork);
    cusolverDnSorgqr_bufferSize(*cusolverH, m, n, k, &A_f, lda, &TAU_f, &Lwork);
    status = cusolverDnSorgqr(*cusolverH, m, n, k, &A_f, lda, &TAU_f, &workspace_f, Lwork, &devInfo);
    cusolverDnSorgqr(*cusolverH, m, n, k, &A_f, lda, &TAU_f, &workspace_f, Lwork, &devInfo);

    status = cusolverDnDorgqr_bufferSize(*cusolverH, m, n, k, &A_d, lda, &TAU_d, &Lwork);
    cusolverDnDorgqr_bufferSize(*cusolverH, m, n, k, &A_d, lda, &TAU_d, &Lwork);
    status = cusolverDnDorgqr(*cusolverH, m, n, k, &A_d, lda, &TAU_d, &workspace_d, Lwork, &devInfo);
    cusolverDnDorgqr(*cusolverH, m, n, k, &A_d, lda, &TAU_d, &workspace_d, Lwork, &devInfo);

    status = cusolverDnCungqr_bufferSize(*cusolverH, m, n, k, &A_c, lda, &TAU_c, &Lwork);
    cusolverDnCungqr_bufferSize(*cusolverH, m, n, k, &A_c, lda, &TAU_c, &Lwork);
    status = cusolverDnCungqr(*cusolverH, m, n, k, &A_c, lda, &TAU_c, &workspace_c, Lwork, &devInfo);
    cusolverDnCungqr(*cusolverH, m, n, k, &A_c, lda, &TAU_c, &workspace_c, Lwork, &devInfo);

    status = cusolverDnZungqr_bufferSize(*cusolverH, m, n, k, &A_z, lda, &TAU_z, &Lwork);
    cusolverDnZungqr_bufferSize(*cusolverH, m, n, k, &A_z, lda, &TAU_z, &Lwork);
    status = cusolverDnZungqr(*cusolverH, m, n, k, &A_z, lda, &TAU_z, &workspace_z, Lwork, &devInfo);
    cusolverDnZungqr(*cusolverH, m, n, k, &A_z, lda, &TAU_z, &workspace_z, Lwork, &devInfo);

    status = cusolverDnSsytrf_bufferSize(*cusolverH, n, &A_f, lda, &Lwork);
    cusolverDnSsytrf_bufferSize(*cusolverH, n, &A_f, lda, &Lwork);
    status = cusolverDnSsytrf(*cusolverH, uplo, n, &A_f, lda, &devIpiv, &workspace_f, Lwork, &devInfo);
    cusolverDnSsytrf(*cusolverH, uplo, n, &A_f, lda, &devIpiv, &workspace_f, Lwork, &devInfo);

    status = cusolverDnDsytrf_bufferSize(*cusolverH, n, &A_d, lda, &Lwork);
    cusolverDnDsytrf_bufferSize(*cusolverH, n, &A_d, lda, &Lwork);
    status = cusolverDnDsytrf(*cusolverH, uplo, n, &A_d, lda, &devIpiv, &workspace_d, Lwork, &devInfo);
    cusolverDnDsytrf(*cusolverH, uplo, n, &A_d, lda, &devIpiv, &workspace_d, Lwork, &devInfo);

    status = cusolverDnCsytrf_bufferSize(*cusolverH, n, &A_c, lda, &Lwork);
    cusolverDnCsytrf_bufferSize(*cusolverH, n, &A_c, lda, &Lwork);
    status = cusolverDnCsytrf(*cusolverH, uplo, n, &A_c, lda, &devIpiv, &workspace_c, Lwork, &devInfo);
    cusolverDnCsytrf(*cusolverH, uplo, n, &A_c, lda, &devIpiv, &workspace_c, Lwork, &devInfo);

    status = cusolverDnZsytrf_bufferSize(*cusolverH, n, &A_z, lda, &Lwork);
    cusolverDnZsytrf_bufferSize(*cusolverH, n, &A_z, lda, &Lwork);
    status = cusolverDnZsytrf(*cusolverH, uplo, n, &A_z, lda, &devIpiv, &workspace_z, Lwork, &devInfo);
    cusolverDnZsytrf(*cusolverH, uplo, n, &A_z, lda, &devIpiv, &workspace_z, Lwork, &devInfo);
}
