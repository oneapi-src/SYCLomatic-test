// ====------ cusolverDnEi.cu---------- *- CUDA -* ----===////
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
    cusolverEigMode_t jobz;

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

    signed char jobu;
    signed char jobvt;

    float S_f = 0;
    double S_d = 0.0;
    cuComplex S_c = make_cuComplex(1,0);
    cuDoubleComplex S_z = make_cuDoubleComplex(1,0);

    float U_f = 0;
    double U_d = 0.0;
    cuComplex U_c = make_cuComplex(1,0);
    cuDoubleComplex U_z = make_cuDoubleComplex(1,0);
    int ldu;

    float VT_f = 0;
    double VT_d = 0.0;
    cuComplex VT_c = make_cuComplex(1,0);
    cuDoubleComplex VT_z = make_cuDoubleComplex(1,0);
    int ldvt;

    float Rwork_f = 0;
    double Rwork_d = 0.0;
    cuComplex Rwork_c = make_cuComplex(1,0);
    cuDoubleComplex Rwork_z = make_cuDoubleComplex(1,0);

    float W_f = 0;
    double W_d = 0.0;
    cuComplex W_c = make_cuComplex(1,0);
    cuDoubleComplex W_z = make_cuDoubleComplex(1,0);

    status = cusolverDnSgebrd_bufferSize(*cusolverH, m, n, &Lwork);
    cusolverDnSgebrd_bufferSize(*cusolverH, m, n, &Lwork);
    status = cusolverDnSgebrd(*cusolverH, m, n, &A_f, lda, &D_f, &E_f, &TAUQ_f, &TAUP_f, &workspace_f, Lwork, &devInfo);
    cusolverDnSgebrd(*cusolverH, m, n, &A_f, lda, &D_f, &E_f, &TAUQ_f, &TAUP_f, &workspace_f, Lwork, &devInfo);

    status = cusolverDnDgebrd_bufferSize(*cusolverH, m, n, &Lwork);
    cusolverDnDgebrd_bufferSize(*cusolverH, m, n, &Lwork);
    status = cusolverDnDgebrd(*cusolverH, m, n, &A_d, lda, &D_d, &E_d, &TAUQ_d, &TAUP_d, &workspace_d, Lwork, &devInfo);
    cusolverDnDgebrd(*cusolverH, m, n, &A_d, lda, &D_d, &E_d, &TAUQ_d, &TAUP_d, &workspace_d, Lwork, &devInfo);

    status = cusolverDnCgebrd_bufferSize(*cusolverH, m, n, &Lwork);
    cusolverDnCgebrd_bufferSize(*cusolverH, m, n, &Lwork);
    status = cusolverDnCgebrd(*cusolverH, m, n, &A_c, lda, &D_f, &E_f, &TAUQ_c, &TAUP_c, &workspace_c, Lwork, &devInfo);
    cusolverDnCgebrd(*cusolverH, m, n, &A_c, lda, &D_f, &E_f, &TAUQ_c, &TAUP_c, &workspace_c, Lwork, &devInfo);

    status = cusolverDnZgebrd_bufferSize(*cusolverH, m, n, &Lwork);
    cusolverDnZgebrd_bufferSize(*cusolverH, m, n, &Lwork);
    status = cusolverDnZgebrd(*cusolverH, m, n, &A_z, lda, &D_d, &E_d, &TAUQ_z, &TAUP_z, &workspace_z, Lwork, &devInfo);
    cusolverDnZgebrd(*cusolverH, m, n, &A_z, lda, &D_d, &E_d, &TAUQ_z, &TAUP_z, &workspace_z, Lwork, &devInfo);


    status = cusolverDnSorgbr_bufferSize(*cusolverH, side, m, n, k, &A_f, lda, &TAU_f, &Lwork);
    cusolverDnSorgbr_bufferSize(*cusolverH, side, m, n, k, &A_f, lda, &TAU_f, &Lwork);
    status = cusolverDnSorgbr(*cusolverH, side, m, n, k, &A_f, lda, &TAU_f, &workspace_f, Lwork, &devInfo);
    cusolverDnSorgbr(*cusolverH, side, m, n, k, &A_f, lda, &TAU_f, &workspace_f, Lwork, &devInfo);


    status = cusolverDnDorgbr_bufferSize(*cusolverH, side, m, n, k, &A_d, lda, &TAU_d, &Lwork);
    cusolverDnDorgbr_bufferSize(*cusolverH, side, m, n, k, &A_d, lda, &TAU_d, &Lwork);
    status = cusolverDnDorgbr(*cusolverH, side, m, n, k, &A_d, lda, &TAU_d, &workspace_d, Lwork, &devInfo);
    cusolverDnDorgbr(*cusolverH, side, m, n, k, &A_d, lda, &TAU_d, &workspace_d, Lwork, &devInfo);

    status = cusolverDnCungbr_bufferSize(*cusolverH, side, m, n, k, &A_c, lda, &TAU_c, &Lwork);
    cusolverDnCungbr_bufferSize(*cusolverH, side, m, n, k, &A_c, lda, &TAU_c, &Lwork);
    status = cusolverDnCungbr(*cusolverH, side, m, n, k, &A_c, lda, &TAU_c, &workspace_c, Lwork, &devInfo);
    cusolverDnCungbr(*cusolverH, side, m, n, k, &A_c, lda, &TAU_c, &workspace_c, Lwork, &devInfo);

    status = cusolverDnZungbr_bufferSize(*cusolverH, side, m, n, k, &A_z, lda, &TAU_z, &Lwork);
    cusolverDnZungbr_bufferSize(*cusolverH, side, m, n, k, &A_z, lda, &TAU_z, &Lwork);
    status = cusolverDnZungbr(*cusolverH, side, m, n, k, &A_z, lda, &TAU_z, &workspace_z, Lwork, &devInfo);
    cusolverDnZungbr(*cusolverH, side, m, n, k, &A_z, lda, &TAU_z, &workspace_z, Lwork, &devInfo);


    status = cusolverDnSsytrd_bufferSize(*cusolverH, uplo, n, &A_f, lda, &D_f, &E_f, &TAU_f, &Lwork);
    cusolverDnSsytrd_bufferSize(*cusolverH, uplo, n, &A_f, lda, &D_f, &E_f, &TAU_f, &Lwork);
    status = cusolverDnSsytrd(*cusolverH, uplo, n, &A_f, lda, &D_f, &E_f, &TAU_f, &workspace_f, Lwork, &devInfo);
    cusolverDnSsytrd(*cusolverH, uplo, n, &A_f, lda, &D_f, &E_f, &TAU_f, &workspace_f, Lwork, &devInfo);

    status = cusolverDnDsytrd_bufferSize(*cusolverH, uplo, n, &A_d, lda, &D_d, &E_d, &TAU_d, &Lwork);
    cusolverDnDsytrd_bufferSize(*cusolverH, uplo, n, &A_d, lda, &D_d, &E_d, &TAU_d, &Lwork);
    status = cusolverDnDsytrd(*cusolverH, uplo, n, &A_d, lda, &D_d, &E_d, &TAU_d, &workspace_d, Lwork, &devInfo);
    cusolverDnDsytrd(*cusolverH, uplo, n, &A_d, lda, &D_d, &E_d, &TAU_d, &workspace_d, Lwork, &devInfo);

    status = cusolverDnChetrd_bufferSize(*cusolverH, uplo, n, &A_c, lda, &D_f, &E_f, &TAU_c, &Lwork);
    cusolverDnChetrd_bufferSize(*cusolverH, uplo, n, &A_c, lda, &D_f, &E_f, &TAU_c, &Lwork);
    status = cusolverDnChetrd(*cusolverH, uplo, n, &A_c, lda, &D_f, &E_f, &TAU_c, &workspace_c, Lwork, &devInfo);
    cusolverDnChetrd(*cusolverH, uplo, n, &A_c, lda, &D_f, &E_f, &TAU_c, &workspace_c, Lwork, &devInfo);

    status = cusolverDnZhetrd_bufferSize(*cusolverH, uplo, n, &A_z, lda, &D_d, &E_d, &TAU_z, &Lwork);
    cusolverDnZhetrd_bufferSize(*cusolverH, uplo, n, &A_z, lda, &D_d, &E_d, &TAU_z, &Lwork);
    status = cusolverDnZhetrd(*cusolverH, uplo, n, &A_z, lda, &D_d, &E_d, &TAU_z, &workspace_z, Lwork, &devInfo);
    cusolverDnZhetrd(*cusolverH, uplo, n, &A_z, lda, &D_d, &E_d, &TAU_z, &workspace_z, Lwork, &devInfo);

    status = cusolverDnSormtr_bufferSize(*cusolverH, side, uplo, trans, m, n, &A_f, lda, &TAU_f, &B_f, ldb, &Lwork);
    cusolverDnSormtr_bufferSize(*cusolverH, side, uplo, trans, m, n, &A_f, lda, &TAU_f, &B_f, ldb, &Lwork);
    status = cusolverDnSormtr(*cusolverH, side, uplo, trans, m, n, &A_f, lda, &TAU_f, &B_f, ldb, &workspace_f, Lwork, &devInfo);
    cusolverDnSormtr(*cusolverH, side, uplo, trans, m, n, &A_f, lda, &TAU_f, &B_f, ldb, &workspace_f, Lwork, &devInfo);

    status = cusolverDnDormtr_bufferSize(*cusolverH, side, uplo, trans, m, n, &A_d, lda, &TAU_d, &B_d, ldb, &Lwork);
    cusolverDnDormtr_bufferSize(*cusolverH, side, uplo, trans, m, n, &A_d, lda, &TAU_d, &B_d, ldb, &Lwork);
    status = cusolverDnDormtr(*cusolverH, side, uplo, trans, m, n, &A_d, lda, &TAU_d, &B_d, ldb, &workspace_d, Lwork, &devInfo);
    cusolverDnDormtr(*cusolverH, side, uplo, trans, m, n, &A_d, lda, &TAU_d, &B_d, ldb, &workspace_d, Lwork, &devInfo);

    status = cusolverDnCunmtr_bufferSize(*cusolverH, side, uplo, trans, m, n, &A_c, lda, &TAU_c, &B_c, ldb, &Lwork);
    cusolverDnCunmtr_bufferSize(*cusolverH, side, uplo, trans, m, n, &A_c, lda, &TAU_c, &B_c, ldb, &Lwork);
    status = cusolverDnCunmtr(*cusolverH, side, uplo, trans, m, n, &A_c, lda, &TAU_c, &B_c, ldb, &workspace_c, Lwork, &devInfo);
    cusolverDnCunmtr(*cusolverH, side, uplo, trans, m, n, &A_c, lda, &TAU_c, &B_c, ldb, &workspace_c, Lwork, &devInfo);

    status = cusolverDnZunmtr_bufferSize(*cusolverH, side, uplo, trans, m, n, &A_z, lda, &TAU_z, &B_z, ldb, &Lwork);
    cusolverDnZunmtr_bufferSize(*cusolverH, side, uplo, trans, m, n, &A_z, lda, &TAU_z, &B_z, ldb, &Lwork);
    status = cusolverDnZunmtr(*cusolverH, side, uplo, trans, m, n, &A_z, lda, &TAU_z, &B_z, ldb, &workspace_z, Lwork, &devInfo);
    cusolverDnZunmtr(*cusolverH, side, uplo, trans, m, n, &A_z, lda, &TAU_z, &B_z, ldb, &workspace_z, Lwork, &devInfo);

    status = cusolverDnSorgtr_bufferSize(*cusolverH, uplo, n, &A_f, lda, &TAU_f, &Lwork);
    cusolverDnSorgtr_bufferSize(*cusolverH, uplo, n, &A_f, lda, &TAU_f, &Lwork);
    status = cusolverDnSorgtr(*cusolverH, uplo, n, &A_f, lda, &TAU_f, &workspace_f, Lwork, &devInfo);
    cusolverDnSorgtr(*cusolverH, uplo, n, &A_f, lda, &TAU_f, &workspace_f, Lwork, &devInfo);

    status = cusolverDnDorgtr_bufferSize(*cusolverH, uplo, n, &A_d, lda, &TAU_d, &Lwork);
    cusolverDnDorgtr_bufferSize(*cusolverH, uplo, n, &A_d, lda, &TAU_d, &Lwork);
    status = cusolverDnDorgtr(*cusolverH, uplo, n, &A_d, lda, &TAU_d, &workspace_d, Lwork, &devInfo);
    cusolverDnDorgtr(*cusolverH, uplo, n, &A_d, lda, &TAU_d, &workspace_d, Lwork, &devInfo);

    status = cusolverDnCungtr_bufferSize(*cusolverH, uplo, n, &A_c, lda, &TAU_c, &Lwork);
    cusolverDnCungtr_bufferSize(*cusolverH, uplo, n, &A_c, lda, &TAU_c, &Lwork);
    status = cusolverDnCungtr(*cusolverH, uplo, n, &A_c, lda, &TAU_c, &workspace_c, Lwork, &devInfo);
    cusolverDnCungtr(*cusolverH, uplo, n, &A_c, lda, &TAU_c, &workspace_c, Lwork, &devInfo);

    status = cusolverDnZungtr_bufferSize(*cusolverH, uplo, n, &A_z, lda, &TAU_z, &Lwork);
    cusolverDnZungtr_bufferSize(*cusolverH, uplo, n, &A_z, lda, &TAU_z, &Lwork);
    status = cusolverDnZungtr(*cusolverH, uplo, n, &A_z, lda, &TAU_z, &workspace_z, Lwork, &devInfo);
    cusolverDnZungtr(*cusolverH, uplo, n, &A_z, lda, &TAU_z, &workspace_z, Lwork, &devInfo);


    status = cusolverDnSgesvd_bufferSize(*cusolverH, m, n, &Lwork);
    cusolverDnSgesvd_bufferSize(*cusolverH, m, n, &Lwork);
    status = cusolverDnSgesvd (*cusolverH, jobu, jobvt, m, n, &A_f, lda, &S_f, &U_f, ldu, &VT_f, ldvt, &workspace_f, Lwork, &Rwork_f, &devInfo);
    cusolverDnSgesvd (*cusolverH, jobu, jobvt, m, n, &A_f, lda, &S_f, &U_f, ldu, &VT_f, ldvt, &workspace_f, Lwork, &Rwork_f, &devInfo);

    status = cusolverDnDgesvd_bufferSize(*cusolverH, m, n, &Lwork);
    cusolverDnDgesvd_bufferSize(*cusolverH, m, n, &Lwork);
    status = cusolverDnDgesvd (*cusolverH, jobu, jobvt, m, n, &A_d, lda, &S_d, &U_d, ldu, &VT_d, ldvt, &workspace_d, Lwork, &Rwork_d, &devInfo);
    cusolverDnDgesvd (*cusolverH, jobu, jobvt, m, n, &A_d, lda, &S_d, &U_d, ldu, &VT_d, ldvt, &workspace_d, Lwork, &Rwork_d, &devInfo);

    status = cusolverDnCgesvd_bufferSize(*cusolverH, m, n, &Lwork);
    cusolverDnCgesvd_bufferSize(*cusolverH, m, n, &Lwork);
    status = cusolverDnCgesvd (*cusolverH, jobu, jobvt, m, n, &A_c, lda, &S_f, &U_c, ldu, &VT_c, ldvt, &workspace_c, Lwork, &Rwork_f, &devInfo);
    cusolverDnCgesvd (*cusolverH, jobu, jobvt, m, n, &A_c, lda, &S_f, &U_c, ldu, &VT_c, ldvt, &workspace_c, Lwork, &Rwork_f, &devInfo);

    status = cusolverDnZgesvd_bufferSize(*cusolverH, m, n, &Lwork);
    cusolverDnZgesvd_bufferSize(*cusolverH, m, n, &Lwork);
    status = cusolverDnZgesvd (*cusolverH, jobu, jobvt, m, n, &A_z, lda, &S_d, &U_z, ldu, &VT_z, ldvt, &workspace_z, Lwork, &Rwork_d, &devInfo);
    cusolverDnZgesvd (*cusolverH, jobu, jobvt, m, n, &A_z, lda, &S_d, &U_z, ldu, &VT_z, ldvt, &workspace_z, Lwork, &Rwork_d, &devInfo);

}
