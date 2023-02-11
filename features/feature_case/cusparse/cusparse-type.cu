// ====------ cusparse-type.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <cstdio>
#include <cusparse_v2.h>
#include <cuda_runtime.h>

int main(){
  cusparseFillMode_t a1;
  a1 = CUSPARSE_FILL_MODE_LOWER;
  a1 = CUSPARSE_FILL_MODE_UPPER;

  cusparseDiagType_t a2;
  a2 = CUSPARSE_DIAG_TYPE_NON_UNIT;
  a2 = CUSPARSE_DIAG_TYPE_UNIT;

  cusparseIndexBase_t a3;
  a3 = CUSPARSE_INDEX_BASE_ZERO;
  a3 = CUSPARSE_INDEX_BASE_ONE;

  cusparseMatrixType_t a4;
  a4 = CUSPARSE_MATRIX_TYPE_GENERAL;
  a4 = CUSPARSE_MATRIX_TYPE_SYMMETRIC;
  a4 = CUSPARSE_MATRIX_TYPE_HERMITIAN;
  a4 = CUSPARSE_MATRIX_TYPE_TRIANGULAR;

  cusparseOperation_t a5;
  a5 = CUSPARSE_OPERATION_NON_TRANSPOSE;
  a5 = CUSPARSE_OPERATION_TRANSPOSE;
  a5 = CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;

  cusparseStatus_t a6;
  a6 = CUSPARSE_STATUS_SUCCESS;
  a6 = CUSPARSE_STATUS_NOT_INITIALIZED;
  a6 = CUSPARSE_STATUS_ALLOC_FAILED;
  a6 = CUSPARSE_STATUS_INVALID_VALUE;
  a6 = CUSPARSE_STATUS_ARCH_MISMATCH;
  a6 = CUSPARSE_STATUS_MAPPING_ERROR;
  a6 = CUSPARSE_STATUS_EXECUTION_FAILED;
  a6 = CUSPARSE_STATUS_INTERNAL_ERROR;
  a6 = CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
  a6 = CUSPARSE_STATUS_ZERO_PIVOT;

  cusparseMatDescr_t a7;

  cusparseHandle_t a8;
}

void foo(cusparseFillMode_t a1,
         cusparseDiagType_t a2,
         cusparseIndexBase_t a3,
         cusparseMatrixType_t a4,
         cusparseOperation_t a5,
         cusparseStatus_t a6,
         cusparseMatDescr_t a7,
         cusparseHandle_t a8);

cusparseFillMode_t foo1();
cusparseDiagType_t foo2();
cusparseIndexBase_t foo3();
cusparseMatrixType_t foo4();
cusparseOperation_t foo5();
cusparseStatus_t foo6();
cusparseMatDescr_t foo7();
cusparseHandle_t foo8();

template<typename T>
void bar1(cusparseFillMode_t a1,
         cusparseDiagType_t a2,
         cusparseIndexBase_t a3,
         cusparseMatrixType_t a4,
         cusparseOperation_t a5,
         cusparseStatus_t a6,
         cusparseMatDescr_t a7,
         cusparseHandle_t a8){}

template<typename T>
void bar2(cusparseFillMode_t a1,
         cusparseDiagType_t a2,
         cusparseIndexBase_t a3,
         cusparseMatrixType_t a4,
         cusparseOperation_t a5,
         cusparseStatus_t a6,
         cusparseMatDescr_t a7,
         cusparseHandle_t a8){}

// specialization
template<>
void bar2<double>(cusparseFillMode_t a1,
                  cusparseDiagType_t a2,
                  cusparseIndexBase_t a3,
                  cusparseMatrixType_t a4,
                  cusparseOperation_t a5,
                  cusparseStatus_t a6,
                  cusparseMatDescr_t a7,
                  cusparseHandle_t a8){}

