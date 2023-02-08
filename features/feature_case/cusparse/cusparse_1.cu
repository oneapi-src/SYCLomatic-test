// ===------- cusparse_1.cu -------------------------------- *- CUDA -* ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include "cusparse.h"
#include <cstdio>

bool test1() {
  cusparseMatDescr_t descr;
  cusparseMatrixType_t mt;
  cusparseDiagType_t dt;
  cusparseFillMode_t fm;
  cusparseIndexBase_t ib;

  cusparseCreateMatDescr(&descr);

  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  mt = cusparseGetMatType(descr);
  if (mt != CUSPARSE_MATRIX_TYPE_GENERAL)
    return false;

  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
  mt = cusparseGetMatType(descr);
  if (mt != CUSPARSE_MATRIX_TYPE_SYMMETRIC)
    return false;

  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_HERMITIAN);
  mt = cusparseGetMatType(descr);
  if (mt != CUSPARSE_MATRIX_TYPE_HERMITIAN)
    return false;

  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_TRIANGULAR);
  mt = cusparseGetMatType(descr);
  if (mt != CUSPARSE_MATRIX_TYPE_TRIANGULAR)
    return false;

  cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_NON_UNIT);
  dt = cusparseGetMatDiagType(descr);
  if (dt != CUSPARSE_DIAG_TYPE_NON_UNIT)
    return false;

  cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_UNIT);
  dt = cusparseGetMatDiagType(descr);
  if (dt != CUSPARSE_DIAG_TYPE_UNIT)
    return false;

  cusparseSetMatFillMode(descr, CUSPARSE_FILL_MODE_LOWER);
  fm = cusparseGetMatFillMode(descr);
  if (fm != CUSPARSE_FILL_MODE_LOWER)
    return false;

  cusparseSetMatFillMode(descr, CUSPARSE_FILL_MODE_UPPER);
  fm = cusparseGetMatFillMode(descr);
  if (fm != CUSPARSE_FILL_MODE_UPPER)
    return false;

  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
  ib = cusparseGetMatIndexBase(descr);
  if (ib != CUSPARSE_INDEX_BASE_ZERO)
    return false;

  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ONE);
  ib = cusparseGetMatIndexBase(descr);
  if (ib != CUSPARSE_INDEX_BASE_ONE)
    return false;

  cusparseDestroyMatDescr(descr);
  return true;
}

int main() {
  bool res = true;

  if ((res = test1())) {
    printf("test1 passed\n");
  } else {
    printf("test1 failed\n");
  }

  return 0;
}
