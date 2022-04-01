// ====------ cufft-type.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <cstdio>
#include <cufft.h>
#include <cuda_runtime.h>

size_t size;

int main() {
  cufftReal fftreal;
  cufftDoubleReal fftdreal;
  cufftComplex fftcomplex;
  cufftDoubleComplex fftdcomplex;
  cuComplex ccomplex;
  cuDoubleComplex cdcomplex;
  size = sizeof(cufftReal);
  size = sizeof(cufftDoubleReal);
  size = sizeof(cufftComplex);
  size = sizeof(cufftDoubleComplex);
  size = sizeof(cuComplex);
  size = sizeof(cuDoubleComplex);

  int forward = CUFFT_FORWARD;
  int inverse = CUFFT_INVERSE;

  cufftType_t fftt_t;
  cufftType fftt;
  size = sizeof(cufftType_t);
  size = sizeof(cufftType);
  fftt = CUFFT_R2C;
  fftt = CUFFT_C2R;
  fftt = CUFFT_C2C;
  fftt = CUFFT_D2Z;
  fftt = CUFFT_Z2D;
  fftt = CUFFT_Z2Z;

  cufftResult_t fftr_t;
  cufftResult fftr;
  size = sizeof(cufftResult_t);
  size = sizeof(cufftResult);
  fftr = CUFFT_SUCCESS;
  fftr = CUFFT_INVALID_PLAN;
  fftr = CUFFT_ALLOC_FAILED;
  fftr = CUFFT_INVALID_TYPE;
  fftr = CUFFT_INVALID_VALUE;
  fftr = CUFFT_INTERNAL_ERROR;
  fftr = CUFFT_EXEC_FAILED;
  fftr = CUFFT_SETUP_FAILED;
  fftr = CUFFT_INVALID_SIZE;
  fftr = CUFFT_UNALIGNED_DATA;
  fftr = CUFFT_INCOMPLETE_PARAMETER_LIST;
  fftr = CUFFT_INVALID_DEVICE;
  fftr = CUFFT_PARSE_ERROR;
  fftr = CUFFT_NO_WORKSPACE;
  fftr = CUFFT_NOT_IMPLEMENTED;
  fftr = CUFFT_LICENSE_ERROR;
  fftr = CUFFT_NOT_SUPPORTED;

  return 0;
}


template<
typename A = cufftReal,
typename B = cufftDoubleReal,
typename C = cufftComplex,
typename D = cufftDoubleComplex,
typename E = cuComplex,
typename F = cuDoubleComplex,
typename G = cufftType_t,
typename H = cufftType,
typename J = cufftResult_t,
typename K = cufftResult>
void foo1(
cufftReal a,
cufftDoubleReal b,
cufftComplex c,
cufftDoubleComplex d,
cuComplex e,
cuDoubleComplex f,
cufftType_t g,
cufftType h,
cufftResult_t j,
cufftResult k
){}


template<
cufftType A1 = CUFFT_R2C,
cufftType A2 = CUFFT_C2R,
cufftType A3 = CUFFT_C2C,
cufftType A4 = CUFFT_D2Z,
cufftType A5 = CUFFT_Z2D,
cufftType A6 = CUFFT_Z2Z,
cufftResult B1 = CUFFT_SUCCESS,
cufftResult B2 = CUFFT_INVALID_PLAN,
cufftResult B3 = CUFFT_ALLOC_FAILED,
cufftResult B4 = CUFFT_INVALID_TYPE,
cufftResult B5 = CUFFT_INVALID_VALUE,
cufftResult B6 = CUFFT_INTERNAL_ERROR,
cufftResult B7 = CUFFT_EXEC_FAILED,
cufftResult B8 = CUFFT_SETUP_FAILED,
cufftResult B9 = CUFFT_INVALID_SIZE,
cufftResult B10 = CUFFT_UNALIGNED_DATA,
cufftResult B11 = CUFFT_INCOMPLETE_PARAMETER_LIST,
cufftResult B12 = CUFFT_INVALID_DEVICE,
cufftResult B13 = CUFFT_PARSE_ERROR,
cufftResult B14 = CUFFT_NO_WORKSPACE,
cufftResult B15 = CUFFT_NOT_IMPLEMENTED,
cufftResult B16 = CUFFT_LICENSE_ERROR,
cufftResult B17 = CUFFT_NOT_SUPPORTED>
void foo2(
cufftType a1 = CUFFT_R2C,
cufftType a2 = CUFFT_C2R,
cufftType a3 = CUFFT_C2C,
cufftType a4 = CUFFT_D2Z,
cufftType a5 = CUFFT_Z2D,
cufftType a6 = CUFFT_Z2Z,
cufftResult b1 = CUFFT_SUCCESS,
cufftResult b2 = CUFFT_INVALID_PLAN,
cufftResult b3 = CUFFT_ALLOC_FAILED,
cufftResult b4 = CUFFT_INVALID_TYPE,
cufftResult b5 = CUFFT_INVALID_VALUE,
cufftResult b6 = CUFFT_INTERNAL_ERROR,
cufftResult b7 = CUFFT_EXEC_FAILED,
cufftResult b8 = CUFFT_SETUP_FAILED,
cufftResult b9 = CUFFT_INVALID_SIZE,
cufftResult b10 = CUFFT_UNALIGNED_DATA,
cufftResult b11 = CUFFT_INCOMPLETE_PARAMETER_LIST,
cufftResult b12 = CUFFT_INVALID_DEVICE,
cufftResult b13 = CUFFT_PARSE_ERROR,
cufftResult b14 = CUFFT_NO_WORKSPACE,
cufftResult b15 = CUFFT_NOT_IMPLEMENTED,
cufftResult b16 = CUFFT_LICENSE_ERROR,
cufftResult b17 = CUFFT_NOT_SUPPORTED
){}


template<typename T>
cufftReal foo3(){}

template<typename T>
cufftDoubleReal foo4(){}

template<typename T>
cufftComplex foo5(){}

template<typename T>
cufftDoubleComplex foo6(){}

template<typename T>
cuComplex foo7(){}

template<typename T>
cuDoubleComplex foo8(){}

template<typename T>
cufftType_t foo9(){}

template<typename T>
cufftType foo10(){}


template<typename T>
cufftResult_t foo12(){}

template<typename T>
cufftResult foo13(){}
