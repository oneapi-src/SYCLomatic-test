//===--- complex.cu -------------------------------*- CUDA -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------==//

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <iostream>

#define COMPLEX_D_MAKE(r,i) make_cuDoubleComplex(r, i)
#define COMPLEX_D_REAL(a) (a).x
#define COMPLEX_D_IMAG(a) (a).y
#define COMPLEX_D_FREAL(a) cuCreal(a)
#define COMPLEX_D_FIMAG(a) cuCimag(a)
#define COMPLEX_D_ADD(a, b) cuCadd(a, b)
#define COMPLEX_D_SUB(a, b) cuCsub(a, b)
#define COMPLEX_D_MUL(a, b) cuCmul(a, b)
#define COMPLEX_D_DIV(a, b) cuCdiv(a, b)
#define COMPLEX_D_ABS(a) cuCabs(a)
#define COMPLEX_D_ABS1(a) (fabs((a).x) + fabs((a).y))
#define COMPLEX_D_CONJ(a) cuConj(a)

#define COMPLEX_F_MAKE(r,i) make_cuFloatComplex(r, i)
#define COMPLEX_F_REAL(a) (a).x
#define COMPLEX_F_IMAG(a) (a).y
#define COMPLEX_F_FREAL(a) cuCrealf(a)
#define COMPLEX_F_FIMAG(a) cuCimagf(a)
#define COMPLEX_F_ADD(a, b) cuCaddf(a, b)
#define COMPLEX_F_SUB(a, b) cuCsubf(a, b)
#define COMPLEX_F_MUL(a, b) cuCmulf(a, b)
#define COMPLEX_F_DIV(a, b) cuCdivf(a, b)
#define COMPLEX_F_ABS(a) cuCabsf(a)
#define COMPLEX_F_ABS1(a) (fabsf((a).x) + fabsf((a).y))
#define COMPLEX_F_CONJ(a) cuConjf(a)

template <typename T>
__host__ __device__ bool check(T x, float e[], int& index) {
    float precison = 0.001f;
    if((std::abs(x.x - e[index++]) < precison) && (std::abs(x.y - e[index++]) < precison)) {
        return true;
    }
    return false;
}

template <>
__host__ __device__ bool check<float>(float x, float e[], int& index) {
  float precison = 0.001f;
  if(std::abs(x - e[index++]) < precison) {
      return true;
  }
  return false;
}

template <>
__host__ __device__ bool check<double>(double x, float e[], int& index) {
    float precison = 0.001f;
  if(std::abs(x - e[index++]) < precison) {
      return true;
  }
  return false;
}

__global__ void kernel(int *result) {

    cuFloatComplex f1, f2;
    cuDoubleComplex d1, d2;

    f1 = COMPLEX_F_MAKE(1.8, -2.7);
    f2 = COMPLEX_F_MAKE(-3.6, 4.5);
    d1 = COMPLEX_D_MAKE(5.4, -6.3);
    d2 = COMPLEX_D_MAKE(-7.2, 8.1);

    int index = 0;
    bool r = true;
    float expect[32] = {5.400000,  -6.300000,
        5.400000,  -6.300000, -1.800000, 1.800000, 12.600000, -14.400000, 12.150000,
        89.100000, -0.765517, 0.013793,  8.297590, 11.700000, 5.400000,   6.300000,
        1.800000,  -2.700000, -1.800000, 1.800000, 5.400000,  -7.200000,  5.670001,
        17.820000, -0.560976, 0.048780,  3.244996, 4.500000,  1.800000,   2.700000,
        1.800000,   -2.700000};

    auto a1 = COMPLEX_D_FREAL(d1);
    r = r && check(a1, expect, index);

    auto a2 = COMPLEX_D_FIMAG(d1);
    r = r && check(a2, expect, index);

    auto a3 = COMPLEX_D_REAL(d1);
    r = r && check(a3, expect, index);

    auto a4 = COMPLEX_D_IMAG(d1);
    r = r && check(a4, expect, index);

    auto a5 = COMPLEX_D_ADD(d1, d2);
    r = r && check(a5, expect, index);

    auto a6 = COMPLEX_D_SUB(d1, d2);
    r = r && check(a6, expect, index);

    auto a7 = COMPLEX_D_MUL(d1, d2);
    r = r && check(a7, expect, index);

    auto a8 = COMPLEX_D_DIV(d1, d2);
    r = r && check(a8, expect, index);

    auto a9 = COMPLEX_D_ABS(d1);
    r = r && check(a9, expect, index);

    auto a10 = COMPLEX_D_ABS1(d1);
    r = r && check(a10, expect, index);

    auto a11 = COMPLEX_D_CONJ(d1);
    r = r && check(a11, expect, index);

    auto a13 = COMPLEX_F_REAL(f1);
    r = r && check(a13, expect, index);

    auto a14 = COMPLEX_F_IMAG(f1);
    r = r && check(a14, expect, index);

    auto a15 = COMPLEX_F_ADD(f1, f2);
    r = r && check(a15, expect, index);

    auto a16 = COMPLEX_F_SUB(f1, f2);
    r = r && check(a16, expect, index);

    auto a17 = COMPLEX_F_MUL(f1, f2);
    r = r && check(a17, expect, index);

    auto a18 = COMPLEX_F_DIV(f1, f2);
    r = r && check(a18, expect, index);

    auto a19 = COMPLEX_F_ABS(f1);
    r = r && check(a19, expect, index);

    auto a20 = COMPLEX_F_ABS1(f1);
    r = r && check(a20, expect, index);

    auto a21 = COMPLEX_F_CONJ(f1);
    r = r && check(a21, expect, index);

    auto a22 = COMPLEX_F_FREAL(f1);
    r = r && check(a22, expect, index);

    auto a23 = COMPLEX_F_FIMAG(f1);
    r = r && check(a23, expect, index);
    *result = r;
}

int main() {

    cuFloatComplex f1, f2;
    cuDoubleComplex d1, d2;

    f1 = COMPLEX_F_MAKE(1.8, -2.7);
    f2 = COMPLEX_F_MAKE(-3.6, 4.5);
    d1 = COMPLEX_D_MAKE(5.4, -6.3);
    d2 = COMPLEX_D_MAKE(-7.2, 8.1);
    int index = 0;
    bool r = true;
    float expect[32] = {5.400000,  -6.300000,
        5.400000,  -6.300000, -1.800000, 1.800000, 12.600000, -14.400000, 12.150000,
        89.100000, -0.765517, 0.013793,  8.297590, 11.700000, 5.400000,   6.300000,
        1.800000,  -2.700000, -1.800000, 1.800000, 5.400000,  -7.200000,  5.670001,
        17.820000, -0.560976, 0.048780,  3.244996, 4.500000,  1.800000,   2.700000,
        1.800000,   -2.700000};

    auto a1 = COMPLEX_D_FREAL(d1);
    r = r && check(a1, expect, index);

    auto a2 = COMPLEX_D_FIMAG(d1);
    r = r && check(a2, expect, index);

    auto a3 = COMPLEX_D_REAL(d1);
    r = r && check(a3, expect, index);

    auto a4 = COMPLEX_D_IMAG(d1);
    r = r && check(a4, expect, index);

    auto a5 = COMPLEX_D_ADD(d1, d2);
    r = r && check(a5, expect, index);

    auto a6 = COMPLEX_D_SUB(d1, d2);
    r = r && check(a6, expect, index);

    auto a7 = COMPLEX_D_MUL(d1, d2);
    r = r && check(a7, expect, index);

    auto a8 = COMPLEX_D_DIV(d1, d2);
    r = r && check(a8, expect, index);

    auto a9 = COMPLEX_D_ABS(d1);
    r = r && check(a9, expect, index);

    auto a10 = COMPLEX_D_ABS1(d1);
    r = r && check(a10, expect, index);

    auto a11 = COMPLEX_D_CONJ(d1);
    r = r && check(a11, expect, index);

    auto a13 = COMPLEX_F_REAL(f1);
    r = r && check(a13, expect, index);

    auto a14 = COMPLEX_F_IMAG(f1);
    r = r && check(a14, expect, index);

    auto a15 = COMPLEX_F_ADD(f1, f2);
    r = r && check(a15, expect, index);

    auto a16 = COMPLEX_F_SUB(f1, f2);
    r = r && check(a16, expect, index);

    auto a17 = COMPLEX_F_MUL(f1, f2);
    r = r && check(a17, expect, index);

    auto a18 = COMPLEX_F_DIV(f1, f2);
    r = r && check(a18, expect, index);

    auto a19 = COMPLEX_F_ABS(f1);
    r = r && check(a19, expect, index);

    auto a20 = COMPLEX_F_ABS1(f1);
    r = r && check(a20, expect, index);

    auto a21 = COMPLEX_F_CONJ(f1);
    r = r && check(a21, expect, index);

    auto a22 = COMPLEX_F_FREAL(f1);
    r = r && check(a22, expect, index);

    auto a23 = COMPLEX_F_FIMAG(f1);
    r = r && check(a23, expect, index);
    
    f1 = cuComplexDoubleToFloat(d1);
    
    d1 = cuComplexFloatToDouble(f1);
    
    int *result = nullptr;
    cudaMallocManaged(&result, sizeof(int));
    *result = 0;

    kernel<<<1,1>>>(result);
    cudaDeviceSynchronize();

    if(*result && r) {
      std::cout << "pass" << std::endl;
    } else {
      std::cout << "fail" << std::endl;
      exit(-1);
    }
    return 0;
}
