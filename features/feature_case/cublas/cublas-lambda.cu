// ====------ cublas-lambda.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <cstdio>
#include "cublas_v2.h"
#include <cuda_runtime.h>

cublasStatus_t status;
cublasHandle_t handle;
int N = 275;
float *d_A_S = 0;
float *d_B_S = 0;
float *d_C_S = 0;
float alpha_S = 1.0f;
float beta_S = 0.0f;

int main() {
  cublasStatus_t status;
  cublasHandle_t handle;
  cublasCreate(&handle);

  if (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N)) {
  }
  else if (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N)) {
  }


  if(int stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N)){
  }


  for(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);;){
  }

  for(;cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);){
  }

  while(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N)!=0){
  }



  do{
  }while(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N));


  switch (int stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N)){
  }


  return 0;
}

int foo() {
  return cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
}

void foo2() {
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
}