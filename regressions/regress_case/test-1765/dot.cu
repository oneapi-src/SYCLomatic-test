// ====------ dot.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <cublas_v2.h>
#include <cuda.h>
#include <iostream>

const int num = 1e7;

bool foo1(){
  double *d_A;
  double *d_B;
  double result;
  std::cout << "size " << num * sizeof(double) << " byte" << std::endl;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cudaMalloc(&d_A, num * sizeof(double));
  cudaMalloc(&d_B, num * sizeof(double));
  double *A = new double[num];
  double *B = new double[num];
  for (int i = 0; i < num; i++) {
    A[i] = 1;
    B[i] = 1;
  }
  cudaMemcpy(d_A, A, num * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, num * sizeof(double), cudaMemcpyHostToDevice);
  cublasDdot(handle, num, d_A, 1, d_B, 1, &result);
  cudaFree(d_A);
  cudaFree(d_B);
  cublasDestroy(handle);
  delete[] A;
  delete[] B;
  cudaDeviceSynchronize();
  if (abs(result - 1e7) >= 0.01) {
    std::cout << "foo1() failed" << std::endl;
    return false;
  } else {
    return true;
  }
}

bool foo2(){
  double *d_A;
  double *d_B;
  double *result;
  double result_h;
  std::cout << "size " << num * sizeof(double) << " byte" << std::endl;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cudaMalloc(&d_A, num * sizeof(double));
  cudaMalloc(&d_B, num * sizeof(double));
  cudaMalloc(&result, sizeof(double));
  double *A = new double[num];
  double *B = new double[num];
  for (int i = 0; i < num; i++) {
    A[i] = 1;
    B[i] = 1;
  }
  cudaMemcpy(d_A, A, num * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, num * sizeof(double), cudaMemcpyHostToDevice);
  cublasDdot(handle, num, d_A, 1, d_B, 1, result);
  cudaDeviceSynchronize();
  cudaMemcpy(&result_h, result, sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(result);
  cublasDestroy(handle);
  delete[] A;
  delete[] B;
  cudaDeviceSynchronize();
  if (abs(result_h - 1e7) >= 0.01) {
    std::cout << "foo2() failed" << std::endl;
    return false;
  } else {
    return true;
  }
}

bool foo3(){
  double2 *d_A;
  double2 *d_B;
  double2 result;
  std::cout << "size " << num * sizeof(double2) << " byte" << std::endl;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cudaMalloc(&d_A, num * sizeof(double2));
  cudaMalloc(&d_B, num * sizeof(double2));
  double2 *A = new double2[num];
  double2 *B = new double2[num];
  for (int i = 0; i < num; i++) {
    A[i].x = 1;
    A[i].y = 1;
    B[i].x = 1;
    B[i].y = 1;
  }
  cudaMemcpy(d_A, A, num * sizeof(double2), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, num * sizeof(double2), cudaMemcpyHostToDevice);
  cublasZdotu(handle, num, d_A, 1, d_B, 1, &result);
  cudaFree(d_A);
  cudaFree(d_B);
  cublasDestroy(handle);
  delete[] A;
  delete[] B;
  cudaDeviceSynchronize();
  if ((abs(result.x) >= 0.01) || (abs(result.y - 2e7) >= 0.01)) {
    std::cout << "foo3() failed" << std::endl;
    return false;
  } else {
    return true;
  }
}

bool foo4(){
  double2 *d_A;
  double2 *d_B;
  double2 *result;
  double2 result_h;
  std::cout << "size " << num * sizeof(double2) << " byte" << std::endl;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cudaMalloc(&d_A, num * sizeof(double2));
  cudaMalloc(&d_B, num * sizeof(double2));
  cudaMalloc(&result, sizeof(double2));
  double2 *A = new double2[num];
  double2 *B = new double2[num];
  for (int i = 0; i < num; i++) {
    A[i].x = 1;
    A[i].y = 1;
    B[i].x = 1;
    B[i].y = 1;
  }
  cudaMemcpy(d_A, A, num * sizeof(double2), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, num * sizeof(double2), cudaMemcpyHostToDevice);
  cublasZdotc(handle, num, d_A, 1, d_B, 1, result);
  cudaDeviceSynchronize();
  cudaMemcpy(&result_h, result, sizeof(double2), cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(result);
  cublasDestroy(handle);
  delete[] A;
  delete[] B;
  cudaDeviceSynchronize();
  if ((abs(result_h.x - 2e7) >= 0.01) || (abs(result_h.y) >= 0.01)) {
    std::cout << "foo4() failed" << std::endl;
    return false;
  } else {
    return true;
  }
}

bool foo5(){
  float *d_A;
  float *d_B;
  float result;
  std::cout << "size " << num * sizeof(float) << " byte" << std::endl;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cudaMalloc(&d_A, num * sizeof(float));
  cudaMalloc(&d_B, num * sizeof(float));
  float *A = new float[num];
  float *B = new float[num];
  for (int i = 0; i < num; i++) {
    A[i] = 1;
    B[i] = 1;
  }
  cudaMemcpy(d_A, A, num * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, num * sizeof(float), cudaMemcpyHostToDevice);
  cublasSdot(handle, num, d_A, 1, d_B, 1, &result);
  cudaFree(d_A);
  cudaFree(d_B);
  cublasDestroy(handle);
  delete[] A;
  delete[] B;
  cudaDeviceSynchronize();
  if (abs(result - 1e7) >= 0.01) {
    std::cout << "foo5() failed" << std::endl;
    return false;
  } else {
    return true;
  }
}

bool foo6(){
  float *d_A;
  float *d_B;
  float *result;
  float result_h;
  std::cout << "size " << num * sizeof(float) << " byte" << std::endl;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cudaMalloc(&d_A, num * sizeof(float));
  cudaMalloc(&d_B, num * sizeof(float));
  cudaMalloc(&result, sizeof(float));
  float *A = new float[num];
  float *B = new float[num];
  for (int i = 0; i < num; i++) {
    A[i] = 1;
    B[i] = 1;
  }
  cudaMemcpy(d_A, A, num * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, num * sizeof(float), cudaMemcpyHostToDevice);
  cublasSdot(handle, num, d_A, 1, d_B, 1, result);
  cudaDeviceSynchronize();
  cudaMemcpy(&result_h, result, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(result);
  cublasDestroy(handle);
  delete[] A;
  delete[] B;
  cudaDeviceSynchronize();
  if (abs(result_h - 1e7) >= 0.01) {
    std::cout << "foo6() failed" << std::endl;
    return false;
  } else {
    return true;
  }
}

bool foo7(){
  float2 *d_A;
  float2 *d_B;
  float2 result;
  std::cout << "size " << num * sizeof(float2) << " byte" << std::endl;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cudaMalloc(&d_A, num * sizeof(float2));
  cudaMalloc(&d_B, num * sizeof(float2));
  float2 *A = new float2[num];
  float2 *B = new float2[num];
  for (int i = 0; i < num; i++) {
    A[i].x = 1;
    A[i].y = 1;
    B[i].x = 1;
    B[i].y = 1;
  }
  cudaMemcpy(d_A, A, num * sizeof(float2), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, num * sizeof(float2), cudaMemcpyHostToDevice);
  cublasCdotc(handle, num, d_A, 1, d_B, 1, &result);
  cudaFree(d_A);
  cudaFree(d_B);
  cublasDestroy(handle);
  delete[] A;
  delete[] B;
  cudaDeviceSynchronize();
  if ((abs(result.x - 2e7) >= 0.01) || (abs(result.y) >= 0.01)) {
    std::cout << "foo7() failed" << std::endl;
    return false;
  } else {
    return true;
  }
}

bool foo8(){
  float2 *d_A;
  float2 *d_B;
  float2 *result;
  float2 result_h;
  std::cout << "size " << num * sizeof(float2) << " byte" << std::endl;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cudaMalloc(&d_A, num * sizeof(float2));
  cudaMalloc(&d_B, num * sizeof(float2));
  cudaMalloc(&result, sizeof(float2));
  float2 *A = new float2[num];
  float2 *B = new float2[num];
  for (int i = 0; i < num; i++) {
    A[i].x = 1;
    A[i].y = 1;
    B[i].x = 1;
    B[i].y = 1;
  }
  cudaMemcpy(d_A, A, num * sizeof(float2), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, num * sizeof(float2), cudaMemcpyHostToDevice);
  cublasCdotu(handle, num, d_A, 1, d_B, 1, result);
  cudaDeviceSynchronize();
  cudaMemcpy(&result_h, result, sizeof(float2), cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(result);
  cublasDestroy(handle);
  delete[] A;
  delete[] B;
  cudaDeviceSynchronize();
  if ((abs(result_h.x) >= 0.01) || (abs(result_h.y - 2e7) >= 0.01)) {
    std::cout << "foo8() failed" << std::endl;
    return false;
  } else {
    return true;
  }
}

int main(){
  if (foo1() && foo2() && foo3() && foo4() &&
      foo5() && foo6() && foo7() && foo8()) {
    std::cout << "pass" << std::endl;
    return 0;
  } else {
    std::cout << "fail" << std::endl;
    return 1;
  }
}



