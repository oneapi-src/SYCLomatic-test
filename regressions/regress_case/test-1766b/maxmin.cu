// ====------ maxmin.cu---------- *- CUDA -* ----===////
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

const int num = 10;

bool foo1(){
  double *d_A;
  int result;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cudaMalloc(&d_A, num * sizeof(double));
  double *A = new double[num];
  for (int i = 0; i < num; i++) {
    A[i] = i;
  }
  cudaMemcpy(d_A, A, num * sizeof(double), cudaMemcpyHostToDevice);
  cublasIdamax(handle, num, d_A, 1, &result);
  cudaFree(d_A);
  cublasDestroy(handle);
  delete[] A;
  cudaDeviceSynchronize();
  if (abs(result - 9) >= 0.01) {
    std::cout << "foo1() failed" << std::endl;
    return false;
  } else {
    return true;
  }
}

bool foo2(){
  double *d_A;
  int *result;
  int result_h;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cudaMalloc(&d_A, num * sizeof(double));
  cudaMalloc(&result, sizeof(int));
  double *A = new double[num];
  for (int i = 0; i < num; i++) {
    A[i] = i;
  }
  cudaMemcpy(d_A, A, num * sizeof(double), cudaMemcpyHostToDevice);
  cublasIdamax(handle, num, d_A, 1, result);
  cudaDeviceSynchronize();
  cudaMemcpy(&result_h, result, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(result);
  cublasDestroy(handle);
  delete[] A;
  cudaDeviceSynchronize();
  if (abs(result_h - 9) >= 0.01) {
    std::cout << "foo2() failed" << std::endl;
    return false;
  } else {
    return true;
  }
}

bool foo3(){
  float *d_A;
  int result;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cudaMalloc(&d_A, num * sizeof(float));
  float *A = new float[num];
  for (int i = 0; i < num; i++) {
    A[i] = i;
  }
  cudaMemcpy(d_A, A, num * sizeof(float), cudaMemcpyHostToDevice);
  cublasIsamax(handle, num, d_A, 1, &result);
  cudaFree(d_A);
  cublasDestroy(handle);
  delete[] A;
  cudaDeviceSynchronize();
  if (abs(result - 9) >= 0.01) {
    std::cout << "foo3() failed" << std::endl;
    return false;
  } else {
    return true;
  }
}

bool foo4(){
  float *d_A;
  int *result;
  int result_h;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cudaMalloc(&d_A, num * sizeof(float));
  cudaMalloc(&result, sizeof(int));
  float *A = new float[num];
  for (int i = 0; i < num; i++) {
    A[i] = i;
  }
  cudaMemcpy(d_A, A, num * sizeof(float), cudaMemcpyHostToDevice);
  cublasIsamax(handle, num, d_A, 1, result);
  cudaDeviceSynchronize();
  cudaMemcpy(&result_h, result, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(result);
  cublasDestroy(handle);
  delete[] A;
  cudaDeviceSynchronize();
  if (abs(result_h - 9) >= 0.01) {
    std::cout << "foo4() failed" << std::endl;
    return false;
  } else {
    return true;
  }
}

bool foo5(){
  double2 *d_A;
  int result;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cudaMalloc(&d_A, num * sizeof(double2));
  double2 *A = new double2[num];
  for (int i = 0; i < num; i++) {
    A[i].x = i;
    A[i].y = i;
  }
  cudaMemcpy(d_A, A, num * sizeof(double2), cudaMemcpyHostToDevice);
  cublasIzamax(handle, num, d_A, 1, &result);
  cudaFree(d_A);
  cublasDestroy(handle);
  delete[] A;
  cudaDeviceSynchronize();
  if (abs(result - 9) >= 0.01) {
    std::cout << "foo5() failed" << std::endl;
    return false;
  } else {
    return true;
  }
}

bool foo6(){
  double2 *d_A;
  int *result;
  int result_h;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cudaMalloc(&d_A, num * sizeof(double2));
  cudaMalloc(&result, sizeof(int));
  double2 *A = new double2[num];
  for (int i = 0; i < num; i++) {
    A[i].x = i;
    A[i].y = i;
  }
  cudaMemcpy(d_A, A, num * sizeof(double2), cudaMemcpyHostToDevice);
  cublasIzamax(handle, num, d_A, 1, result);
  cudaDeviceSynchronize();
  cudaMemcpy(&result_h, result, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(result);
  cublasDestroy(handle);
  delete[] A;
  cudaDeviceSynchronize();
  if (abs(result_h - 9) >= 0.01) {
    std::cout << "foo6() failed" << std::endl;
    return false;
  } else {
    return true;
  }
}

bool foo7(){
  float2 *d_A;
  int result;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cudaMalloc(&d_A, num * sizeof(float2));
  float2 *A = new float2[num];
  for (int i = 0; i < num; i++) {
    A[i].x = i;
    A[i].y = i;
  }
  cudaMemcpy(d_A, A, num * sizeof(float2), cudaMemcpyHostToDevice);
  cublasIcamax(handle, num, d_A, 1, &result);
  cudaFree(d_A);
  cublasDestroy(handle);
  delete[] A;
  cudaDeviceSynchronize();
  if (abs(result - 9) >= 0.01) {
    std::cout << "foo7() failed" << std::endl;
    return false;
  } else {
    return true;
  }
}

bool foo8(){
  float2 *d_A;
  int *result;
  int result_h;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cudaMalloc(&d_A, num * sizeof(float2));
  cudaMalloc(&result, sizeof(int));
  float2 *A = new float2[num];
  for (int i = 0; i < num; i++) {
    A[i].x = i;
    A[i].y = i;
  }
  cudaMemcpy(d_A, A, num * sizeof(float2), cudaMemcpyHostToDevice);
  cublasIcamax(handle, num, d_A, 1, result);
  cudaDeviceSynchronize();
  cudaMemcpy(&result_h, result, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(result);
  cublasDestroy(handle);
  delete[] A;
  cudaDeviceSynchronize();
  if (abs(result_h - 9) >= 0.01) {
    std::cout << "foo8() failed" << std::endl;
    return false;
  } else {
    return true;
  }
}

bool foo1b(){
  double *d_A;
  int result;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cudaMalloc(&d_A, num * sizeof(double));
  double *A = new double[num];
  for (int i = 0; i < num; i++) {
    A[i] = i;
  }
  cudaMemcpy(d_A, A, num * sizeof(double), cudaMemcpyHostToDevice);
  cublasIdamin(handle, num, d_A, 1, &result);
  cudaFree(d_A);
  cublasDestroy(handle);
  delete[] A;
  cudaDeviceSynchronize();
  if (abs(result) >= 0.01) {
    std::cout << "foo1b() failed" << std::endl;
    return false;
  } else {
    return true;
  }
}

bool foo2b(){
  double *d_A;
  int *result;
  int result_h;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cudaMalloc(&d_A, num * sizeof(double));
  cudaMalloc(&result, sizeof(int));
  double *A = new double[num];
  for (int i = 0; i < num; i++) {
    A[i] = i;
  }
  cudaMemcpy(d_A, A, num * sizeof(double), cudaMemcpyHostToDevice);
  cublasIdamin(handle, num, d_A, 1, result);
  cudaDeviceSynchronize();
  cudaMemcpy(&result_h, result, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(result);
  cublasDestroy(handle);
  delete[] A;
  cudaDeviceSynchronize();
  if (abs(result_h) >= 0.01) {
    std::cout << "foo2b() failed" << std::endl;
    return false;
  } else {
    return true;
  }
}

bool foo3b(){
  float *d_A;
  int result;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cudaMalloc(&d_A, num * sizeof(float));
  float *A = new float[num];
  for (int i = 0; i < num; i++) {
    A[i] = i;
  }
  cudaMemcpy(d_A, A, num * sizeof(float), cudaMemcpyHostToDevice);
  cublasIsamin(handle, num, d_A, 1, &result);
  cudaFree(d_A);
  cublasDestroy(handle);
  delete[] A;
  cudaDeviceSynchronize();
  if (abs(result) >= 0.01) {
    std::cout << "foo3b() failed" << std::endl;
    return false;
  } else {
    return true;
  }
}

bool foo4b(){
  float *d_A;
  int *result;
  int result_h;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cudaMalloc(&d_A, num * sizeof(float));
  cudaMalloc(&result, sizeof(int));
  float *A = new float[num];
  for (int i = 0; i < num; i++) {
    A[i] = i;
  }
  cudaMemcpy(d_A, A, num * sizeof(float), cudaMemcpyHostToDevice);
  cublasIsamin(handle, num, d_A, 1, result);
  cudaDeviceSynchronize();
  cudaMemcpy(&result_h, result, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(result);
  cublasDestroy(handle);
  delete[] A;
  cudaDeviceSynchronize();
  if (abs(result_h) >= 0.01) {
    std::cout << "foo4b() failed" << std::endl;
    return false;
  } else {
    return true;
  }
}

bool foo5b(){
  double2 *d_A;
  int result;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cudaMalloc(&d_A, num * sizeof(double2));
  double2 *A = new double2[num];
  for (int i = 0; i < num; i++) {
    A[i].x = i;
    A[i].y = i;
  }
  cudaMemcpy(d_A, A, num * sizeof(double2), cudaMemcpyHostToDevice);
  cublasIzamin(handle, num, d_A, 1, &result);
  cudaFree(d_A);
  cublasDestroy(handle);
  delete[] A;
  cudaDeviceSynchronize();
  if (abs(result) >= 0.01) {
    std::cout << "foo5b() failed" << std::endl;
    return false;
  } else {
    return true;
  }
}

bool foo6b(){
  double2 *d_A;
  int *result;
  int result_h;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cudaMalloc(&d_A, num * sizeof(double2));
  cudaMalloc(&result, sizeof(int));
  double2 *A = new double2[num];
  for (int i = 0; i < num; i++) {
    A[i].x = i;
    A[i].y = i;
  }
  cudaMemcpy(d_A, A, num * sizeof(double2), cudaMemcpyHostToDevice);
  cublasIzamin(handle, num, d_A, 1, result);
  cudaDeviceSynchronize();
  cudaMemcpy(&result_h, result, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(result);
  cublasDestroy(handle);
  delete[] A;
  cudaDeviceSynchronize();
  if (abs(result_h) >= 0.01) {
    std::cout << "foo6b() failed" << std::endl;
    return false;
  } else {
    return true;
  }
}

bool foo7b(){
  float2 *d_A;
  int result;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cudaMalloc(&d_A, num * sizeof(float2));
  float2 *A = new float2[num];
  for (int i = 0; i < num; i++) {
    A[i].x = i;
    A[i].y = i;
  }
  cudaMemcpy(d_A, A, num * sizeof(float2), cudaMemcpyHostToDevice);
  cublasIcamin(handle, num, d_A, 1, &result);
  cudaFree(d_A);
  cublasDestroy(handle);
  delete[] A;
  cudaDeviceSynchronize();
  if (abs(result) >= 0.01) {
    std::cout << "foo7b() failed" << std::endl;
    return false;
  } else {
    return true;
  }
}

bool foo8b(){
  float2 *d_A;
  int *result;
  int result_h;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cudaMalloc(&d_A, num * sizeof(float2));
  cudaMalloc(&result, sizeof(int));
  float2 *A = new float2[num];
  for (int i = 0; i < num; i++) {
    A[i].x = i;
    A[i].y = i;
  }
  cudaMemcpy(d_A, A, num * sizeof(float2), cudaMemcpyHostToDevice);
  cublasIcamin(handle, num, d_A, 1, result);
  cudaDeviceSynchronize();
  cudaMemcpy(&result_h, result, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(result);
  cublasDestroy(handle);
  delete[] A;
  cudaDeviceSynchronize();
  if (abs(result_h) >= 0.01) {
    std::cout << "foo8b() failed" << std::endl;
    return false;
  } else {
    return true;
  }
}

int main(){
  if (foo1() && foo2() && foo3() && foo4() &&
      foo5() && foo6() && foo7() && foo8() &&
      foo1b() && foo2b() && foo3b() && foo4b() &&
      foo5b() && foo6b() && foo7b() && foo8b()) {
    std::cout << "pass" << std::endl;
    return 0;
  } else {
    std::cout << "fail" << std::endl;
    return 1;
  }
}



