// ====------ rotg.cu---------- *- CUDA -* ----===////
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

bool foo1(){
  float a = 1.0f;
  float b = 1.0f;
  float c;
  float s;

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
  cublasSrotg(handle, &a, &b, &c, &s);
  cublasDestroy(handle);
  if ((std::abs(a - 1.41421) < 0.01) && (std::abs(b - 1.41421) < 0.01) &&
      (std::abs(c - 0.707107) < 0.01) && (std::abs(s - 0.707107) < 0.01)) {
    return true;
  } else {
    std::cout << "foo1() failed" << std::endl;
    return false;
  }
}

bool foo2(){
  float a = 1.0f;
  float b = 1.0f;
  float c;
  float s;

  float *d_a, *d_b, *d_c, *d_s;

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  cudaMalloc(&d_a, sizeof(float));
  cudaMalloc(&d_b, sizeof(float));
  cudaMalloc(&d_c, sizeof(float));
  cudaMalloc(&d_s, sizeof(float));

  cudaMemcpy(d_a, &a, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, &b, sizeof(float), cudaMemcpyHostToDevice);
  cublasSrotg(handle, d_a, d_b, d_c, d_s);
  cudaDeviceSynchronize();

  cudaMemcpy(&a, d_a, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&b, d_b, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&c, d_c, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&s, d_s, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFree(d_s);
  cublasDestroy(handle);

  if ((std::abs(a - 1.41421) < 0.01) && (std::abs(b - 1.41421) < 0.01) &&
      (std::abs(c - 0.707107) < 0.01) && (std::abs(s - 0.707107) < 0.01)) {
    return true;
  } else {
    std::cout << "foo2() failed" << std::endl;
    return false;
  }
}

bool foo3(){
  double a = 1.0;
  double b = 1.0;
  double c;
  double s;

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
  cublasDrotg(handle, &a, &b, &c, &s);
  cublasDestroy(handle);
  if ((std::abs(a - 1.41421) < 0.01) && (std::abs(b - 1.41421) < 0.01) &&
      (std::abs(c - 0.707107) < 0.01) && (std::abs(s - 0.707107) < 0.01)) {
    return true;
  } else {
    std::cout << "foo3() failed" << std::endl;
    return false;
  }
}

bool foo4(){
  double a = 1.0;
  double b = 1.0;
  double c;
  double s;

  double *d_a, *d_b, *d_c, *d_s;

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  cudaMalloc(&d_a, sizeof(double));
  cudaMalloc(&d_b, sizeof(double));
  cudaMalloc(&d_c, sizeof(double));
  cudaMalloc(&d_s, sizeof(double));

  cudaMemcpy(d_a, &a, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, &b, sizeof(double), cudaMemcpyHostToDevice);
  cublasDrotg(handle, d_a, d_b, d_c, d_s);
  cudaDeviceSynchronize();

  cudaMemcpy(&a, d_a, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&b, d_b, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&c, d_c, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&s, d_s, sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFree(d_s);
  cublasDestroy(handle);

  if ((std::abs(a - 1.41421) < 0.01) && (std::abs(b - 1.41421) < 0.01) &&
      (std::abs(c - 0.707107) < 0.01) && (std::abs(s - 0.707107) < 0.01)) {
    return true;
  } else {
    std::cout << "foo4() failed" << std::endl;
    return false;
  }
}

bool foo5(){
  float2 a;
  a.x = 1.0f;
  a.y = 1.0f;
  float2 b;
  b.x = 1.0f;
  b.y = 1.0f;
  float c;
  float2 s;

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
  cublasCrotg(handle, &a, &b, &c, &s);
  cublasDestroy(handle);

  if ((std::abs(a.x - 1.41421) < 0.01) && (std::abs(a.y - 1.41421) < 0.01) &&
      (std::abs(b.x - 1) < 0.01) && (std::abs(b.y - 1) < 0.01)  &&
      (std::abs(c - 0.707107) < 0.01) && (std::abs(s.x - 0.707107) < 0.01) &&
      (std::abs(s.y) < 0.01)) {
    return true;
  } else {
    std::cout << "foo5() failed" << std::endl;
    return false;
  }
}

bool foo6(){
  float2 a;
  a.x = 1.0;
  a.y = 1.0;
  float2 b;
  b.x = 1.0;
  b.y = 1.0;
  float c;
  float2 s;

  float2 *d_a, *d_b, *d_s;
  float *d_c;

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  cudaMalloc(&d_a, sizeof(float2));
  cudaMalloc(&d_b, sizeof(float2));
  cudaMalloc(&d_c, sizeof(float));
  cudaMalloc(&d_s, sizeof(float2));

  cudaMemcpy(d_a, &a, sizeof(float2), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, &b, sizeof(float2), cudaMemcpyHostToDevice);
  cublasCrotg(handle, d_a, d_b, d_c, d_s);
  cudaDeviceSynchronize();


  cudaMemcpy(&a, d_a, sizeof(float2), cudaMemcpyDeviceToHost);
  cudaMemcpy(&b, d_b, sizeof(float2), cudaMemcpyDeviceToHost);
  cudaMemcpy(&c, d_c, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&s, d_s, sizeof(float2), cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFree(d_s);

  cublasDestroy(handle);

  if ((std::abs(a.x - 1.41421) < 0.01) && (std::abs(a.y - 1.41421) < 0.01) &&
      (std::abs(b.x - 1) < 0.01) && (std::abs(b.y - 1) < 0.01)  &&
      (std::abs(c - 0.707107) < 0.01) && (std::abs(s.x - 0.707107) < 0.01) &&
      (std::abs(s.y) < 0.01)) {
    return true;
  } else {
    std::cout << "foo6() failed" << std::endl;
    return false;
  }
}

bool foo7(){
  double2 a;
  a.x = 1.0;
  a.y = 1.0;
  double2 b;
  b.x = 1.0;
  b.y = 1.0;
  double c;
  double2 s;

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
  cublasZrotg(handle, &a, &b, &c, &s);
  cublasDestroy(handle);

  if ((std::abs(a.x - 1.41421) < 0.01) && (std::abs(a.y - 1.41421) < 0.01) &&
      (std::abs(b.x - 1) < 0.01) && (std::abs(b.y - 1) < 0.01)  &&
      (std::abs(c - 0.707107) < 0.01) && (std::abs(s.x - 0.707107) < 0.01) &&
      (std::abs(s.y) < 0.01)) {
    return true;
  } else {
    std::cout << "foo7() failed" << std::endl;
    return false;
  }
}

bool foo8(){
  double2 a;
  a.x = 1.0;
  a.y = 1.0;
  double2 b;
  b.x = 1.0;
  b.y = 1.0;
  double c;
  double2 s;

  double2 *d_a, *d_b, *d_s;
  double *d_c;

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  cudaMalloc(&d_a, sizeof(double2));
  cudaMalloc(&d_b, sizeof(double2));
  cudaMalloc(&d_c, sizeof(double));
  cudaMalloc(&d_s, sizeof(double2));

  cudaMemcpy(d_a, &a, sizeof(double2), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, &b, sizeof(double2), cudaMemcpyHostToDevice);
  cublasZrotg(handle, d_a, d_b, d_c, d_s);
  cudaDeviceSynchronize();


  cudaMemcpy(&a, d_a, sizeof(double2), cudaMemcpyDeviceToHost);
  cudaMemcpy(&b, d_b, sizeof(double2), cudaMemcpyDeviceToHost);
  cudaMemcpy(&c, d_c, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&s, d_s, sizeof(double2), cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFree(d_s);

  cublasDestroy(handle);

  if ((std::abs(a.x - 1.41421) < 0.01) && (std::abs(a.y - 1.41421) < 0.01) &&
      (std::abs(b.x - 1) < 0.01) && (std::abs(b.y - 1) < 0.01)  &&
      (std::abs(c - 0.707107) < 0.01) && (std::abs(s.x - 0.707107) < 0.01) &&
      (std::abs(s.y) < 0.01)) {
    return true;
  } else {
    std::cout << "foo8() failed" << std::endl;
    return false;
  }
}

bool foo9(){
  float d1 = 1.0f;
  float d2 = 4.0f;
  float x1 = 1.0f;
  float y1 = 1.0f;
  float *param = (float *)malloc(5 * sizeof(float));

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
  cublasSrotmg(handle, &d1, &d2, &x1, &y1, param);
  cublasDestroy(handle);

  if ((std::abs(d1 - 3.2) < 0.01) && (std::abs(d2 - 0.8) < 0.01) &&
      (std::abs(x1 - 1.25) < 0.01) && (std::abs(param[0] - 1) < 0.01)  &&
      (std::abs(param[1] - 0.25) < 0.01) && (std::abs(param[4] - 1) < 0.01)) {
    free(param);
    return true;
  } else {
    free(param);
    std::cout << "foo9() failed" << std::endl;
    return false;
  }
}

bool foo10(){
  float d1 = 1.0f;
  float d2 = 4.0f;
  float x1 = 1.0f;
  float y1 = 1.0f;
  float *param = (float *)malloc(5 * sizeof(float));

  float *d_d1, *d_d2, *d_x1, *d_y1, *d_param;

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  cudaMalloc(&d_d1, sizeof(float));
  cudaMalloc(&d_d2, sizeof(float));
  cudaMalloc(&d_x1, sizeof(float));
  cudaMalloc(&d_y1, sizeof(float));
  cudaMalloc(&d_param, 5 * sizeof(float));

  cudaMemcpy(d_d1, &d1, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_d2, &d2, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x1, &x1, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y1, &y1, sizeof(float), cudaMemcpyHostToDevice);
  cublasSrotmg(handle, d_d1, d_d2, d_x1, d_y1, d_param);
  cudaDeviceSynchronize();

  cudaMemcpy(&d1, d_d1, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&d2, d_d2, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&x1, d_x1, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(param, d_param, 5 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_d1);
  cudaFree(d_d2);
  cudaFree(d_x1);
  cudaFree(d_y1);
  cudaFree(d_param);
  cublasDestroy(handle);

  if ((std::abs(d1 - 3.2) < 0.01) && (std::abs(d2 - 0.8) < 0.01) &&
      (std::abs(x1 - 1.25) < 0.01) && (std::abs(param[0] - 1) < 0.01)  &&
      (std::abs(param[1] - 0.25) < 0.01) && (std::abs(param[4] - 1) < 0.01)) {
    free(param);
    return true;
  } else {
    free(param);
    std::cout << "foo10() failed" << std::endl;
    return false;
  }
}

bool foo11(){
  double d1 = 1.0;
  double d2 = 4.0;
  double x1 = 1.0;
  double y1 = 1.0;
  double *param = (double *)malloc(5 * sizeof(double));

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
  cublasDrotmg(handle, &d1, &d2, &x1, &y1, param);
  cublasDestroy(handle);

  if ((std::abs(d1 - 3.2) < 0.01) && (std::abs(d2 - 0.8) < 0.01) &&
      (std::abs(x1 - 1.25) < 0.01) && (std::abs(param[0] - 1) < 0.01)  &&
      (std::abs(param[1] - 0.25) < 0.01) && (std::abs(param[4] - 1) < 0.01)) {
    free(param);
    return true;
  } else {
    free(param);
    std::cout << "foo11() failed" << std::endl;
    return false;
  }
}

bool foo12(){
  double d1 = 1.0;
  double d2 = 4.0;
  double x1 = 1.0;
  double y1 = 1.0;
  double *param = (double *)malloc(5 * sizeof(double));

  double *d_d1, *d_d2, *d_x1, *d_y1, *d_param;

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  cudaMalloc(&d_d1, sizeof(double));
  cudaMalloc(&d_d2, sizeof(double));
  cudaMalloc(&d_x1, sizeof(double));
  cudaMalloc(&d_y1, sizeof(double));
  cudaMalloc(&d_param, 5 * sizeof(double));

  cudaMemcpy(d_d1, &d1, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_d2, &d2, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x1, &x1, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y1, &y1, sizeof(double), cudaMemcpyHostToDevice);
  cublasDrotmg(handle, d_d1, d_d2, d_x1, d_y1, d_param);
  cudaDeviceSynchronize();

  cudaMemcpy(&d1, d_d1, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&d2, d_d2, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&x1, d_x1, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(param, d_param, 5 * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_d1);
  cudaFree(d_d2);
  cudaFree(d_x1);
  cudaFree(d_y1);
  cudaFree(d_param);
  cublasDestroy(handle);

  if ((std::abs(d1 - 3.2) < 0.01) && (std::abs(d2 - 0.8) < 0.01) &&
      (std::abs(x1 - 1.25) < 0.01) && (std::abs(param[0] - 1) < 0.01)  &&
      (std::abs(param[1] - 0.25) < 0.01) && (std::abs(param[4] - 1) < 0.01)) {
    free(param);
    return true;
  } else {
    free(param);
    std::cout << "foo12() failed" << std::endl;
    return false;
  }
}

int main(){
  if (foo1() && foo2() && foo3() && foo4() &&
      foo5() && foo6() && foo7() && foo8() &&
      foo9() && foo10() && foo11() && foo12()) {
    std::cout << "pass" << std::endl;
    return 0;
  } else {
    std::cout << "fail" << std::endl;
    return 1;
  }
}

