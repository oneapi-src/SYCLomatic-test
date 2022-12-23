// ====------ cublas_v1_runable.cu ----------------------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include "cublas.h"
#include <cmath>
#include <vector>
#include <cstdio>
#include <complex>

template<class d_data_t>
struct Data {
  float *h_data;
  d_data_t *d_data;
  int element_num;
  Data(int element_num) : element_num(element_num) {
    h_data = (float*)malloc(sizeof(float) * element_num);
    memset(h_data, 0, sizeof(float) * element_num);
    cudaMalloc(&d_data, sizeof(d_data_t) * element_num);
    cudaMemset(d_data, 0, sizeof(d_data_t) * element_num);
  }
  Data(float* input_data, int element_num) : element_num(element_num) {
    h_data = (float*)malloc(sizeof(float) * element_num);
    cudaMalloc(&d_data, sizeof(d_data_t) * element_num);
    cudaMemset(d_data, 0, sizeof(d_data_t) * element_num);
    memcpy(h_data, input_data, sizeof(float) * element_num);
  }
  ~Data() {
    free(h_data);
    cudaFree(d_data);
  }
  void H2D() {
    d_data_t* h_temp = (d_data_t*)malloc(sizeof(d_data_t) * element_num);
    memset(h_temp, 0, sizeof(d_data_t) * element_num);
    from_float_convert(h_data, h_temp);
    cudaMemcpy(d_data, h_temp, sizeof(d_data_t) * element_num, cudaMemcpyHostToDevice);
    free(h_temp);
  }
  void D2H() {
    d_data_t* h_temp = (d_data_t*)malloc(sizeof(d_data_t) * element_num);
    memset(h_temp, 0, sizeof(d_data_t) * element_num);
    cudaMemcpy(h_temp, d_data, sizeof(d_data_t) * element_num, cudaMemcpyDeviceToHost);
    to_float_convert(h_temp, h_data);
    free(h_temp);
  }
private:
  inline void from_float_convert(float* in, d_data_t* out) {
    for (int i = 0; i < element_num; i++)
      out[i] = in[i];
  }
  inline void to_float_convert(d_data_t* in, float* out) {
    for (int i = 0; i < element_num; i++)
      out[i] = in[i];
  }
};
template <>
inline void Data<float2>::from_float_convert(float* in, float2* out) {
  for (int i = 0; i < element_num; i++)
    out[i].x = in[i];
}
template <>
inline void Data<double2>::from_float_convert(float* in, double2* out) {
  for (int i = 0; i < element_num; i++)
    out[i].x = in[i];
}
template <>
inline void Data<std::complex<int8_t>>::from_float_convert(float* in, std::complex<int8_t>* out) {
  for (int i = 0; i < element_num; i++)
    reinterpret_cast<int8_t(&)[2]>(out[i])[0] = int8_t(in[i]);
}

template <>
inline void Data<float2>::to_float_convert(float2* in, float* out) {
  for (int i = 0; i < element_num; i++)
    out[i] = in[i].x;
}
template <>
inline void Data<double2>::to_float_convert(double2* in, float* out) {
  for (int i = 0; i < element_num; i++)
    out[i] = in[i].x;
}
template <>
inline void Data<half>::to_float_convert(half* in, float* out) {
  for (int i = 0; i < element_num; i++)
    out[i] = __half2float(in[i]);
}
template <>
inline void Data<__nv_bfloat16>::to_float_convert(__nv_bfloat16* in, float* out) {
  for (int i = 0; i < element_num; i++)
    out[i] = in[i];
}

bool compare_result(float* expect, float* result, int element_num) {
  for (int i = 0; i < element_num; i++) {
    if (std::abs(result[i]-expect[i]) >= 0.1) {
      return false;
    }
  }
  return true;
}

bool test_passed = true;

void test_cublasTrot() {
  std::vector<float> v = {2, 3, 5, 7};
  Data<float2> x0(v.data(), 4);
  Data<double2> x1(v.data(), 4);
  Data<float2> y0(v.data(), 4);
  Data<double2> y1(v.data(), 4);

  float cos = 0.866;
  float sin = 0.5;

  cublasInit();

  x0.H2D();
  x1.H2D();
  y0.H2D();
  y1.H2D();

  cublasCrot(4, x0.d_data, 1,  y0.d_data, 1, cos, float2{sin, 0});
  cublasZrot(4, x1.d_data, 1,  y1.d_data, 1, cos, double2{sin, 0});

  x0.D2H();
  x1.D2H();
  y0.D2H();
  y1.D2H();

  cudaStreamSynchronize(0);
  cublasShutdown();

  float expect_x[4] = {2.732000,4.098000,6.830000,9.562000};
  float expect_y[4] = {0.732000,1.098000,1.830000,2.562000};
  if (compare_result(expect_x, x0.h_data, 4) &&
      compare_result(expect_x, x1.h_data, 4) &&
      compare_result(expect_y, y0.h_data, 4) &&
      compare_result(expect_y, y1.h_data, 4))
    printf("Trot pass\n");
  else {
    printf("Trot fail\n");
    test_passed = false;
  }
}

int main() {
  test_cublasTrot();

  if (test_passed)
    return 0;
  return -1;
}

