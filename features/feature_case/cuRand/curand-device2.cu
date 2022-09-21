// ====------ curand-device2.cu ---------------------------- *- CUDA -* ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include "curand_kernel.h"


__global__ void kernel1() {
  unsigned int u;
  uint4 u4;
  float f;
  float2 f2;
  float4 f4;
  double d;
  double2 d2;
  double4 d4;

  curandStatePhilox4_32_10_t rng1;
  curand_init(1, 2, 3, &rng1);
  u = curand(&rng1);

  curandStatePhilox4_32_10_t rng2;
  curand_init(1, 2, 3, &rng2);
  u4 = curand4(&rng2);

  curandStatePhilox4_32_10_t rng3;
  curand_init(1, 2, 3, &rng3);
  f = curand_normal(&rng3);

  curandStatePhilox4_32_10_t rng4;
  curand_init(1, 2, 3, &rng4);
  f2 = curand_normal2(&rng4);

  curandStatePhilox4_32_10_t rng5;
  curand_init(1, 2, 3, &rng5);
  d2 = curand_normal2_double(&rng5);

  curandStatePhilox4_32_10_t rng6;
  curand_init(1, 2, 3, &rng6);
  f4 = curand_normal4(&rng6);

  curandStatePhilox4_32_10_t rng7;
  curand_init(1, 2, 3, &rng7);
  d = curand_normal_double(&rng7);

  curandStatePhilox4_32_10_t rng8;
  curand_init(1, 2, 3, &rng8);
  f = curand_log_normal(&rng8, 3, 7);

  curandStatePhilox4_32_10_t rng9;
  curand_init(1, 2, 3, &rng9);
  f2 = curand_log_normal2(&rng9, 3, 7);

  curandStatePhilox4_32_10_t rng10;
  curand_init(1, 2, 3, &rng10);
  d2 = curand_log_normal2_double(&rng10, 3, 7);

  curandStatePhilox4_32_10_t rng11;
  curand_init(1, 2, 3, &rng11);
  f4 = curand_log_normal4(&rng11, 3, 7);

  curandStatePhilox4_32_10_t rng12;
  curand_init(1, 2, 3, &rng12);
  d = curand_log_normal_double(&rng12, 3, 7);

  curandStatePhilox4_32_10_t rng13;
  curand_init(1, 2, 3, &rng13);
  f = curand_uniform(&rng13);

  curandStatePhilox4_32_10_t rng14;
  curand_init(1, 2, 3, &rng14);
  d2 = curand_uniform2_double(&rng14);

  curandStatePhilox4_32_10_t rng15;
  curand_init(1, 2, 3, &rng15);
  f4 = curand_uniform4(&rng15);

  curandStatePhilox4_32_10_t rng16;
  curand_init(1, 2, 3, &rng16);
  d = curand_uniform_double(&rng16);

  curandStatePhilox4_32_10_t rng17;
  curand_init(1, 2, 3, &rng17);
  u = curand_poisson(&rng17, 3);

  curandStatePhilox4_32_10_t rng18;
  curand_init(1, 2, 3, &rng18);
  u4 = curand_poisson4(&rng18, 3);

  curandStatePhilox4_32_10_t rng19;
  curand_init(1, 2, 3, &rng19);
  d4 = curand_uniform4_double(&rng19);

  curandStatePhilox4_32_10_t rng20;
  curand_init(1, 2, 3, &rng20);
  d4 = curand_normal4_double(&rng20);

  curandStatePhilox4_32_10_t rng21;
  curand_init(1, 2, 3, &rng21);
  d4 = curand_log_normal4_double(&rng21, 3, 7);
}

__global__ void kernel2() {
  curandStatePhilox4_32_10_t rng1;
  curandStatePhilox4_32_10_t rng2;
  curand_init(11, 1, 1234, &rng1);
  curand_init(22, 2, 4321, &rng2);
  float x = curand_uniform(&rng1);
  float2 y = curand_normal2(&rng2);
}

__global__ void kernel3() {
  curandStateMRG32k3a_t rng1;
  curandStatePhilox4_32_10_t rng2;
  curandStateXORWOW_t rng3;

  curand_init(1, 2, 3, &rng1);
  curand_init(1, 2, 3, &rng2);
  curand_init(1, 2, 3, &rng3);

  skipahead(1, &rng1);
  skipahead(2, &rng2);
  skipahead(3, &rng3);

  skipahead_sequence(1, &rng1);
  skipahead_sequence(2, &rng2);
  skipahead_sequence(3, &rng3);

  skipahead_subsequence(1, &rng1);

  curand_uniform(&rng1);
  curand_uniform(&rng2);
  curand_uniform(&rng3);
}

__global__ void type_test() {
  curandStateXORWOW_t rng1;
  curandStateXORWOW rng2;
  curandState_t rng3;
  curandState rng4;
  curandStatePhilox4_32_10_t rng5;
  curandStatePhilox4_32_10 rng6;
  curandStateMRG32k3a_t rng7;
  curandStateMRG32k3a rng8;
}

int main() {
  kernel1<<<1,1>>>();
  kernel2<<<1,1>>>();
  kernel3<<<1,1>>>();
  return 0;
}

__global__ void kernel4() {
  curandStateMRG32k3a_t rng;
  curand_init(1, 2 + 3, 4, &rng);
  skipahead_sequence(2 + 3, &rng);
  skipahead_subsequence(2 + 3, &rng);
}

