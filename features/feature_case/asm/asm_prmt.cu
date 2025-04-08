// ====------ asm_prmt.cu ------------------------- *- CUDA -* -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <cuda_runtime.h>
#include <iostream>

#define TEST(FN)                                                               \
  {                                                                            \
    if (FN()) {                                                                \
      printf("Test " #FN " PASS\n");                                           \
    } else {                                                                   \
      printf("Test " #FN " FAIL\n");                                           \
      return 1;                                                                \
    }                                                                          \
  }

__device__ inline unsigned int
byte_level_permute(unsigned int a, unsigned int b, unsigned int s) {
  unsigned int ret;
  ret =
      ((((std::uint64_t)b << 32 | a) >> (s & 0x7) * 8) & 0xff) |
      (((((std::uint64_t)b << 32 | a) >> ((s >> 4) & 0x7) * 8) & 0xff) << 8) |
      (((((std::uint64_t)b << 32 | a) >> ((s >> 8) & 0x7) * 8) & 0xff) << 16) |
      (((((std::uint64_t)b << 32 | a) >> ((s >> 12) & 0x7) * 8) & 0xff) << 24);
  return ret;
}

__device__ uint32_t custom_byte_level_permute_reference(uint32_t a, uint32_t b,
                                                        uint32_t sel,
                                                        int mode = 0) {
  constexpr uint16_t lookup[6][4] = {
      {0x3210, 0x4321, 0x5432, 0x6543}, // Forward 4-byte extract
      {0x5670, 0x6701, 0x7012, 0x0123}, // Backward 4-byte extract
      {0x0000, 0x1111, 0x2222, 0x3333}, // Replicate 8-bit values
      {0x3210, 0x3211, 0x3222, 0x3333}, // Edge clamp left
      {0x0000, 0x1110, 0x2210, 0x3210}, // Edge clamp right
      {0x1010, 0x3232, 0x1010, 0x3232}  // Replicate 16-bit values
  };

  if (mode >= 1 && mode <= 6) {
    return byte_level_permute(a, b, lookup[mode - 1][sel & 0x3]);
  }
  return byte_level_permute(a, b, sel);
}

__global__ void prmt_b32_kernel(uint32_t *d_result, uint32_t a) {
  static constexpr uint32_t sel = 0x3210;
  static constexpr uint32_t b = 0;
  uint32_t h_ptx, h_byte_perm;

  asm volatile("prmt.b32 %0, %1, %2, %3;\n"
               : "=r"(h_ptx)
               : "r"(a), "n"(b), "n"(sel));

  h_byte_perm = custom_byte_level_permute_reference(a, b, sel, 0);

  d_result[0] = h_ptx;
  d_result[1] = h_byte_perm;
}

__global__ void prmt_b32_f4e_kernel_0(uint32_t *d_result, uint32_t a) {
  static constexpr uint32_t sel = 0;
  static constexpr uint32_t b = 0x64646464;
  uint32_t h_ptx, h_byte_perm;

  asm volatile("prmt.b32.f4e %0, %1, %2, %3;\n"
               : "=r"(h_ptx)
               : "r"(a), "n"(b), "n"(sel));

  h_byte_perm = custom_byte_level_permute_reference(a, b, sel, 1);

  d_result[0] = h_ptx;
  d_result[1] = h_byte_perm;
}

__global__ void prmt_b32_f4e_kernel_1(uint32_t *d_result, uint32_t a) {
  static constexpr uint32_t sel = 1;
  static constexpr uint32_t b = 0x64646464;
  uint32_t h_ptx, h_byte_perm;

  asm volatile("prmt.b32.f4e %0, %1, %2, %3;\n"
               : "=r"(h_ptx)
               : "r"(a), "n"(b), "n"(sel));

  h_byte_perm = custom_byte_level_permute_reference(a, b, sel, 1);

  d_result[0] = h_ptx;
  d_result[1] = h_byte_perm;
}

__global__ void prmt_b32_f4e_kernel_2(uint32_t *d_result, uint32_t a) {
  static constexpr uint32_t sel = 2;
  static constexpr uint32_t b = 0x64646464;
  uint32_t h_ptx, h_byte_perm;

  asm volatile("prmt.b32.f4e %0, %1, %2, %3;\n"
               : "=r"(h_ptx)
               : "r"(a), "n"(b), "n"(sel));

  h_byte_perm = custom_byte_level_permute_reference(a, b, sel, 1);

  d_result[0] = h_ptx;
  d_result[1] = h_byte_perm;
}

__global__ void prmt_b32_f4e_kernel_3(uint32_t *d_result, uint32_t a) {
  static constexpr uint32_t sel = 3;
  static constexpr uint32_t b = 0x64646464;
  uint32_t h_ptx, h_byte_perm;

  asm volatile("prmt.b32.f4e %0, %1, %2, %3;\n"
               : "=r"(h_ptx)
               : "r"(a), "n"(b), "n"(sel));

  h_byte_perm = custom_byte_level_permute_reference(a, b, sel, 1);

  d_result[0] = h_ptx;
  d_result[1] = h_byte_perm;
}

__global__ void prmt_b32_b4e_kernel_0(uint32_t *d_result, uint32_t a) {
  static constexpr uint32_t sel = 0;
  static constexpr uint32_t b = 0x64646464;
  uint32_t h_ptx, h_byte_perm;

  asm volatile("prmt.b32.b4e %0, %1, %2, %3;\n"
               : "=r"(h_ptx)
               : "r"(a), "n"(b), "n"(sel));

  h_byte_perm = custom_byte_level_permute_reference(a, b, sel, 2);

  d_result[0] = h_ptx;
  d_result[1] = h_byte_perm;
}

__global__ void prmt_b32_b4e_kernel_1(uint32_t *d_result, uint32_t a) {
  static constexpr uint32_t sel = 1;
  static constexpr uint32_t b = 0x64646464;
  uint32_t h_ptx, h_byte_perm;

  asm volatile("prmt.b32.b4e %0, %1, %2, %3;\n"
               : "=r"(h_ptx)
               : "r"(a), "n"(b), "n"(sel));

  h_byte_perm = custom_byte_level_permute_reference(a, b, sel, 2);

  d_result[0] = h_ptx;
  d_result[1] = h_byte_perm;
}

__global__ void prmt_b32_b4e_kernel_2(uint32_t *d_result, uint32_t a) {
  static constexpr uint32_t sel = 2;
  static constexpr uint32_t b = 0x64646464;
  uint32_t h_ptx, h_byte_perm;

  asm volatile("prmt.b32.b4e %0, %1, %2, %3;\n"
               : "=r"(h_ptx)
               : "r"(a), "n"(b), "n"(sel));

  h_byte_perm = custom_byte_level_permute_reference(a, b, sel, 2);

  d_result[0] = h_ptx;
  d_result[1] = h_byte_perm;
}

__global__ void prmt_b32_b4e_kernel_3(uint32_t *d_result, uint32_t a) {
  static constexpr uint32_t sel = 3;
  static constexpr uint32_t b = 0x64646464;
  uint32_t h_ptx, h_byte_perm;

  asm volatile("prmt.b32.b4e %0, %1, %2, %3;\n"
               : "=r"(h_ptx)
               : "r"(a), "n"(b), "n"(sel));

  h_byte_perm = custom_byte_level_permute_reference(a, b, sel, 2);

  d_result[0] = h_ptx;
  d_result[1] = h_byte_perm;
}

__global__ void prmt_b32_rc8_kernel_0(uint32_t *d_result, uint32_t a) {
  static constexpr uint32_t sel = 0;
  static constexpr uint32_t b = 0x64646464;
  uint32_t h_ptx, h_byte_perm;

  asm volatile("prmt.b32.rc8 %0, %1, %2, %3;\n"
               : "=r"(h_ptx)
               : "r"(a), "n"(b), "n"(sel));

  h_byte_perm = custom_byte_level_permute_reference(a, b, sel, 3);

  d_result[0] = h_ptx;
  d_result[1] = h_byte_perm;
}

__global__ void prmt_b32_rc8_kernel_1(uint32_t *d_result, uint32_t a) {
  static constexpr uint32_t sel = 1;
  static constexpr uint32_t b = 0x64646464;
  uint32_t h_ptx, h_byte_perm;

  asm volatile("prmt.b32.rc8 %0, %1, %2, %3;\n"
               : "=r"(h_ptx)
               : "r"(a), "n"(b), "n"(sel));

  h_byte_perm = custom_byte_level_permute_reference(a, b, sel, 3);

  d_result[0] = h_ptx;
  d_result[1] = h_byte_perm;
}

__global__ void prmt_b32_rc8_kernel_2(uint32_t *d_result, uint32_t a) {
  static constexpr uint32_t sel = 2;
  static constexpr uint32_t b = 0x64646464;
  uint32_t h_ptx, h_byte_perm;

  asm volatile("prmt.b32.rc8 %0, %1, %2, %3;\n"
               : "=r"(h_ptx)
               : "r"(a), "n"(b), "n"(sel));

  h_byte_perm = custom_byte_level_permute_reference(a, b, sel, 3);

  d_result[0] = h_ptx;
  d_result[1] = h_byte_perm;
}

__global__ void prmt_b32_rc8_kernel_3(uint32_t *d_result, uint32_t a) {
  static constexpr uint32_t sel = 3;
  static constexpr uint32_t b = 0x64646464;
  uint32_t h_ptx, h_byte_perm;

  asm volatile("prmt.b32.rc8 %0, %1, %2, %3;\n"
               : "=r"(h_ptx)
               : "r"(a), "n"(b), "n"(sel));

  h_byte_perm = custom_byte_level_permute_reference(a, b, sel, 3);

  d_result[0] = h_ptx;
  d_result[1] = h_byte_perm;
}

__global__ void prmt_b32_ecl_kernel_0(uint32_t *d_result, uint32_t a) {
  static constexpr uint32_t sel = 0;
  static constexpr uint32_t b = 0x64646464;
  uint32_t h_ptx, h_byte_perm;

  asm volatile("prmt.b32.ecl  %0, %1, %2, %3;\n"
               : "=r"(h_ptx)
               : "r"(a), "n"(b), "n"(sel));

  h_byte_perm = custom_byte_level_permute_reference(a, b, sel, 4);

  d_result[0] = h_ptx;
  d_result[1] = h_byte_perm;
}

__global__ void prmt_b32_ecl_kernel_1(uint32_t *d_result, uint32_t a) {
  static constexpr uint32_t sel = 1;
  static constexpr uint32_t b = 0x64646464;
  uint32_t h_ptx, h_byte_perm;

  asm volatile("prmt.b32.ecl  %0, %1, %2, %3;\n"
               : "=r"(h_ptx)
               : "r"(a), "n"(b), "n"(sel));

  h_byte_perm = custom_byte_level_permute_reference(a, b, sel, 4);

  d_result[0] = h_ptx;
  d_result[1] = h_byte_perm;
}

__global__ void prmt_b32_ecl_kernel_2(uint32_t *d_result, uint32_t a) {
  static constexpr uint32_t sel = 2;
  static constexpr uint32_t b = 0x64646464;
  uint32_t h_ptx, h_byte_perm;

  asm volatile("prmt.b32.ecl  %0, %1, %2, %3;\n"
               : "=r"(h_ptx)
               : "r"(a), "n"(b), "n"(sel));

  h_byte_perm = custom_byte_level_permute_reference(a, b, sel, 4);

  d_result[0] = h_ptx;
  d_result[1] = h_byte_perm;
}

__global__ void prmt_b32_ecl_kernel_3(uint32_t *d_result, uint32_t a) {
  static constexpr uint32_t sel = 3;
  static constexpr uint32_t b = 0x64646464;
  uint32_t h_ptx, h_byte_perm;

  asm volatile("prmt.b32.ecl  %0, %1, %2, %3;\n"
               : "=r"(h_ptx)
               : "r"(a), "n"(b), "n"(sel));

  h_byte_perm = custom_byte_level_permute_reference(a, b, sel, 4);

  d_result[0] = h_ptx;
  d_result[1] = h_byte_perm;
}

__global__ void prmt_b32_ecr_kernel_0(uint32_t *d_result, uint32_t a) {
  static constexpr uint32_t sel = 0x0;
  static constexpr uint32_t b = 0x64646464;
  uint32_t h_ptx, h_byte_perm;

  asm volatile("prmt.b32.ecr %0, %1, %2, %3;\n"
               : "=r"(h_ptx)
               : "r"(a), "n"(b), "n"(sel));

  h_byte_perm = custom_byte_level_permute_reference(a, b, sel, 5);

  d_result[0] = h_ptx;
  d_result[1] = h_byte_perm;
}

__global__ void prmt_b32_ecr_kernel_1(uint32_t *d_result, uint32_t a) {
  static constexpr uint32_t sel = 1;
  static constexpr uint32_t b = 0x64646464;
  uint32_t h_ptx, h_byte_perm;

  asm volatile("prmt.b32.ecr %0, %1, %2, %3;\n"
               : "=r"(h_ptx)
               : "r"(a), "n"(b), "n"(sel));

  h_byte_perm = custom_byte_level_permute_reference(a, b, sel, 5);

  d_result[0] = h_ptx;
  d_result[1] = h_byte_perm;
}

__global__ void prmt_b32_ecr_kernel_2(uint32_t *d_result, uint32_t a) {
  static constexpr uint32_t sel = 2;
  static constexpr uint32_t b = 0x64646464;
  uint32_t h_ptx, h_byte_perm;

  asm volatile("prmt.b32.ecr %0, %1, %2, %3;\n"
               : "=r"(h_ptx)
               : "r"(a), "n"(b), "n"(sel));

  h_byte_perm = custom_byte_level_permute_reference(a, b, sel, 5);

  d_result[0] = h_ptx;
  d_result[1] = h_byte_perm;
}

__global__ void prmt_b32_ecr_kernel_3(uint32_t *d_result, uint32_t a) {
  static constexpr uint32_t sel = 3;
  static constexpr uint32_t b = 0x64646464;
  uint32_t h_ptx, h_byte_perm;

  asm volatile("prmt.b32.ecr %0, %1, %2, %3;\n"
               : "=r"(h_ptx)
               : "r"(a), "n"(b), "n"(sel));

  h_byte_perm = custom_byte_level_permute_reference(a, b, sel, 5);

  d_result[0] = h_ptx;
  d_result[1] = h_byte_perm;
}

__global__ void prmt_b32_rc16_kernel_0(uint32_t *d_result, uint32_t a) {
  static constexpr uint32_t sel = 0;
  static constexpr uint32_t b = 0x64646464;
  uint32_t h_ptx, h_byte_perm;

  asm volatile("prmt.b32.rc16 %0, %1, %2, %3;\n"
               : "=r"(h_ptx)
               : "r"(a), "n"(b), "n"(sel));

  h_byte_perm = custom_byte_level_permute_reference(a, b, sel, 6);

  d_result[0] = h_ptx;
  d_result[1] = h_byte_perm;
}

__global__ void prmt_b32_rc16_kernel_1(uint32_t *d_result, uint32_t a) {
  static constexpr uint32_t sel = 1;
  static constexpr uint32_t b = 0x64646464;
  uint32_t h_ptx, h_byte_perm;

  asm volatile("prmt.b32.rc16 %0, %1, %2, %3;\n"
               : "=r"(h_ptx)
               : "r"(a), "n"(b), "n"(sel));

  h_byte_perm = custom_byte_level_permute_reference(a, b, sel, 6);

  d_result[0] = h_ptx;
  d_result[1] = h_byte_perm;
}

__global__ void prmt_b32_rc16_kernel_2(uint32_t *d_result, uint32_t a) {
  static constexpr uint32_t sel = 2;
  static constexpr uint32_t b = 0x64646464;
  uint32_t h_ptx, h_byte_perm;

  asm volatile("prmt.b32.rc16 %0, %1, %2, %3;\n"
               : "=r"(h_ptx)
               : "r"(a), "n"(b), "n"(sel));

  h_byte_perm = custom_byte_level_permute_reference(a, b, sel, 6);

  d_result[0] = h_ptx;
  d_result[1] = h_byte_perm;
}

__global__ void prmt_b32_rc16_kernel_3(uint32_t *d_result, uint32_t a) {
  static constexpr uint32_t sel = 3;
  static constexpr uint32_t b = 0x64646464;
  uint32_t h_ptx, h_byte_perm;

  asm volatile("prmt.b32.rc16 %0, %1, %2, %3;\n"
               : "=r"(h_ptx)
               : "r"(a), "n"(b), "n"(sel));

  h_byte_perm = custom_byte_level_permute_reference(a, b, sel, 6);

  d_result[0] = h_ptx;
  d_result[1] = h_byte_perm;
}

bool test_1(void) {

  uint32_t h_result[2];
  uint32_t *d_result;
  uint32_t a = 0x12345678; // Example input value

  cudaMalloc(&d_result, 2 * sizeof(uint32_t));
  prmt_b32_kernel<<<1, 1>>>(d_result, a);
  cudaMemcpy(h_result, d_result, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaFree(d_result);

  bool pass = true;
  if (h_result[0] != h_result[1]) {
    std::cout << "Verification passed: prmt_b32_kernel() matches prmt.b32"
              << std::endl;
    pass = false;
  }

  return pass;
}

bool test_2(void) {

  uint32_t h_result[2];
  uint32_t *d_result;
  uint32_t a = 0x12345678; // Example input value
  cudaMalloc(&d_result, 2 * sizeof(uint32_t));

  bool pass = true;

  prmt_b32_f4e_kernel_0<<<1, 1>>>(d_result, a);
  cudaDeviceSynchronize();
  cudaMemcpy(h_result, d_result, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  if (h_result[0] != h_result[1]) {
    std::cout << "Verification passed: prmt_b32_f4e_kernel_0() matches prmt.b32"
              << std::endl;
    pass = false;
  }

  prmt_b32_f4e_kernel_1<<<1, 1>>>(d_result, a);
  cudaDeviceSynchronize();
  cudaMemcpy(h_result, d_result, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  if (h_result[0] != h_result[1]) {
    std::cout << "Verification passed: prmt_b32_f4e_kernel_1() matches prmt.b32"
              << std::endl;
    pass = false;
  }

  prmt_b32_f4e_kernel_2<<<1, 1>>>(d_result, a);
  cudaDeviceSynchronize();
  cudaMemcpy(h_result, d_result, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  if (h_result[0] != h_result[1]) {
    std::cout << "Verification passed: prmt_b32_f4e_kernel_2() matches prmt.b32"
              << std::endl;
    pass = false;
  }

  prmt_b32_f4e_kernel_3<<<1, 1>>>(d_result, a);
  cudaDeviceSynchronize();
  cudaMemcpy(h_result, d_result, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  if (h_result[0] != h_result[1]) {
    std::cout << "Verification passed: prmt_b32_f4e_kernel_3() matches prmt.b32"
              << std::endl;
    pass = false;
  }

  cudaFree(d_result);

  return pass;
}

bool test_3(void) {

  uint32_t h_result[2];
  uint32_t *d_result;
  uint32_t a = 0x12345678; // Example input value

  cudaMalloc(&d_result, 2 * sizeof(uint32_t));

  bool pass = true;

  prmt_b32_b4e_kernel_0<<<1, 1>>>(d_result, a);
  cudaDeviceSynchronize();
  cudaMemcpy(h_result, d_result, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  if (h_result[0] != h_result[1]) {
    std::cout << "Verification passed: prmt_b32_b4e_kernel_0() matches prmt.b32"
              << std::endl;
    pass = false;
  }

  prmt_b32_b4e_kernel_1<<<1, 1>>>(d_result, a);
  cudaDeviceSynchronize();
  cudaMemcpy(h_result, d_result, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  if (h_result[0] != h_result[1]) {
    std::cout << "Verification passed: prmt_b32_b4e_kernel_1() matches prmt.b32"
              << std::endl;
    pass = false;
  }

  prmt_b32_f4e_kernel_2<<<1, 1>>>(d_result, a);
  cudaDeviceSynchronize();
  cudaMemcpy(h_result, d_result, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  if (h_result[0] != h_result[1]) {
    std::cout << "Verification passed: prmt_b32_f4e_kernel_2() matches prmt.b32"
              << std::endl;
    pass = false;
  }

  prmt_b32_b4e_kernel_3<<<1, 1>>>(d_result, a);
  cudaDeviceSynchronize();
  cudaMemcpy(h_result, d_result, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  if (h_result[0] != h_result[1]) {
    std::cout << "Verification passed: prmt_b32_b4e_kernel_3() matches prmt.b32"
              << std::endl;
    pass = false;
  }

  cudaFree(d_result);
  return pass;
}

bool test_4(void) {

  uint32_t h_result[2];
  uint32_t *d_result;
  uint32_t a = 0x12345678; // Example input value
  cudaMalloc(&d_result, 2 * sizeof(uint32_t));

  bool pass = true;

  prmt_b32_rc8_kernel_0<<<1, 1>>>(d_result, a);
  cudaDeviceSynchronize();
  cudaMemcpy(h_result, d_result, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  if (h_result[0] != h_result[1]) {
    std::cout << "Verification passed: prmt_b32_rc8_kernel_0() matches prmt.b32"
              << std::endl;
    pass = false;
  }

  prmt_b32_rc8_kernel_1<<<1, 1>>>(d_result, a);
  cudaDeviceSynchronize();
  cudaMemcpy(h_result, d_result, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  if (h_result[0] != h_result[1]) {
    std::cout << "Verification passed: prmt_b32_rc8_kernel_1() matches prmt.b32"
              << std::endl;
    pass = false;
  }

  prmt_b32_rc8_kernel_2<<<1, 1>>>(d_result, a);
  cudaDeviceSynchronize();
  cudaMemcpy(h_result, d_result, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  if (h_result[0] != h_result[1]) {
    std::cout << "Verification passed: prmt_b32_rc8_kernel_2() matches prmt.b32"
              << std::endl;
    pass = false;
  }

  prmt_b32_rc8_kernel_3<<<1, 1>>>(d_result, a);
  cudaDeviceSynchronize();
  cudaMemcpy(h_result, d_result, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  if (h_result[0] != h_result[1]) {
    std::cout << "Verification passed: prmt_b32_rc8_kernel_3() matches prmt.b32"
              << std::endl;
    pass = false;
  }

  cudaFree(d_result);
  return pass;
}

bool test_5(void) {

  uint32_t h_result[2];
  uint32_t *d_result;
  uint32_t a = 0x12345678; // Example input value

  cudaMalloc(&d_result, 2 * sizeof(uint32_t));

  bool pass = true;

  prmt_b32_ecl_kernel_0<<<1, 1>>>(d_result, a);
  cudaDeviceSynchronize();
  cudaMemcpy(h_result, d_result, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  if (h_result[0] != h_result[1]) {
    std::cout << "Verification passed: prmt_b32_ecl_kernel_0() matches prmt.b32"
              << std::endl;
    pass = false;
  }

  prmt_b32_ecl_kernel_1<<<1, 1>>>(d_result, a);
  cudaDeviceSynchronize();
  cudaMemcpy(h_result, d_result, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  if (h_result[0] != h_result[1]) {
    std::cout << "Verification passed: prmt_b32_ecl_kernel_1() matches prmt.b32"
              << std::endl;
    pass = false;
  }

  prmt_b32_ecl_kernel_2<<<1, 1>>>(d_result, a);
  cudaDeviceSynchronize();
  cudaMemcpy(h_result, d_result, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  if (h_result[0] != h_result[1]) {
    std::cout << "Verification passed: prmt_b32_ecl_kernel_2() matches prmt.b32"
              << std::endl;
    pass = false;
  }

  prmt_b32_ecl_kernel_3<<<1, 1>>>(d_result, a);
  cudaDeviceSynchronize();
  cudaMemcpy(h_result, d_result, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  if (h_result[0] != h_result[1]) {
    std::cout << "Verification passed: prmt_b32_ecl_kernel_3() matches prmt.b32"
              << std::endl;
    pass = false;
  }

  cudaFree(d_result);
  return pass;
}

bool test_6(void) {

  uint32_t h_result[2];
  uint32_t *d_result;
  uint32_t a = 0x12345678; // Example input value

  cudaMalloc(&d_result, 2 * sizeof(uint32_t));

  bool pass = true;

  prmt_b32_ecr_kernel_0<<<1, 1>>>(d_result, a);
  cudaDeviceSynchronize();
  cudaMemcpy(h_result, d_result, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  if (h_result[0] != h_result[1]) {
    std::cout << "Verification passed: prmt_b32_ecr_kernel_0() matches prmt.b32"
              << std::endl;
    pass = false;
  }

  prmt_b32_ecr_kernel_1<<<1, 1>>>(d_result, a);
  cudaDeviceSynchronize();
  cudaMemcpy(h_result, d_result, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  if (h_result[0] != h_result[1]) {
    std::cout << "Verification passed: prmt_b32_ecr_kernel_1() matches prmt.b32"
              << std::endl;
    pass = false;
  }

  prmt_b32_ecr_kernel_2<<<1, 1>>>(d_result, a);
  cudaDeviceSynchronize();
  cudaMemcpy(h_result, d_result, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  if (h_result[0] != h_result[1]) {
    std::cout << "Verification passed: prmt_b32_ecr_kernel_2() matches prmt.b32"
              << std::endl;
    pass = false;
  }

  prmt_b32_ecr_kernel_3<<<1, 1>>>(d_result, a);
  cudaDeviceSynchronize();
  cudaMemcpy(h_result, d_result, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  if (h_result[0] != h_result[1]) {
    std::cout << "Verification passed: prmt_b32_ecr_kernel_3() matches prmt.b32"
              << std::endl;
    pass = false;
  }

  cudaFree(d_result);
  return pass;
}

bool test_7(void) {
  uint32_t h_result[2];
  uint32_t *d_result;
  uint32_t a = 0x12345678; // Example input value

  cudaMalloc(&d_result, 2 * sizeof(uint32_t));

  bool pass = true;

  prmt_b32_rc16_kernel_0<<<1, 1>>>(d_result, a);
  cudaDeviceSynchronize();
  cudaMemcpy(h_result, d_result, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  if (h_result[0] != h_result[1]) {
    std::cout
        << "Verification passed: prmt_b32_rc16_kernel_0() matches prmt.b32"
        << std::endl;
    pass = false;
  }

  prmt_b32_rc16_kernel_1<<<1, 1>>>(d_result, a);
  cudaDeviceSynchronize();
  cudaMemcpy(h_result, d_result, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  if (h_result[0] != h_result[1]) {
    std::cout
        << "Verification passed: prmt_b32_rc16_kernel_1() matches prmt.b32"
        << std::endl;
    pass = false;
  }

  prmt_b32_rc16_kernel_2<<<1, 1>>>(d_result, a);
  cudaDeviceSynchronize();
  cudaMemcpy(h_result, d_result, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  if (h_result[0] != h_result[1]) {
    std::cout
        << "Verification passed: prmt_b32_rc16_kernel_2() matches prmt.b32"
        << std::endl;
    pass = false;
  }

  prmt_b32_rc16_kernel_3<<<1, 1>>>(d_result, a);
  cudaDeviceSynchronize();
  cudaMemcpy(h_result, d_result, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  if (h_result[0] != h_result[1]) {
    std::cout
        << "Verification passed: prmt_b32_rc16_kernel_3() matches prmt.b32"
        << std::endl;
    pass = false;
  }

  cudaFree(d_result);
  return pass;
}

int main() {
  TEST(test_1);
  TEST(test_2);
  TEST(test_3);
  TEST(test_4);
  TEST(test_5);
  TEST(test_6);
  TEST(test_7);
  return 0;
}

