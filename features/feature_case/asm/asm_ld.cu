// ====------ asm_ld.cu ----------------------------------- *- CUDA -* ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;
using bf16_2 = __nv_bfloat162;
using half_2 = __half2;


#define TEST(FUNC_CALL)                                                               \
  {                                                                            \
    if (FUNC_CALL) {                                                                \
      printf("Test " #FUNC_CALL " PASS\n");                                           \
    } else {                                                                   \
      printf("Test " #FUNC_CALL " FAIL\n");                                           \
      return 1;                                                                \
    }                                                                          \
  }

__device__ inline void load_global_short4(short4 &a, const short4 *addr) {
  short x, y, z, w;
  asm("ld.cg.global.v4.s16 {%0, %1, %2, %3}, [%4+0];"
      : "=h"(x), "=h"(y), "=h"(z), "=h"(w)
      : "l"(addr));
  a.x = x;
  a.y = y;
  a.z = z;
  a.w = w;
}

__global__ void test_kernel(short4 *d_out, const short4 *d_in) {
  short4 val;
  load_global_short4(val, d_in);
  *d_out = val;
}

__device__ inline void load_global_short2(short2 &a, const short2 *addr) {
  short x, y, z, w;
  asm("ld.cg.global.v2.s16 {%0, %1}, [%2+0];" : "=h"(x), "=h"(y) : "l"(addr));
  a.x = x;
  a.y = y;
}

__global__ void test_kernel(short2 *d_out, const short2 *d_in) {
  short2 val;
  load_global_short2(val, d_in);
  *d_out = val;
}

bool test_1() {
  short4 h_in = {1, 2, 3, 4};
  short4 h_out;
  short4 *d_in, *d_out;

  cudaMalloc(&d_in, sizeof(short4));
  cudaMalloc(&d_out, sizeof(short4));
  cudaMemcpy(d_in, &h_in, sizeof(short4), cudaMemcpyHostToDevice);

  test_kernel<<<1, 1>>>(d_out, d_in);
  cudaMemcpy(&h_out, d_out, sizeof(short4), cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);

  return (h_out.x == h_in.x && h_out.y == h_in.y && h_out.z == h_in.z &&
          h_out.w == h_in.w)
             ? true
             : false;
}

bool test_2() {
  short2 h_in = {1, 2};
  short2 h_out;
  short2 *d_in, *d_out;

  cudaMalloc(&d_in, sizeof(short2));
  cudaMalloc(&d_out, sizeof(short2));
  cudaMemcpy(d_in, &h_in, sizeof(short2), cudaMemcpyHostToDevice);

  test_kernel<<<1, 1>>>(d_out, d_in);
  cudaMemcpy(&h_out, d_out, sizeof(short2), cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);

  return (h_out.x == h_in.x && h_out.y == h_in.y) ? true : false;
}

__device__ __forceinline__ int ld_flag_volatile(int *flag_addr) {
  int flag;
  asm volatile("ld.volatile.global.u32 %0, [%1]; membar.gl;"
               : "=r"(flag)
               : "l"(flag_addr));
  return flag;
}

__global__ void test_ld_flag_acquire(int *flag_addr, int *out_value) {
  int val = ld_flag_volatile(flag_addr);
  *out_value = val;
}

bool test_3() {

  int h_flag_value = 999;
  int h_result = 0;

  int *d_flag_addr;
  int *d_result;

  cudaMalloc(&d_flag_addr, sizeof(int));
  cudaMalloc(&d_result, sizeof(int));

  cudaMemcpy(d_flag_addr, &h_flag_value, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(d_result, 0, sizeof(int));

  test_ld_flag_acquire<<<1, 1>>>(d_flag_addr, d_result);
  cudaDeviceSynchronize();

  cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_flag_addr);
  cudaFree(d_result);

  return (h_result == h_flag_value) ? true : false;
}


__device__ uint16_t ptx_ld_global_b16(uint16_t *src) {
  uint16_t dst;
  asm volatile("ld.global.b16 %0, [%1];" : "=h"(dst) : "l"(src));
  return dst;
}

__device__ uint16_t ptx_ld_shared_b16(uint16_t *src) {
  uint16_t dst;
  uint64_t addr = static_cast<uint64_t>(__cvta_generic_to_shared(src));
  asm volatile("ld.shared.b16 %0, [%1];" : "=h"(dst) : "l"(addr));
  return dst;
}

__global__ void test_kernel_ld_global_b16(uint16_t *d_src, uint16_t *d_dst) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  d_dst[idx] = ptx_ld_global_b16(&d_src[idx]);
}

__global__ void test_kernel_ld_shared_b16(uint16_t *d_src, uint16_t *d_dst,
                                       int N) {
  extern __shared__ uint16_t smem_b16[];

  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Copy data in global memory to shared memory
  if (idx < N) {
    smem_b16[threadIdx.x] = d_src[idx];
  }

  // Wait until the shared memory operatoins are done.
  __syncthreads(); 

  if (idx < N) {
    d_dst[idx] = ptx_ld_shared_b16(&smem_b16[threadIdx.x]);
  }
}

bool test_ptx_ld_bf16(bool global_mem) {
  const int N = 4;
  bf16 h_src[N] = {__float2bfloat16(1.5f), __float2bfloat16(2.0f),
                   __float2bfloat16(2.5f), __float2bfloat16(3.0f)};
  bf16 h_dst[N] = {0};

  bf16 *d_src = nullptr;
  bf16 *d_dst = nullptr;

  cudaMalloc(&d_src, N * sizeof(bf16));
  cudaMalloc(&d_dst, N * sizeof(bf16));

  cudaMemcpy(d_src, h_src, N * sizeof(bf16), cudaMemcpyHostToDevice);
  cudaMemset(d_dst, 0, N * sizeof(bf16));

  if (global_mem) {
    test_kernel_ld_global_b16<<<1, N>>>(reinterpret_cast<uint16_t *>(d_src),
                            reinterpret_cast<uint16_t *>(d_dst));
  } else {
    test_kernel_ld_shared_b16<<<1, N>>>(reinterpret_cast<uint16_t *>(d_src),
                            reinterpret_cast<uint16_t *>(d_dst), N);
  }
  cudaDeviceSynchronize();

  cudaMemcpy(h_dst, d_dst, N * sizeof(bf16), cudaMemcpyDeviceToHost);

  bool res = true;
  for (int i = 0; i < N; ++i) {
    if (h_dst[i] != h_src[i]) {
      res = false;
      printf("error: h_dst[%d] = %f (expected %f)\n", i,
             __bfloat162float(h_dst[i]), __bfloat162float(h_src[i]));
    }
    printf("h_dst[%d] = %f (expected %f)\n", i, __bfloat162float(h_dst[i]),
           __bfloat162float(h_src[i]));
  }

  cudaFree(d_src);
  cudaFree(d_dst);

  return res;
}

bool test_ptx_ld_half(bool global_mem) {
  const int N = 4;
  half h_src[N] = {__float2half(1.5f), __float2half(2.0f), __float2half(2.5f),
                   __float2half(3.0f)};
  half h_dst[N] = {0};

  half *d_src = nullptr;
  half *d_dst = nullptr;

  cudaMalloc(&d_src, N * sizeof(half));
  cudaMalloc(&d_dst, N * sizeof(half));

  cudaMemcpy(d_src, h_src, N * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemset(d_dst, 0, N * sizeof(half));

  if (global_mem) {
    test_kernel_ld_global_b16<<<1, N>>>(reinterpret_cast<uint16_t *>(d_src),
                                        reinterpret_cast<uint16_t *>(d_dst));
  } else {
    test_kernel_ld_shared_b16<<<1, N>>>(reinterpret_cast<uint16_t *>(d_src),
                                        reinterpret_cast<uint16_t *>(d_dst), N);
  }

  cudaDeviceSynchronize();

  cudaMemcpy(h_dst, d_dst, N * sizeof(bf16), cudaMemcpyDeviceToHost);

  bool res = true;
  for (int i = 0; i < N; ++i) {
    if (h_dst[i] != h_src[i]) {
      res = false;
      printf("error: h_dst[%d] = %f (expected %f)\n", i, __half2float(h_dst[i]),
             __half2float(h_src[i]));
    }
    printf("h_dst[%d] = %f (expected %f)\n", i, __half2float(h_dst[i]),
           __half2float(h_src[i]));
  }

  cudaFree(d_src);
  cudaFree(d_dst);

  return res;
}

__device__ uint32_t ptx_ld_global_b32(uint32_t *src) {
  uint32_t dst;
  uint64_t addr = reinterpret_cast<uint64_t>(src);
  asm volatile("ld.global.b32 %0, [%1];" : "=r"(dst) : "l"(addr));
  return dst;
}

__device__ uint32_t ptx_ld_shared_b32(uint32_t *src) {
  uint32_t dst;
  uint64_t addr = static_cast<uint64_t>(__cvta_generic_to_shared(src));
  asm volatile("ld.shared.b32 %0, [%1];" : "=r"(dst) : "l"(addr));
  return dst;
}

__global__ void test_kernel_ld_global_b32(uint32_t *d_src, uint32_t *d_dst) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  d_dst[idx] = ptx_ld_global_b32(&d_src[idx]);
}


__global__ void test_kernel_ld_shared_b32(uint32_t *d_src, uint32_t *d_dst,
                                          int N) {
  extern __shared__ uint32_t smem_b32[];

  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Copy data in global memory to shared memory
  if (idx < N) {
    smem_b32[threadIdx.x] = d_src[idx];
  }

  // Wait until the shared memory operatoins are done.
  __syncthreads();

  if (idx < N) {
    d_dst[idx] = ptx_ld_shared_b32(&smem_b32[threadIdx.x]);
  }
}

inline __nv_bfloat162 bf16_pair(float a, float b) {
  return __floats2bfloat162_rn(a, b);
}

bool test_ptx_ld_bf16_2(bool global_mem) {
  const int N = 4;
  bf16_2 h_src[N] = {bf16_pair(1.0f, 2.0f), bf16_pair(3.0f, 4.0f),
                     bf16_pair(5.0f, 6.0f), bf16_pair(7.0f, 8.0f)};
  bf16_2 h_dst[N] = {bf16_pair(0.0f, 0.0f), bf16_pair(0.0f, 0.0f),
                     bf16_pair(0.0f, 0.0f), bf16_pair(0.0f, 0.0f)};

  bf16_2 *d_src = nullptr;
  bf16_2 *d_dst = nullptr;

  cudaMalloc(&d_src, N * sizeof(bf16_2));
  cudaMalloc(&d_dst, N * sizeof(bf16_2));

  cudaMemcpy(d_src, h_src, N * sizeof(bf16_2), cudaMemcpyHostToDevice);
  cudaMemset(d_dst, 0, N * sizeof(bf16_2));

  if (global_mem) {
    test_kernel_ld_global_b32<<<1, N>>>(reinterpret_cast<uint32_t *>(d_src),
                              reinterpret_cast<uint32_t *>(d_dst));
  } else {
    test_kernel_ld_shared_b32<<<1, N>>>(reinterpret_cast<uint32_t *>(d_src),
                                        reinterpret_cast<uint32_t *>(d_dst), N);
  }

  cudaDeviceSynchronize();

  cudaMemcpy(h_dst, d_dst, N * sizeof(bf16_2), cudaMemcpyDeviceToHost);

  bool res = true;
  for (int i = 0; i < N; ++i) {
    float src_x = __bfloat162float(h_src[i].x); 
    float src_y = __bfloat162float(h_src[i].y); 
    float dst_x = __bfloat162float(h_dst[i].x); 
    float dst_y = __bfloat162float(h_dst[i].y); 
    if ((src_x != dst_x) && (src_y != dst_y)) {
      res = false;
      printf("error: h_dst[%d] = (%f, %f) (expected (%f, %f))\n", i, dst_x,
             dst_y, dst_x, dst_y);
    }
    printf("h_dst[%d] = (%f, %f) (expected (%f, %f))\n", i, dst_x, dst_y,
           dst_x, dst_y);
  }

  cudaFree(d_src);
  cudaFree(d_dst);

  return res;
}

inline __half2 half_pair(float a, float b) { return __floats2half2_rn(a, b); }

bool test_ptx_ld_half_2(bool global_mem) {
  const int N = 4;
  half_2 h_src[N] = {half_pair(1.0f, 2.0f), half_pair(3.0f, 4.0f),
                     half_pair(5.0f, 6.0f), half_pair(7.0f, 8.0f)};
  half_2 h_dst[N] = {half_pair(0.0f, 0.0f), half_pair(0.0f, 0.0f),
                     half_pair(0.0f, 0.0f), half_pair(0.0f, 0.0f)};

  half_2 *d_src = nullptr;
  half_2 *d_dst = nullptr;

  cudaMalloc(&d_src, N * sizeof(half_2));
  cudaMalloc(&d_dst, N * sizeof(half_2));

  cudaMemcpy(d_src, h_src, N * sizeof(half_2), cudaMemcpyHostToDevice);
  cudaMemset(d_dst, 0, N * sizeof(half_2));

  if (global_mem) {
    test_kernel_ld_global_b32<<<1, N>>>(reinterpret_cast<uint32_t *>(d_src),
                                        reinterpret_cast<uint32_t *>(d_dst));
  } else {
    test_kernel_ld_shared_b32<<<1, N>>>(reinterpret_cast<uint32_t *>(d_src),
                                        reinterpret_cast<uint32_t *>(d_dst), N);
  }

  cudaDeviceSynchronize();

  cudaMemcpy(h_dst, d_dst, N * sizeof(half_2), cudaMemcpyDeviceToHost);

  bool res = true;
  for (int i = 0; i < N; ++i) {
    float src_x = __half2float(h_src[i].x);
    float src_y = __half2float(h_src[i].y);
    float dst_x = __half2float(h_dst[i].x);
    float dst_y = __half2float(h_dst[i].y);
    if ((src_x != dst_x) && (src_y != dst_y)) {
      res = false;
      printf("error: h_dst[%d] = (%f, %f) (expected (%f, %f))\n", i, dst_x,
             dst_y, dst_x, dst_y);
    }
    printf("h_dst[%d] = (%f, %f) (expected (%f, %f))\n", i, dst_x, dst_y, dst_x,
           dst_y);
  }

  cudaFree(d_src);
  cudaFree(d_dst);

  return res;
}

 __device__ uint32_t ptx_ld_global_f32(float * src) {
    float dst;
    asm volatile("ld.global.f32 %0, [%1];" : "=f"(dst) : "l"(src));
    return dst;
 }

 __device__ uint32_t ptx_ld_shared_f32(float *src) {
    float dst;
    uint64_t addr = static_cast<uint64_t>(__cvta_generic_to_shared(src));
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(dst) : "l"(addr));
    return dst;
}

__global__ void test_kernel_ld_global_f32(float * d_src, float * d_dst) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    d_dst[idx] = ptx_ld_global_f32(&d_src[idx]);
}

__global__ void test_kernel_ld_shared_f32(float *d_src, float *d_dst, int N) {
    extern __shared__ float smem_f32[];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Copy data in global memory to shared memory
    if (idx < N) {
        smem_f32[threadIdx.x] = d_src[idx];
    }

    // Wait until the shared memory operatoins are done.
    __syncthreads();

    if (idx < N) {
        d_dst[idx] = ptx_ld_shared_f32(&smem_f32[threadIdx.x]);
    }
}

bool test_ptx_ld_f32(bool global_mem) {
    const int N = 4;
    float h_src[N] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_dst[N] = {0};

    float *d_src = nullptr;
    float *d_dst = nullptr;

    cudaMalloc(&d_src, N * sizeof(float));
    cudaMalloc(&d_dst, N * sizeof(float));

    cudaMemcpy(d_src, h_src, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_dst, 0, N * sizeof(float));

    if (global_mem) {
        test_kernel_ld_global_f32<<<1, N>>>(reinterpret_cast<float *>(d_src),
                                            reinterpret_cast<float *>(d_dst));
    } else {
        test_kernel_ld_shared_f32<<<1, N>>>(reinterpret_cast<float *>(d_src),
                                            reinterpret_cast<float *>(d_dst),
                                            N);
    }

    cudaDeviceSynchronize();

    cudaMemcpy(h_dst, d_dst, N * sizeof(float), cudaMemcpyDeviceToHost);

    bool res = true;
    for (int i = 0; i < N; ++i) {
        if (h_dst[i] != h_src[i]) {
          res = false;
          printf("error: h_dst[%d] = %f (expected %f)\n", i, h_dst[i], h_src[i]);
        }
        printf("h_dst[%d] = %f (expected %f)\n", i, h_dst[i], h_src[i]);
    }
    cudaFree(d_src);
    cudaFree(d_dst);

    return res;
  }

/* 
// ld.v2 and ld.v4 is not supported by auto migration yet.
// When they are support, will uncomment this part.

  __device__ void ptx_ld_global_v2_f32(const float2 *src, float2 &dst) {
    asm volatile("ld.global.v2.f32 {%0, %1}, [%2];"
                 : "=f"(dst.x), "=f"(dst.y)
                 : "l"(src));
  }

  __device__ void ptx_ld_shared_v2_f32(const float2 *src, float2 &dst) {
    uint64_t addr = static_cast<uint64_t>(__cvta_generic_to_shared(src));
    asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];"
                 : "=f"(dst.x), "=f"(dst.y)
                 : "l"(addr));
  }

  __global__ void test_kernel_ld_global_v2_f32(const float2 *d_src, float2 *d_dst) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    ptx_ld_global_v2_f32(&d_src[idx], d_dst[idx]);
  }

  __global__ void test_kernel_ld_shared_v2_f32(const float2 *d_src, float2 *d_dst, int N) {
    extern __shared__ float2 smem_v2_f32[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Copy data in global memory to shared memory
    if (idx < N) {
        smem_v2_f32[threadIdx.x] = d_src[idx];
    }

    // Wait until the shared memory operatoins are done.
    __syncthreads();

    if (idx < N) {
        ptx_ld_shared_v2_f32(&smem_v2_f32[threadIdx.x], d_dst[idx]);
    }
  }

  bool test_ptx_ld_v2_f32(bool global_mem) {
    const int N = 4;
    float2 h_src[N] = {{1.1f, 2.2f}, {3.3f, 4.4f}, {5.5f, 6.6f}, {7.7f, 8.8f}};
    float2 h_dst[N];

    float2 *d_src = nullptr;
    float2 *d_dst = nullptr;

    cudaMalloc(&d_src, N * sizeof(float2));
    cudaMalloc(&d_dst, N * sizeof(float2));

    cudaMemcpy(d_src, h_src, N * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemset(d_dst, 0, N * sizeof(float2));

    if (global_mem) {
        test_kernel_ld_global_v2_f32<<<1, N>>>(
            reinterpret_cast<float2 *>(d_src),
            reinterpret_cast<float2 *>(d_dst));
    } else {
        test_kernel_ld_shared_v2_f32<<<1, N>>>(
            reinterpret_cast<float2 *>(d_src),
            reinterpret_cast<float2 *>(d_dst),
                                            N);
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_dst, d_dst, N * sizeof(float2), cudaMemcpyDeviceToHost);

    bool res = true;
    for (int i = 0; i < N; ++i) {
        float src_x = h_src[i].x;
        float src_y = h_src[i].y;
        float dst_x = h_dst[i].x;
        float dst_y = h_dst[i].y;
        if ((src_x != dst_x) && (src_y != dst_y)) {
          res = false;
          printf("error: h_dst[%d] = (%f, %f) (expected (%f, %f))\n", i, dst_x,
                 dst_y, src_x, src_y);
        }
        printf("h_dst[%d] = (%f, %f) (expected (%f, %f))\n", i, dst_x, dst_y,
               src_x, src_y);
    }

    cudaFree(d_src);
    cudaFree(d_dst);

    return res;
  }

  __device__ void ptx_ld_global_v4_f32(const float4 *src, float4 &dst) {
    asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];\n"
                 : "=f"(dst.x), "=f"(dst.y), "=f"(dst.z), "=f"(dst.w)
                 : "l"(src));
  }

  __device__ void ptx_ld_shared_v4_f32(const float4 *src, float4 &dst) {
    uint64_t addr = static_cast<uint64_t>(__cvta_generic_to_shared(src));
    asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n"
                 : "=f"(dst.x), "=f"(dst.y), "=f"(dst.z), "=f"(dst.w)
                 : "l"(addr));
  }

  __global__ void test_kernel_ld_global_v4_f32(const float4 *d_src, float4 *d_dst) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    ptx_ld_global_v4_f32(&d_src[idx], d_dst[idx]);
  }

  __global__ void test_kernel_ld_shared_v4_f32(const float4 *d_src, float4 *d_dst, int N) {
    extern __shared__ float4 smem_v4_f32[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Copy data in global memory to shared memory
    if (idx < N) {
        smem_v4_f32[threadIdx.x] = d_src[idx];
    }

    // Wait until the shared memory operatoins are done.
    __syncthreads();

    if (idx < N) {
        ptx_ld_shared_v4_f32(&smem_v4_f32[threadIdx.x], d_dst[idx]);
    }
  }

bool test_ptx_ld_v4_f32(bool global_mem) {
    const int N = 4;
    float4 h_src[N] = {{1.1f, 2.2f, 3.3f, 4.4f},
                       {5.5f, 6.6f, 7.7f, 8.8f},
                       {9.9f, 10.1f, 11.11f, 12.12f},
                       {13.13f, 14.14f, 15.15f, 16.16f}};
    float4 h_dst[N];

    float4 *d_src = nullptr;
    float4 *d_dst = nullptr;

    cudaMalloc(&d_src, N * sizeof(float4));
    cudaMalloc(&d_dst, N * sizeof(float4));

    cudaMemcpy(d_src, h_src, N * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemset(d_dst, 0, N * sizeof(float4));


    if (global_mem) {
        test_kernel_ld_global_v4_f32<<<1, N>>>(
            reinterpret_cast<float4 *>(d_src),
            reinterpret_cast<float4 *>(d_dst));
    } else {
        test_kernel_ld_shared_v4_f32<<<1, N>>>(
            reinterpret_cast<float4 *>(d_src),
            reinterpret_cast<float4 *>(d_dst), N);
    }

    cudaDeviceSynchronize();

    cudaMemcpy(h_dst, d_dst, N * sizeof(float4), cudaMemcpyDeviceToHost);

    bool res = true;
    for (int i = 0; i < N; ++i) {
        float src_x = h_src[i].x;
        float src_y = h_src[i].y;
        float src_z = h_src[i].z;
        float src_w = h_src[i].w;
        float dst_x = h_dst[i].x;
        float dst_y = h_dst[i].y;
        float dst_z = h_dst[i].z;
        float dst_w = h_dst[i].w;
        if ((src_x != dst_x) && (src_y != dst_y) && (src_z != dst_z) &&
            (src_w != dst_w)) {
          res = false;
          printf("error: h_dst[%d] = (%f, %f, %f, %f) (expected (%f, %f, %f, "
                 "%f))\n",
                 i, dst_x, dst_y, dst_z, dst_w, src_x, src_y, src_z, src_w);
        }
        printf("h_dst[%d] = (%f, %f, %f, %f) (expected (%f, %f, %f, %f))\n", i, dst_x, dst_y,
               dst_x, dst_y, src_x, src_y, src_z, src_w);
    }

    cudaFree(d_src);
    cudaFree(d_dst);

    return res;
 }
*/
int main() {
    TEST(test_1);
    TEST(test_2);
    TEST(test_3);

#define GLOBAL_MEM true
#define SHARED_MEM false

    TEST(test_ptx_ld_bf16(GLOBAL_MEM)); 
    TEST(test_ptx_ld_bf16(SHARED_MEM)); 
    TEST(test_ptx_ld_half(GLOBAL_MEM)); 
    TEST(test_ptx_ld_half(SHARED_MEM));
    TEST(test_ptx_ld_bf16_2(GLOBAL_MEM));
    TEST(test_ptx_ld_bf16_2(SHARED_MEM));
    TEST(test_ptx_ld_half_2(GLOBAL_MEM));
    TEST(test_ptx_ld_half_2(SHARED_MEM));
    TEST(test_ptx_ld_f32(GLOBAL_MEM));
    TEST(test_ptx_ld_f32(SHARED_MEM));
   /* TEST(test_ptx_ld_v2_f32(GLOBAL_MEM));
    TEST(test_ptx_ld_v2_f32(SHARED_MEM));
    TEST(test_ptx_ld_v4_f32(GLOBAL_MEM));
    TEST(test_ptx_ld_v4_f32(SHARED_MEM));*/
    
  return 0;
}