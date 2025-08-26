#include <stdio.h>
#include <cstdint>
#include <iostream>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define NO_HALVES_PER_BLOCK 1024
using bf16_2 = __nv_bfloat162;

//Syntax:
//movmatrix.sync.aligned.shape.trans.type d, a;
//.shape = {.m8n8};
//.type = {.b16};#include <cuda_bf16.h>
// Only .m8n8.b16
int Shape_M = 8, Shape_N = 8;

#define TEST(FN)                                                               \
  {                                                                            \
    if (FN()) {                                                                \
      printf("Test " #FN " PASS\n");                                           \
    } else {                                                                   \
      printf("Test " #FN " FAIL\n");                                           \
      return 1;                                                                \
    }                                                                          \
  }


__device__ inline void movmatrix(bf16_2 &dst, const bf16_2 &src) {
  asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
               : "+r"(*(uint32_t *)(&dst))
               : "r"(*(uint32_t *)(&src)));
}

__device__ void ldmatrix(void *addr, volatile int *r) {
  unsigned int addr_int = __cvta_generic_to_shared(addr);

  asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
               : "=r"(r[0])
               : "r"(addr_int));
}

__global__ void test_movmatrix_fragment(half *input, half *output,
                                        const int TOTAL_ELEMENTS) {
  __shared__ half shared_data[NO_HALVES_PER_BLOCK];

  int lane_id = threadIdx.x % 32;

  // Load matrix inputs into shared memory.
  for (int i = threadIdx.x; i < TOTAL_ELEMENTS; i += blockDim.x) {
    shared_data[i] = input[i];
  }

  __syncthreads();

  int row_offset = 0;
  if (lane_id < 8) {
    row_offset += (8 * lane_id);
  }

  void *addr = shared_data + row_offset;

  
  volatile int rt;
  volatile int r;

  // Load matrix fragment from shared memory to register.
  ldmatrix(addr, &r);

  // Transpose matrix fragment in register.
  movmatrix(*(bf16_2 *)&rt, *(bf16_2 *)&r);
  int d_ind = 2 * lane_id;

  if (d_ind + 1 < TOTAL_ELEMENTS) {
    output[d_ind] = ((half *)(&rt))[0];
    output[d_ind + 1] = ((half *)(&rt))[1];
  }
}

bool ldmatrix_movmatrix_m8n8_b16() {
  const int TOTAL_ELEMENTS = Shape_M * Shape_N;
  // Allocate host memory for matrices
  half *h_input = new half[TOTAL_ELEMENTS];
  half *h_output = new half[TOTAL_ELEMENTS];
  half *exp_output = new half[TOTAL_ELEMENTS];

  // Allocate device memory for matrices
  half *d_input;
  half *d_output;
  cudaMalloc(&d_input, TOTAL_ELEMENTS * sizeof(half));
  cudaMalloc(&d_output, TOTAL_ELEMENTS * sizeof(half));
  cudaMemset(d_output, 0, TOTAL_ELEMENTS * sizeof(half));

  //// Initialize input matrix with some values
  for (int i = 0; i < TOTAL_ELEMENTS; i++) {
    h_input[i] = static_cast<half>(i);
  }

  // Copy input matrix to device
  cudaMemcpy(d_input, h_input, TOTAL_ELEMENTS * sizeof(half),
             cudaMemcpyHostToDevice);

  // Initialize expected matrix with some values
  int val = 0;

    for (int c = 0; c < Shape_N; c++) {
      for (int r = 0; r < Shape_M; r++) {
        exp_output[r * Shape_N + c] =
            static_cast<half>(val++);
      }
    }


  test_movmatrix_fragment<<<1, 32>>>(d_input, d_output, TOTAL_ELEMENTS);
  cudaDeviceSynchronize();

  // Copy output matrix back to host
  cudaMemcpy(h_output, d_output, TOTAL_ELEMENTS * sizeof(half),
             cudaMemcpyDeviceToHost);

  
  // Compare input & expected matrices data
  bool res = true;
  for (int r = 0; r < 8; r++) {
    for (int c = 0; c < 8; c++) {
      int index = r * 8 + c;

      float out = __half2float(h_output[index]);
      float exp_out = __half2float(exp_output[index]);
      if (out != exp_out) {
        std::cout << "Mismatch at index " << index << ": expected " << exp_out
                  << ", got " << out << std::endl;
        res = false;
      }
    }
  }
    
  delete[] h_input;
  delete[] h_output;
  cudaFree(d_input);
  cudaFree(d_output);

  return res;
}

int main() {
  TEST(ldmatrix_movmatrix_m8n8_b16);
}