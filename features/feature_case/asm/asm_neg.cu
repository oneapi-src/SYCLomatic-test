#include <cuda_fp16.h>
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

__device__ inline void negate_half2(__half2 *addr) {
  unsigned reg[2];

  reg[0] = *reinterpret_cast<unsigned int *>(addr);
  asm volatile("neg.f16x2 %0, %0;" : "+r"(reg[0]));
  *reinterpret_cast<unsigned int *>(addr) = reg[0];
}

__global__ void test_negate_half2(__half2 *d_output) {
  __half2 val = __halves2half2(__float2half(1.5f), __float2half(-2.5f));
  *d_output = val;
  negate_half2(d_output);
}

bool run_test() {
  __half2 *d_output;
  __half2 h_output;

  // Allocate memory on GPU
  cudaMalloc(&d_output, sizeof(__half2));

  // Launch kernel
  test_negate_half2<<<1, 1>>>(d_output);
  cudaDeviceSynchronize();

  // Copy result back to host
  cudaMemcpy(&h_output, d_output, sizeof(__half2), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_output);

  // Validate results
  float x = __half2float(h_output.x);
  float y = __half2float(h_output.y);

  if (x == -1.5f && y == 2.5f) {
    return true;
  } else {
    std::cout << "Test FAILED: got (" << x << ", " << y << ")" << std::endl;
    return false;
  }
}

int main() {
  TEST(run_test);
  return 0;
}
