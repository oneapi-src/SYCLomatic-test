#include <cuda.h>
#include <stdio.h>
#include <assert.h>

#define N 32

#ifdef __CUDA_ARCH__
__shared__ extern char cuda_shared_memory[];
#endif

__host__ __device__ static char *Env_cuda_shared_memory() {
#ifdef __CUDA_ARCH__
  return cuda_shared_memory;
#else
  return (char *)0;
#endif
}
__global__ void foo(char *buf_d) {
  char *p = Env_cuda_shared_memory();
  for (int i = 0; i != N; i++) {
    p[i] = '0' + i;
    __syncthreads();
    buf_d[i] = p[i];
  }
}

void test_d() {
  char *buf_d;
  cudaMalloc(&buf_d, N * sizeof(char));
  foo<<<32, 32, N * sizeof(char)>>>(buf_d);
  char buf_h[N] = {0};
  cudaMemcpy(buf_h, buf_d, N, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i != 10; i++)
    assert(buf_h[i] == '0' + i);
  printf("test_d done\n");
}

void test_h() {
  char *p = Env_cuda_shared_memory();
  assert(p == NULL);
  printf("test_h done\n");
}

int main() {
  test_d();
  test_h();
  return 0;
}
