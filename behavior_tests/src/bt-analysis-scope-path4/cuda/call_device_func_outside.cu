#include "../inc/utils.cuh"

__global__ void kernel(float *input)
{
  float sum = 0;
  __shared__ float smem[128];
  float total_sum = BlockReduceSum(sum, smem);
}

void foo()
{
  float *input = NULL;
  kernel<<<1, 128>>>(input);
}