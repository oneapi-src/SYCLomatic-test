#include <cuda_runtime.h>
#include <cuda/std/array>

template <class T>
__host__ __device__ void
test(T *res)
{
  cuda::std::array<T, 3> arr = {1, 2, 3.5};
  *(res) = arr.at(0); 
  *(res+1) = arr.at(1); 
  *(res+2) = arr.at(2);
  *(res+3) = *(arr.begin());
  *(res+4) = arr.size();
}

__global__ void test_global(float * res)
{
  test<float>(res);
}

int main(int, char **)
{
  
  float *floatRes = (float *)malloc(5 * sizeof(float));
  test<float>(floatRes);
  //test<double>(doubleRes);
  float *hostRes = (float *)malloc(5 * sizeof(float));
  float *deviceRes;
  cudaMalloc((float **)&deviceRes, 5 * sizeof(float));
  test_global<<<1, 1>>>(deviceRes);
  cudaMemcpy(hostRes, deviceRes, sizeof(float) * 5, cudaMemcpyDeviceToHost);
  cudaFree(deviceRes);

  for (int i = 0;i<5;++i){
    if(hostRes[i]!=floatRes[i]){
      free(hostRes);
      free(floatRes);
      return 1;
    }
  }
  free(hostRes);
  free(floatRes);
  return 0;

}
