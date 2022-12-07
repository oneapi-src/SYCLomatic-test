#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

inline std::vector<int> generate_random(size_t N, int Low, int High) {
  std::vector<int> Vec(N, 0);
  std::random_device Dev;
  std::mt19937 Rng(Dev());
  std::uniform_int_distribution<> Dist;
  for (size_t I = 0; I < N; ++I)
    Vec[I] = Dist(Rng);
  return Vec;
}

template <typename T> inline T *safe_device_malloc(size_t Num = 1) {
  T *Ptr = nullptr;
  cudaError_t Err = cudaMalloc<T>(&Ptr, sizeof(T) * Num);
  if (Err != cudaSuccess) {
    std::cerr << "cudaMalloc failed: " << cudaGetErrorString(Err) << std::endl;
    abort();
  }
  return Ptr;
}

inline void safe_host_copy_to_device(void *Dst, void *Src, size_t Size) {
  cudaError_t Err = cudaMemcpy(Dst, Src, Size, cudaMemcpyHostToDevice);
  if (Err != cudaSuccess) {
    std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(Err) << std::endl;
    abort();
  }
}

inline void safe_device_copy_to_host(void *Dst, void *Src, size_t Size) {
  cudaError_t Err = cudaMemcpy(Dst, Src, Size, cudaMemcpyDeviceToHost);
  if (Err != cudaSuccess) {
    std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(Err) << std::endl;
    abort();
  }
}

inline int *generate_device_random(size_t N, int Low, int High) {
  std::vector<int> Vec = generate_random(N, Low, High);
  int *Buffer = safe_device_malloc<int>(N);
  safe_host_copy_to_device(Buffer, Vec.data(), N * sizeof(int));
  return Buffer;
}

bool host() {
  size_t N = 1000;
  std::vector<int> Input = generate_random(N, 1, 100000);
  cub::ArgIndexInputIterator<int *> Iter(Input.data());
  for (size_t I = 0; I < N; ++I, ++Iter) {
    const auto &P = *Iter;
    if (Input[P.key] != P.value) {
      std::cerr << __func__ << " fun failed\n";
      return false;
    }
  }
  return true;
}

__global__ void device_kernel(int *Input, bool *Ret, size_t N) {
  cub::ArgIndexInputIterator<int *> Iter(Input);
  for (size_t I = 0; I < N; ++I, ++Iter) {
    const auto &P = *Iter;
    if (Input[P.key] != P.value) {
      *Ret = false;
      return;
    }
  }
  *Ret = true;
}

bool device() {
  bool HostRet;
  size_t N = 1000;
  int *Buffer = generate_device_random(N, 1, 10000);
  bool *Ret = safe_device_malloc<bool>(1);
  device_kernel<<<1, 1>>>(Buffer, Ret, N);
  safe_device_copy_to_host(&HostRet, Ret, 1);
  if (!HostRet) {
    std::cerr << __func__ << " fun failed\n";
      return false;
  }
  return true;
}

int main() {
  bool Result = true;
  Result = host() && Result;
  Result = device() && Result;
  if (!Result) {
    std::cout << "cub::ArgIndexInputIterator failed\n";
    return 1;
  }
  return 0;
}
