#include <iostream>
#include <cstdint>
#include <cuda_runtime.h>

struct stream_wrapper {
  cudaStream_t s;
  stream_wrapper() : s(0) {}
  void set(cudaStream_t t) { s = t; }
  void set_int(uintptr_t p) { s = (cudaStream_t) p; }
};

int main() {
  stream_wrapper wr {};
  int x, y;
  int res = 0;
  int i = 0;

  auto run = [&]() {
    x = -1;
    y = 42;
    cudaMemcpyAsync(&x, &y, sizeof(int), cudaMemcpyDefault, wr.s);
    cudaStreamSynchronize(wr.s);
    if (x != y) {
      std::cout << "default stream fail " << i << "\n";
      res = 1;
    }
    ++i;
  };

  run();

  wr.set(cudaStreamDefault);
  run();
  
  wr.set(cudaStreamLegacy);
  run();

  wr.set(cudaStreamPerThread);
  run();

  wr.set_int(0);
  run();

  cudaStream_t s;
  cudaStreamCreate(&s);
  wr.set_int((uintptr_t) s);
  run();

  if (!res) {
    std::cout << "default stream success\n";
  }

  return res;  
}
