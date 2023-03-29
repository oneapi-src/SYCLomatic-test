#include <cub/cub.cuh>

template <typename T> T *init(std::initializer_list<T> list) {
  T *p = nullptr;
  cudaMalloc<T>(&p, sizeof(T) * list.size());
  cudaMemcpy(p, list.begin(), sizeof(T) * list.size(), cudaMemcpyHostToDevice);
  return p;
}

__global__ void iadd3_kernel(int x, int y, int z, int *output) {
  *output = cub::IADD3(x, y, z);
}

bool iadd3(int x, int y, int z) {
  int output, *d_output = init({0});
  iadd3_kernel<<<1, 1>>>(x, y, z, d_output);
  cudaMemcpy(&output, d_output, sizeof(int), cudaMemcpyDeviceToHost);
  if (output != x + y + z) {
    std::cout << "cub::IADD3 test failed"
                 "\n";
    std::cout << "input: " << x << " " << y << " " << z << "\n";
    std::cout << "expected: " << output << "\n";
    std::cout << "result: " << x + y + z << "\n";
    return false;
  }
  return true;
}

bool test_iadd3() {
  return iadd3(1, 2, 3) && iadd3(4, 5, 6) && iadd3(9991, 12, 7) &&
         iadd3(0, 1, 0);
}

int main() {
  if (!test_iadd3()) {
    return 1;
  }
  return 0;
}
