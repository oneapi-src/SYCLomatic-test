#include <cudnn.h>
#include <iostream>

int main() {
  int did_fail = 0;
  auto check_null_assignment = [&](auto &&arg, auto x) {
    arg = nullptr;
    if (arg) {
      std::cout << x << ": null assignment failed\n";
      did_fail = 1;
    }
  };
  {
    cudnnHandle_t d;
    cudnnCreate(&d);
    check_null_assignment(d, "cudnnHandle_t");
  }
  {
    cudnnTensorDescriptor_t d;
    cudnnCreateTensorDescriptor(&d);
    check_null_assignment(d, "cudnnTensorDescriptor_t");
  }
  {
    cudnnConvolutionDescriptor_t d;
    cudnnCreateConvolutionDescriptor(&d);
    check_null_assignment(d, "cudnnConvolutionDescriptor_t");
  }
  if (!did_fail)
    std::cout << "null assignment pass\n";
  return did_fail;
}
