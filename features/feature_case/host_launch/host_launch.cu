#include<cuda_runtime.h>
#include<iostream>

void hostCallback(void *userData) {
  const char *msg = static_cast<const char*>(userData);
  std::cout << "Host callback executed. Message: " << msg << std::endl;
}

int main() {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaError_t err;
  const char *message = "Kernel execution finished.";

  err = cudaLaunchHostFunc(stream, hostCallback, (void*)message);

  cudaHostFn_t fn = hostCallback;
  cudaLaunchHostFunc(stream, fn, (void*)message);

  cudaStreamDestroy(stream);
  return 0;
}

